from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from openai import OpenAI
import os
import torch


class RAGChain:
    def __init__(self, embedding_model, vector_store, llm_provider="huggingface"):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_provider = llm_provider

        if llm_provider == "huggingface":
            model_name = "google/flan-t5-base"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Use CPU on Streamlit Cloud unless you know GPU is available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

        elif llm_provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        else:
            raise ValueError(f"Unsupported llm_provider: {llm_provider}")

    def build_prompt(self, question: str, retrieved_docs: list[dict]) -> str:
        context_parts = []

        for doc in retrieved_docs:
            content = doc["content"]
            context_parts.append(content)

        context_text = "\n\n".join(context_parts)

        prompt = (
            "Use the context below to answer the question.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "If the answer is present in the context, answer in 1-2 sentences.\n"
            "If the answer is not present, say: I don't have enough information.\n\n"
            "Answer:"
        )
        return prompt

    def generate_huggingface(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs, max_new_tokens=150, temperature=0.2, do_sample=False
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    def generate_openai(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    def ask(self, question: str, top_k: int = 3) -> dict:
        query_embedding = self.embedding_model.embed_query(question)
        relevant_docs = self.vector_store.search(query_embedding, top_k=top_k)

        if not relevant_docs:
            return {
                "answer": "I don't have enough information.",
                "sources": [],
                "scores": [],
            }

        prompt = self.build_prompt(question, relevant_docs)

        if self.llm_provider == "huggingface":
            answer = self.generate_huggingface(prompt)
        elif self.llm_provider == "openai":
            answer = self.generate_openai(prompt)
        else:
            raise ValueError(f"Unsupported llm_provider: {self.llm_provider}")

        return {
            "answer": answer,
            "sources": [
                {
                    "file_name": doc["metadata"].get("file_name"),
                    "chunk_index": doc["metadata"].get("chunk_index"),
                    "preview": doc["content"][:200],
                }
                for doc in relevant_docs
            ],
            "scores": [float(doc["score"]) for doc in relevant_docs],
        }
