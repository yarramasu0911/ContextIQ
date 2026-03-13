from transformers import pipeline
from openai import OpenAI
import os


class RAGChain:
    def __init__(self, embedding_model, vector_store, llm_provider="huggingface"):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_provider = llm_provider

        if llm_provider == "huggingface":
            self.generator = pipeline(
                "text2text-generation", model="google/flan-t5-large", max_length=1024
            )
        elif llm_provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def build_prompt(self, question: str, retrieved_docs: list[dict]) -> str:
        context_parts = []

        for i, doc in enumerate(retrieved_docs, start=1):
            # file_name = doc["metadata"].get("file_name", "unknown")
            # chunk_index = doc["metadata"].get("chunk_index", "unknown")
            content = doc["content"]

            context_parts.append(f"{content}")

        context_text = "\n\n".join(context_parts)

        prompt = (
            f"Use the context below to answer the question.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"If the answer is present in the context, answer in 1-2 sentences.\n"
            f"If the answer is not present, say: I don't have enough information.\n\n"
            f"Answer:"
        )
        print(prompt)
        return prompt

    def generate_huggingface(self, prompt: str) -> str:
        result = self.generator(prompt)
        return result[0]["generated_text"]

    def generate_openai(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return response.choices[0].message.content

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
