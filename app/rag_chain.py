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

    def build_prompt(
        self, question: str, context_chunks: list, chat_history: list = None
    ) -> str:
        context_parts = []
        for i, doc in enumerate(context_chunks):
            text = doc["content"] if isinstance(doc, dict) else doc
            context_parts.append(f"[Source {i + 1}]: {text}")

        context_text = "\n\n".join(context_parts)

        if len(context_text) > 1200:
            context_text = context_text[:1200]

        # Build conversation history string
        history_text = ""
        if chat_history and len(chat_history) > 1:
            # Include last 3 exchanges (6 messages)
            recent = chat_history[-6:]
            history_parts = []
            for msg in recent:
                if msg["role"] == "user":
                    history_parts.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    history_parts.append(f"Assistant: {msg['content']}")
            history_text = "\n".join(history_parts)

        if self.llm_provider == "huggingface":
            if history_text:
                prompt = (
                    f"Previous conversation:\n{history_text}\n\n"
                    f"Context: {context_text}\n\n"
                    f"Question: {question}\n\n"
                    "Answer the question using the context and conversation history:"
                )
            else:
                prompt = f"Answer the question based on the context.\n\nContext: {context_text}\n\nQuestion: {question}\n\nAnswer:"
        else:
            if history_text:
                prompt = (
                    "You are a helpful document assistant. Use the sources and conversation history to answer.\n\n"
                    f"Conversation history:\n{history_text}\n\n"
                    f"Sources:\n{context_text}\n\n"
                    f"Question: {question}\n\n"
                    "Answer concisely in 2-3 sentences:"
                )
            else:
                prompt = (
                    "You are a helpful document assistant. Answer based only on the sources.\n\n"
                    f"Sources:\n{context_text}\n\n"
                    f"Question: {question}\n\n"
                    "Answer concisely in 2-3 sentences:"
                )
        return prompt

    def generate_huggingface(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.5,  # encourages longer answers
            min_length=20,  # forces at least 20 tokens
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return (
            answer.strip()
            if answer.strip()
            else "The model could not generate an answer. Try rephrasing your question."
        )

    def generate_openai(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    def is_casual_message(self, question: str) -> bool:
        """Detect if the message is casual chat, not document-related."""
        casual_patterns = [
            "hello",
            "hi ",
            "hey",
            "how are you",
            "good morning",
            "good evening",
            "good afternoon",
            "what's up",
            "whats up",
            "how's it going",
            "thank you",
            "thanks",
            "bye",
            "goodbye",
            "see you",
            "who are you",
            "what are you",
            "what can you do",
            "help me",
            "how do i use",
            "how does this work",
        ]
        question_lower = question.lower().strip()

        # Check if it matches casual patterns
        for pattern in casual_patterns:
            if pattern in question_lower:
                return True

        # Very short messages are usually casual
        if len(question_lower.split()) <= 3 and "?" not in question_lower:
            return True

        return False

    def get_casual_response(self, question: str) -> str:
        """Generate appropriate response for casual messages."""
        question_lower = question.lower().strip()

        if any(
            g in question_lower
            for g in [
                "hello",
                "hi",
                "hey",
                "good morning",
                "good afternoon",
                "good evening",
            ]
        ):
            return "Hello! I'm ContextIQ, your document assistant. Upload a document and ask me anything about it!"

        elif any(t in question_lower for t in ["thank", "thanks"]):
            return (
                "You're welcome! Feel free to ask more questions about your documents."
            )

        elif any(b in question_lower for b in ["bye", "goodbye", "see you"]):
            return "Goodbye! Your documents will be here when you come back."

        elif any(w in question_lower for w in ["who are you", "what are you"]):
            return "I'm ContextIQ — an AI document assistant. I can answer questions about documents you upload, summarize them, and help you find specific information."

        elif any(
            h in question_lower
            for h in ["what can you do", "help", "how do i use", "how does this work"]
        ):
            return (
                "Here's what I can do:\n"
                "1. **Upload** a PDF or TXT document using the sidebar\n"
                "2. **Ask questions** about the document content\n"
                "3. **Summarize** your documents\n"
                "4. **Multi-turn chat** — I remember our conversation context\n\n"
                "Try uploading a document to get started!"
            )

        elif "how are you" in question_lower or "how's it going" in question_lower:
            return "I'm doing great, thanks for asking! Ready to help you with your documents. What would you like to know?"

        else:
            return "I'm designed to help with document-related questions. Upload a PDF or TXT file and ask me anything about it!"

    def ask(
        self,
        question: str,
        top_k: int = 3,
        user_id: str = "default",
        chat_history: list = None,
    ) -> dict:
        # Check for casual messages first
        if self.is_casual_message(question):
            return {"answer": self.get_casual_response(question), "sources": []}

        # Check for summary requests
        summary_keywords = [
            "summarize",
            "summary",
            "overview",
            "what is this about",
            "main points",
        ]
        if any(keyword in question.lower() for keyword in summary_keywords):
            return self.summarize(top_k=5, user_id=user_id)

        # Normal RAG flow
        query_embedding = self.embedding_model.embed_query(question)
        relevant_chunks = self.vector_store.search(
            query_embedding, top_k=top_k, user_id=user_id
        )

        if not relevant_chunks:
            return {
                "answer": "No relevant information found. Please upload documents first.",
                "sources": [],
            }

        # Check if retrieved chunks are actually relevant (low similarity = not relevant)
        best_score = relevant_chunks[0][1]
        if best_score < 0.3:  # cosine similarity too low
            return {
                "answer": "I couldn't find relevant information in your documents for this question. Try rephrasing or ask something related to your uploaded documents.",
                "sources": [],
            }

        docs_for_prompt = [
            {"content": text} for text, score, filename in relevant_chunks
        ]
        prompt = self.build_prompt(question, docs_for_prompt, chat_history=chat_history)

        if self.llm_provider == "huggingface":
            answer = self.generate_huggingface(prompt)
        elif self.llm_provider == "openai":
            answer = self.generate_openai(prompt)

        return {
            "answer": answer,
            "sources": [
                {"text": text[:300], "document": filename, "score": float(score)}
                for text, score, filename in relevant_chunks
            ],
        }

    def summarize(self, top_k: int = 5, user_id: str = "default") -> dict:
        query_embedding = self.embedding_model.embed_query("main topics and key points")
        context_chunks = self.vector_store.search(
            query_embedding, top_k=top_k, user_id=user_id
        )

        # Extract text from tuples
        context_text = " ".join([text for text, score, filename in context_chunks])

        prompt = f"""Summarize the following document in a clear and concise way.

        Document:
        {context_text}

        Summary:"""

        if self.llm_provider == "huggingface":
            answer = self.generate_huggingface(prompt)
        elif self.llm_provider == "openai":
            answer = self.generate_openai(prompt)

        return {"answer": answer, "chunks_used": len(context_chunks)}
