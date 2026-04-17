from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from openai import OpenAI
import os
import torch

from .retrieval import ChunkRegistry, HybridRetriever, Reranker
from .query_transformer import rewrite_with_history, hyde, multi_query
from .cache import SemanticCache
from .faithfulness import check_grounding, annotate_with_citations


class RAGChain:
    def __init__(
        self,
        embedding_model,
        vector_store,
        llm_provider: str = "huggingface",
        registry: ChunkRegistry = None,
        enable_rerank: bool = True,
        enable_cache: bool = True,
        enable_query_rewrite: bool = None,
        enable_hyde: bool = False,
        enable_multi_query: bool = False,
        enable_faithfulness: bool = True,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_provider = llm_provider

        if llm_provider == "huggingface":
            model_name = "google/flan-t5-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        elif llm_provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported llm_provider: {llm_provider}")

        # Query-time enhancements. flan-t5-base is too weak for query
        # transforms — only turn them on by default when running on OpenAI.
        if enable_query_rewrite is None:
            enable_query_rewrite = llm_provider == "openai"
        self.enable_query_rewrite = enable_query_rewrite
        self.enable_hyde = enable_hyde and llm_provider == "openai"
        self.enable_multi_query = enable_multi_query and llm_provider == "openai"
        self.enable_faithfulness = enable_faithfulness

        self.registry = registry if registry is not None else ChunkRegistry()
        self.reranker = Reranker() if enable_rerank else None
        self.retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            registry=self.registry,
            reranker=self.reranker,
        )
        self.cache = SemanticCache(embedding_model) if enable_cache else None

    # ------------------------------------------------------------------ LLM
    def _llm_call(self, prompt: str) -> str:
        """Unified LLM entry point used by query transforms + main generate."""
        if self.llm_provider == "huggingface":
            return self.generate_huggingface(prompt)
        return self.generate_openai(prompt)

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
            length_penalty=1.5,
            min_length=20,
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

    # ---------------------------------------------------------------- prompt
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

        history_text = ""
        if chat_history and len(chat_history) > 1:
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
                return (
                    f"Previous conversation:\n{history_text}\n\n"
                    f"Context: {context_text}\n\n"
                    f"Question: {question}\n\n"
                    "Answer the question using the context and conversation history:"
                )
            return (
                f"Answer the question based on the context.\n\n"
                f"Context: {context_text}\n\n"
                f"Question: {question}\n\nAnswer:"
            )

        if history_text:
            return (
                "You are a helpful document assistant. Use the sources and "
                "conversation history to answer.\n\n"
                f"Conversation history:\n{history_text}\n\n"
                f"Sources:\n{context_text}\n\n"
                f"Question: {question}\n\n"
                "Answer concisely in 2-3 sentences:"
            )
        return (
            "You are a helpful document assistant. Answer based only on the "
            "sources.\n\n"
            f"Sources:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely in 2-3 sentences:"
        )

    # ----------------------------------------------------- casual / greetings
    def is_casual_message(self, question: str) -> bool:
        casual_patterns = [
            "hello", "hi ", "hey", "how are you", "good morning",
            "good evening", "good afternoon", "what's up", "whats up",
            "how's it going", "thank you", "thanks", "bye", "goodbye",
            "see you", "who are you", "what are you", "what can you do",
            "help me", "how do i use", "how does this work",
        ]
        q = question.lower().strip()
        for pattern in casual_patterns:
            if pattern in q:
                return True
        if len(q.split()) <= 3 and "?" not in q:
            return True
        return False

    def get_casual_response(self, question: str) -> str:
        q = question.lower().strip()
        if any(g in q for g in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
            return "Hello! I'm ContextIQ, your document assistant. Upload a document and ask me anything about it!"
        if any(t in q for t in ["thank", "thanks"]):
            return "You're welcome! Feel free to ask more questions about your documents."
        if any(b in q for b in ["bye", "goodbye", "see you"]):
            return "Goodbye! Your documents will be here when you come back."
        if any(w in q for w in ["who are you", "what are you"]):
            return "I'm ContextIQ — an AI document assistant. I can answer questions about documents you upload, summarize them, and help you find specific information."
        if any(h in q for h in ["what can you do", "help", "how do i use", "how does this work"]):
            return (
                "Here's what I can do:\n"
                "1. **Upload** a PDF or TXT document using the sidebar\n"
                "2. **Ask questions** about the document content\n"
                "3. **Summarize** your documents\n"
                "4. **Multi-turn chat** — I remember our conversation context\n\n"
                "Try uploading a document to get started!"
            )
        if "how are you" in q or "how's it going" in q:
            return "I'm doing great, thanks for asking! Ready to help you with your documents. What would you like to know?"
        return "I'm designed to help with document-related questions. Upload a PDF or TXT file and ask me anything about it!"

    # -------------------------------------------------------- registry helper
    def register_chunks(
        self,
        chunks: list,
        filename: str,
        owner_username: str = "default",
        owner_role: str = "user",
        visibility: str = "private",
    ):
        """Mirror uploaded chunks into BM25 + invalidate the semantic cache."""
        texts = [c["content"] if isinstance(c, dict) else str(c) for c in chunks]
        self.registry.add(
            texts,
            filename=filename,
            owner_username=owner_username,
            owner_role=owner_role,
            visibility=visibility,
        )
        # Any upload can affect what a given viewer sees, so nuke the whole
        # semantic cache rather than trying to invalidate per-user.
        if self.cache is not None:
            self.cache.invalidate()

    def clear(
        self,
        viewer_username: str = "default",
        viewer_role: str = "admin",
        scope: str = "mine",
    ):
        self.registry.clear(
            viewer_username=viewer_username, viewer_role=viewer_role, scope=scope
        )
        if self.cache is not None:
            self.cache.invalidate()

    # -------------------------------------------------------------------- ask
    def ask(
        self,
        question: str,
        top_k: int = 3,
        viewer_username: str = "default",
        viewer_role: str = "admin",
        chat_history: list = None,
    ) -> dict:
        user_id = viewer_username  # cache keying
        # 1. Casual short-circuit — no retrieval, no LLM
        if self.is_casual_message(question):
            return {
                "answer": self.get_casual_response(question),
                "sources": [],
                "cached": False,
                "path": "casual",
            }

        # 2. Summary shortcut
        summary_keywords = ["summarize", "summary", "overview", "what is this about", "main points"]
        if any(kw in question.lower() for kw in summary_keywords):
            result = self.summarize(
                top_k=5,
                viewer_username=viewer_username,
                viewer_role=viewer_role,
            )
            result["path"] = "summary"
            return result

        # 3. Semantic cache lookup
        if self.cache is not None:
            hit = self.cache.lookup(question, user_id=user_id)
            if hit is not None:
                hit["path"] = "cache"
                return hit

        # 4. Query transforms (rewrite with history, HyDE, multi-query)
        retrieval_query = question
        extra_queries: list[str] = []

        if self.enable_query_rewrite and chat_history:
            retrieval_query = rewrite_with_history(
                question, chat_history, self._llm_call
            )

        if self.enable_hyde:
            hyp = hyde(retrieval_query, self._llm_call)
            if hyp:
                extra_queries.append(hyp)

        if self.enable_multi_query:
            variants = multi_query(retrieval_query, self._llm_call, n=2)
            # multi_query already includes the original at index 0
            extra_queries.extend(v for v in variants[1:] if v)

        # 5. Hybrid retrieve + rerank
        ranked = self.retriever.retrieve(
            retrieval_query,
            top_k=top_k,
            viewer_username=viewer_username,
            viewer_role=viewer_role,
            extra_queries=extra_queries,
        )

        if not ranked:
            return {
                "answer": "No relevant information found. Please upload documents first.",
                "sources": [],
                "cached": False,
                "path": "no_results",
            }

        best_score = ranked[0][1]
        # RRF / cross-encoder scores are on very different scales than raw
        # cosine, so only apply the low-relevance guard when we're clearly in
        # cosine range (i.e., dense-only fallback).
        if self.reranker is None and best_score < 0.15:
            return {
                "answer": "I couldn't find relevant information in your documents for this question. Try rephrasing or ask something related to your uploaded documents.",
                "sources": [],
                "cached": False,
                "path": "low_relevance",
            }

        docs_for_prompt = [{"content": text} for text, _score, _fn in ranked]
        prompt = self.build_prompt(question, docs_for_prompt, chat_history=chat_history)
        answer = self._llm_call(prompt)

        sources = [
            {"text": text[:300], "document": filename, "score": float(score)}
            for text, score, filename in ranked
        ]

        result = {
            "answer": answer,
            "sources": sources,
            "cached": False,
            "path": "rag",
            "retrieval_query": retrieval_query if retrieval_query != question else None,
        }

        # 6. Faithfulness / grounding check
        if self.enable_faithfulness:
            grounding = check_grounding(answer, sources, self.embedding_model)
            result["grounding"] = grounding
            annotated = annotate_with_citations(grounding)
            if annotated:
                result["answer_with_citations"] = annotated

        # 7. Cache the answer
        if self.cache is not None:
            self.cache.store(question, result, user_id=user_id)

        return result

    def summarize(
        self,
        top_k: int = 5,
        viewer_username: str = "default",
        viewer_role: str = "admin",
    ) -> dict:
        query_embedding = self.embedding_model.embed_query("main topics and key points")
        context_chunks = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            viewer_username=viewer_username,
            viewer_role=viewer_role,
        )
        context_text = " ".join([text for text, _score, _filename in context_chunks])

        prompt = f"""Summarize the following document in a clear and concise way.

        Document:
        {context_text}

        Summary:"""

        answer = self._llm_call(prompt)
        return {"answer": answer, "chunks_used": len(context_chunks)}
