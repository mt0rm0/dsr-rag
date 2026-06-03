"""Core RAGTool class — vector store management, retrieval, and LLM interaction."""

import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import logging
from collections import deque
from typing import Any, Deque, Generator, List, cast

import chromadb
import requests
from chromadb.utils import embedding_functions
from tqdm import tqdm
from yake import KeywordExtractor

import ragsst.parameters as p
from ragsst.utils import hash_file, list_files, read_file, split_text

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(format=os.getenv("LOG_FORMAT", "%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", logging.INFO))
os.makedirs(p.LOG_DIR, exist_ok=True)
logger.addHandler(logging.FileHandler(os.path.join(p.LOG_DIR, p.LOG_FILE), mode="w+"))


# ---------------------------------------------------------------------------
# RAGTool
# ---------------------------------------------------------------------------


class RAGTool:
    """Local RAG system backed by ChromaDB and an Ollama LLM.

    Args:
        model:            Ollama model name (default: first in ``LLM_CHOICES``).
        llm_base_url:     Base URL for the Ollama API.
        data_path:        Directory containing documents to ingest.
        embedding_model:  Sentence-transformer model name.
        collection_name:  ChromaDB collection to use.
    """

    def __init__(
        self,
        model: str = p.LLM_CHOICES[0],
        llm_base_url: str = p.LLMBASEURL,
        data_path: str = p.DATA_PATH,
        embedding_model: str = p.EMBEDDING_MODELS[0],
        collection_name: str = p.COLLECTION_NAME,
    ) -> None:
        self.model = model
        self.llm_base_url = llm_base_url
        self.data_path = data_path
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.max_conversation_length = p.CONVERSATION_LENTGH
        self.conversation: Deque = deque(maxlen=self.max_conversation_length)
        self.rag_conversation: Deque = deque(maxlen=self.max_conversation_length)

        self.vs_client = chromadb.PersistentClient(
            path=p.VECTOR_DB_PATH,
            settings=chromadb.Settings(allow_reset=True),
        )
        self.embedding_func = cast(
            Any,
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model,
                trust_remote_code=True,
            ),
        )

        if p.KEYWORD_SEARCH or p.FILTER_BY_KEYWORD:
            self.kw_extractor = KeywordExtractor(
                lan="auto", n=1, dedupLim=0.9, windowsSize=1, top=1
            )

    # =========================================================================
    # LLM (Ollama)
    # =========================================================================

    def llm_generate(
        self, prompt: str, top_k: int = 5, top_p: float = 0.9, temp: float = 0.2
    ) -> str:
        """Send a single prompt to the LLM and return the response text."""
        url = self.llm_base_url + "/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temp, "top_p": top_p, "top_k": top_k},
        }
        try:
            r = requests.post(url, json=data)
            response_dic = json.loads(r.text)
            response = response_dic.get("response", "")
            return response if response else response_dic.get("error", "Check Ollama settings")
        except Exception as e:
            logger.error(f"llm_generate exception: {e}")
            return ""

    def llm_chat(
        self, user_message: str, top_k: int = 5, top_p: float = 0.9, temp: float = 0.5
    ) -> str:
        """Send a message in a multi-turn conversation and return the reply."""
        url = self.llm_base_url + "/chat"
        self.conversation.append({"role": "user", "content": user_message})
        data = {
            "model": self.model,
            "messages": list(self.conversation),
            "stream": False,
            "options": {"temperature": temp, "top_p": top_p, "top_k": top_k},
        }
        try:
            r = requests.post(url, json=data)
            response_dic = json.loads(r.text)
            response = response_dic.get("message", {})
            self.conversation.append(response)
            return response.get("content", "")
        except Exception as e:
            logger.error(f"llm_chat exception: {e}")
            return ""

    def list_local_models(self) -> List[str]:
        """Return the names of all models available in the local Ollama instance."""
        url = self.llm_base_url + "/tags"
        try:
            r = requests.get(url)
            response_dic = json.loads(r.text)
            return [model.get("name") for model in response_dic.get("models", [])]
        except Exception as e:
            logger.error(f"list_local_models exception: {e}")
            return []

    def pull_model(self, model_name: str) -> Generator[str, None, None]:
        """Download a model from Ollama. Yields status strings as it progresses."""
        url = self.llm_base_url + "/pull"
        try:
            r = requests.post(url, json={"name": model_name}, stream=True)
            r.raise_for_status()
            for content in r.iter_lines():
                if content:
                    yield f"Status: {json.loads(content).get('status')}"
        except Exception as e:
            logger.error(f"pull_model exception: {e}")

    # =========================================================================
    # Vector store
    # =========================================================================

    def set_collection(self, collection_name: str, embedding_model: str | None = None) -> None:
        """Load or create a ChromaDB collection."""
        self.set_collection_name(collection_name)
        if embedding_model is not None:
            self.set_embeddings_model(embedding_model)
        self.collection = self.vs_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
        )
        logger.info(
            f"Set collection: {self.collection_name}. Embedding model: {self.embedding_model}"
        )

    def make_collection(
        self,
        data_path: str,
        collection_name: str,
        skip_included_files: bool = True,
        consider_content: bool = True,
    ) -> None:
        """Ingest documents from *data_path* into a ChromaDB collection.

                Files already present in the collection are skipped unless their
                content has changed (when *consider_content* is True).

                Args:
                    data_path:            Directory containing .txt, .pdf, or .docx files.
                    collection_name:      Target ChromaDB collection.
                    skip_included_files:  Skip files whose name is already in the DB.
                    consider_content:     Re-ingest files whose conteprint(f'{"":42}', '     '.join([f'S{i}' for i in range(len(sentences))]))
        for i, s in enumerate(sentences):
            sims = [f'{cos_sim(embeddings[i], embeddings[j]):.3f}' for j in range(len(sentences))]
            print(f'S{i} {s[:38]:38}', '  '.join(sims))nt has changed.
        """
        logger.info(f"Documents path: {data_path}")
        self.set_collection(collection_name)

        files = list_files(data_path, extensions=(".txt", ".pdf", ".docx"))
        logger.info(f"{len(files)} files found.")

        if skip_included_files:
            existing_meta = self.collection.get(include=["metadatas"]).get("metadatas", [])
            sources = {m.get("source") for m in existing_meta}
            files_hashes = (
                {m.get("file_hash") for m in existing_meta} if consider_content else set()
            )

        for f in files:
            _, file_name = os.path.split(f)
            file_hash = hash_file(f) if consider_content else None

            if skip_included_files and file_name in sources:
                if not consider_content or file_hash in files_hashes:
                    logger.info(f"{file_name} already in vector DB — skipping.")
                    continue
                logger.info(f"Content changed for {file_name} — updating.")
                self.collection.delete(where={"source": file_name})

            logger.info(f"Reading and splitting {file_name}...")
            text = read_file(f)
            chunks = split_text(text)
            logger.info(f"{len(chunks)} chunks created.")

            for i, chunk in tqdm(enumerate(chunks, 1), total=len(chunks), desc=file_name):
                metadata: dict = {"source": file_name, "part": i}
                if consider_content and file_hash:
                    metadata["file_hash"] = file_hash
                self.collection.add(
                    documents=chunk,
                    ids=f"id{file_name[:-4]}.{i}",
                    metadatas=metadata,
                )

        logger.info(f"Collections: {self.list_collections_names_with_info()}")

    # =========================================================================
    # Retrieval
    # =========================================================================

    def retrieve_with_metadata(
        self, query: str = "", nresults: int = 2, sim_th: float | None = None
    ) -> str:
        """Return retrieved chunks with relevance scores and source info.

        Intended for the "Semantic Retrieval" tab in the GUI.
        """
        query_result = self.collection.query(query_texts=query, n_results=nresults)
        docs_selection = []
        for i in range(len(query_result["ids"][0])):
            sim = round(1 - query_result["distances"][0][i], 2)
            if sim_th is not None and sim < sim_th:
                continue
            doc = query_result["documents"][0][i]
            meta = query_result["metadatas"][0][i]
            docs_selection.append(
                "\n".join(
                    [
                        doc,
                        f"Relevance: {sim}",
                        f"Source: {meta.get('source')} (part {meta.get('part')})",
                    ]
                )
            )
        if not docs_selection:
            return "Relevant passage not found. Try lowering the relevance threshold."
        return "\n-----------------\n\n".join(docs_selection)

    def get_relevant_text(
        self,
        query: str = "",
        nresults: int = 2,
        sim_th: float | None = None,
        keyword_filter: bool = p.FILTER_BY_KEYWORD,
        keyword_search: bool = p.KEYWORD_SEARCH,
    ) -> str:
        """Return concatenated relevant chunks for *query*.

        Applies optional similarity thresholding, keyword post-filtering, and
        keyword fallback search as configured in :mod:`ragsst.parameters`.
        """
        query_result = self.collection.query(query_texts=query, n_results=nresults)

        if sim_th is not None:
            filtered = self._filter_query_by_similarity(query_result, sim_th)

            if filtered:
                if keyword_filter:
                    kw = self.kw_extractor.extract_keywords(query)[0][0]
                    kw_filtered = self._filter_query_by_keyword(filtered, kw)
                    if kw_filtered:
                        logger.debug("Semantic retrieval filtered by keyword.")
                        filtered = kw_filtered
                query_result = filtered

            elif keyword_search:
                kw = self.kw_extractor.extract_keywords(query)[0][0]
                logger.debug(f"No semantic results — falling back to keyword: {kw}")
                query_result = self.collection.query(
                    query_texts="", n_results=nresults, where_document={"$contains": kw}
                )
            else:
                logger.info("No results found for query.")
                return ""

        relevant_docs = query_result["documents"][0]
        if relevant_docs:
            logger.info(f"Sources: {', '.join(self._get_sources(query_result))}")
        return "\n".join(relevant_docs)

    # =========================================================================
    # Prompt construction
    # =========================================================================

    def get_context_prompt(self, query: str, context: str) -> str:
        """Build a RAG prompt from a *query* and retrieved *context*."""
        return (
            "Use the following context to answer the query at the end. "
            "Keep the answer as concise as possible.\n"
            f"Context:\n{context}"
            f"\nQuery:\n{query}"
        )

    def get_condenser_prompt(self, query: str, chat_history: Deque) -> str:
        """Build a prompt that rephrases a follow-up *query* as a standalone question."""
        history = "\n".join(list(chat_history))
        return (
            "Given the following chat history and a follow up query, rephrase the follow up "
            "query to be a standalone query. Just create the standalone query without "
            "commentary. Use the same language."
            f"\nChat history:\n{history}"
            f"\nFollow Up Query: {query}"
            "\nStandalone Query:"
        )

    # =========================================================================
    # RAG query / chat
    # =========================================================================

    def rag_query(
        self,
        user_msg: str,
        sim_th: float,
        nresults: int,
        top_k: int,
        top_p: float,
        temp: float,
    ) -> str:
        """Single-turn RAG: retrieve context then generate a response."""
        relevant_text = self.get_relevant_text(user_msg, nresults=nresults, sim_th=sim_th)
        if not relevant_text:
            return "Relevant passage not found. Try lowering the relevance threshold."
        prompt = self.get_context_prompt(user_msg, relevant_text)
        return self.llm_generate(prompt, top_k=top_k, top_p=top_p, temp=temp)

    def rag_chat(
        self,
        user_msg: str,
        sim_th: float,
        nresults: int,
        top_k: int,
        top_p: float,
        temp: float,
        history: Any = None,
    ) -> str:
        """Multi-turn RAG chat. Follow-up questions are condensed before retrieval."""
        no_context_msg = "Relevant passage not found. Try lowering the relevance threshold."

        if not self.rag_conversation:
            relevant_text = self.get_relevant_text(user_msg, nresults=nresults, sim_th=sim_th)
            if not relevant_text:
                return no_context_msg
            self.rag_conversation.append("Query: " + user_msg)
            response = self.llm_generate(
                self.get_context_prompt(user_msg, relevant_text),
                top_k=top_k,
                top_p=top_p,
                temp=temp,
            )
            self.rag_conversation.append("Answer: " + response)
            return response

        standalone_query = self.llm_generate(
            self.get_condenser_prompt(user_msg, self.rag_conversation),
            top_k=top_k,
            top_p=top_p,
            temp=temp,
        )
        relevant_text = self.get_relevant_text(standalone_query, nresults=nresults, sim_th=sim_th)
        if not relevant_text:
            return no_context_msg
        response = self.llm_generate(
            self.get_context_prompt(standalone_query, relevant_text),
            top_k=top_k,
            top_p=top_p,
            temp=temp,
        )
        self.rag_conversation.append("Query:\n" + standalone_query)
        self.rag_conversation.append("Answer:\n" + response)
        return response

    def chat(
        self, user_msg: str, top_k: int, top_p: float, temp: float, history: Any = None
    ) -> str:
        """Plain LLM chat without document context."""
        return self.llm_chat(user_msg, top_k=top_k, top_p=top_p, temp=temp)

    # =========================================================================
    # Utility / GUI helpers
    # =========================================================================

    def setup_vec_store(self, collection_name: str = p.COLLECTION_NAME) -> None:
        """Initialise the vector store on startup.

        If *data_path* contains documents and the vector DB is empty, ingests
        them automatically. Otherwise loads the first available collection.
        """
        if self._check_initdb_conditions():
            self.make_collection(self.data_path, collection_name)
        else:
            collections = self.vs_client.list_collections()
            if collections:
                logger.info(f"Available collections: {self.list_collections_names_with_info()}")
                self.set_collection(
                    collections[0].name,
                    collections[0].metadata.get("embedding_model"),
                )
                if not self.collection.peek(limit=1).get("ids"):
                    logger.info("Collection is empty. Populate it or choose another one.")
            else:
                self.set_collection(collection_name)
                logger.warning("Database is empty. Use Make/Update Database.")

    def _check_initdb_conditions(self) -> bool:
        return (
            os.path.exists(self.data_path)
            and os.listdir(self.data_path)
            and (
                not os.path.exists(p.VECTOR_DB_PATH)
                or not [f.path for f in os.scandir(p.VECTOR_DB_PATH) if f.is_dir()]
            )
        )

    def set_model(self, llm: str) -> None:
        self.model = llm
        logger.info(f"Model: {self.model}")

    def set_embeddings_model(self, emb_model: str) -> None:
        self.embedding_model = emb_model
        self.embedding_func = cast(
            Any,
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model, trust_remote_code=True
            ),
        )

    def set_data_path(self, data_path: str) -> None:
        self.data_path = data_path

    def set_collection_name(self, collection_name: str) -> None:
        self.collection_name = collection_name

    def list_collections_names(self) -> List[str]:
        return [c.name for c in self.vs_client.list_collections()]

    def list_collections_names_with_info(self) -> str:
        return ", ".join(
            f"{c.name} ({c.metadata.get('embedding_model', '')})"
            for c in self.vs_client.list_collections()
        )

    def delete_collection(self, collection_name: str) -> None:
        self.vs_client.delete_collection(collection_name)
        logger.info(f"{collection_name} removed.")
        collections = self.vs_client.list_collections()
        if collections:
            self.set_collection(
                collections[0].name,
                collections[0].metadata.get("embedding_model"),
            )

    def clean_database(self) -> None:
        """Delete all collections and entries from the vector store."""
        self.vs_client.reset()
        self.vs_client.clear_system_cache()
        logger.info("Database cleared.")

    def clear_chat_hist(self) -> None:
        self.conversation.clear()

    def clear_ragchat_hist(self) -> None:
        self.rag_conversation.clear()

    def _filter_query_by_similarity(self, query_result: dict, sim_th: float) -> dict:
        similarities = [round(1 - d, 2) for d in query_result["distances"][0]]
        relevant_docs = [
            doc for doc, s in zip(query_result["documents"][0], similarities) if s >= sim_th
        ]
        if not relevant_docs:
            return {}
        query_result["documents"][0] = relevant_docs
        query_result["metadatas"][0] = [
            meta for meta, s in zip(query_result["metadatas"][0], similarities) if s >= sim_th
        ]
        return query_result

    def _filter_query_by_keyword(self, query_result: dict, keyword: str) -> dict:
        kw = keyword.lower()
        relevant_docs = [d for d in query_result["documents"][0] if kw in d.lower()]
        if not relevant_docs:
            return {}
        query_result["metadatas"][0] = [
            meta
            for meta, doc in zip(query_result["metadatas"][0], query_result["documents"][0])
            if kw in doc.lower()
        ]
        query_result["documents"][0] = relevant_docs
        return query_result

    def _get_sources(self, query_result: dict) -> set:
        return {meta.get("source") for meta in query_result["metadatas"][0]}
