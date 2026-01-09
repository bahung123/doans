from __future__ import annotations
import os
import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING
import re
import requests
from pydantic import Field
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

if TYPE_CHECKING:
    from core.rag.vector_store import ChromaVectorDB

logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass
class RetrievalConfig:
    rerank_api_base_url: str = "https://api.siliconflow.com/v1"
    rerank_model: str = "Qwen/Qwen3-Reranker-4B"
    rerank_top_n: int = 10
    initial_k: int = 25  # Reduced to minimize reranker time
    top_k: int = 5
    vector_weight: float = 0.5
    bm25_weight: float = 0.5



_retrieval_config: RetrievalConfig | None = None


def get_retrieval_config() -> RetrievalConfig:
    global _retrieval_config
    if _retrieval_config is None:
        _retrieval_config = RetrievalConfig()
    return _retrieval_config


class SiliconFlowReranker(BaseDocumentCompressor):
    api_key: str = Field(default="")
    api_base_url: str = Field(default="")
    model: str = Field(default="")
    top_n: Optional[int] = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents or not self.api_key:
            return list(documents)
        
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.api_base_url}/rerank",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": [doc.page_content for doc in documents],
                        "top_n": self.top_n or len(documents),
                    },
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                
                if "results" not in data:
                    return list(documents)
                
                reranked: List[Document] = []
                for result in data["results"]:
                    doc = documents[result["index"]]
                    meta = dict(doc.metadata or {})
                    meta["rerank_score"] = result["relevance_score"]
                    reranked.append(Document(page_content=doc.page_content, metadata=meta))
                
                return reranked
                
            except Exception as e:
                if "rate" in str(e).lower() and attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Rerank error: {e}")
                    return list(documents)
        
        return list(documents)




class Retriever:
    def __init__(self, vector_db: "ChromaVectorDB", use_reranker: bool = True):
        self._vector_db = vector_db
        self._config = get_retrieval_config()
        self._reranker: Optional[SiliconFlowReranker] = None
        
        self._vector_retriever = self._vector_db.vectorstore.as_retriever(
            search_kwargs={"k": self._config.initial_k}
        )
        
        # Lazy-load BM25 - only initialize when needed
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._bm25_initialized = False
        self._ensemble_retriever: Optional[EnsembleRetriever] = None
        
        # BM25 cache path (persist to disk)
        from pathlib import Path
        persist_dir = getattr(self._vector_db.config, 'persist_dir', None)
        if persist_dir:
            self._bm25_cache_path = Path(persist_dir) / "bm25_cache.pkl"
        else:
            self._bm25_cache_path = None
        
        if use_reranker:
            self._reranker = self._init_reranker()
        
        logger.info("Retriever initialized")

    
    def _save_bm25_cache(self, bm25: BM25Retriever) -> None:
        """Save BM25 retriever to disk for fast loading."""
        if not self._bm25_cache_path:
            return
        try:
            import pickle
            with open(self._bm25_cache_path, 'wb') as f:
                pickle.dump(bm25, f)
            logger.info(f"BM25 cache saved to {self._bm25_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save BM25 cache: {e}")
    
    def _load_bm25_cache(self) -> Optional[BM25Retriever]:
        if not self._bm25_cache_path or not self._bm25_cache_path.exists():
            return None
            
        try:
            import pickle
            import time
            start = time.time()
            with open(self._bm25_cache_path, 'rb') as f:
                bm25 = pickle.load(f)
            bm25.k = self._config.initial_k
            logger.info(f"BM25 loaded from cache in {time.time() - start:.2f}s")
            return bm25
        except Exception as e:
            logger.warning(f"Failed to load BM25 cache: {e}")
            return None


    
    def _init_bm25(self) -> Optional[BM25Retriever]:
        if self._bm25_initialized:
            return self._bm25_retriever
        
        self._bm25_initialized = True
        
        # Try loading from cache first
        cached = self._load_bm25_cache()
        if cached:
            self._bm25_retriever = cached
            return cached
        
        # Build from scratch
        try:
            import time
            start = time.time()
            logger.info("Building BM25 index from documents...")
            
            docs = self._vector_db.get_all_documents()
            if not docs:
                logger.warning("No documents found for BM25")
                return None
            
            lc_docs = [
                Document(page_content=d["content"], metadata=d.get("metadata", {}))
                for d in docs
            ]
            bm25 = BM25Retriever.from_documents(lc_docs)
            bm25.k = self._config.initial_k
            
            self._bm25_retriever = bm25
            logger.info(f"BM25 built with {len(docs)} docs in {time.time() - start:.2f}s")
            
            # Save to cache for next time
            self._save_bm25_cache(bm25)
            
            return bm25
        except Exception as e:
            logger.error(f"Failed to init BM25: {e}")
            return None

    
    def _get_ensemble_retriever(self) -> EnsembleRetriever:
        """Get or create ensemble retriever (lazy-loaded)."""
        if self._ensemble_retriever is not None:
            return self._ensemble_retriever
        
        bm25 = self._init_bm25()
        if bm25:
            self._ensemble_retriever = EnsembleRetriever(
                retrievers=[self._vector_retriever, bm25],
                weights=[self._config.vector_weight, self._config.bm25_weight]
            )
        else:
            self._ensemble_retriever = EnsembleRetriever(
                retrievers=[self._vector_retriever], 
                weights=[1.0]
            )
        return self._ensemble_retriever

    
    def _init_reranker(self) -> Optional[SiliconFlowReranker]:
        api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
        if not api_key:
            return None
        return SiliconFlowReranker(
            api_key=api_key,
            api_base_url=self._config.rerank_api_base_url,
            model=self._config.rerank_model,
            top_n=self._config.rerank_top_n,
        )
    
    def _build_final(self):
        ensemble = self._get_ensemble_retriever()
        if self._reranker:
            return ContextualCompressionRetriever(
                base_compressor=self._reranker,
                base_retriever=ensemble
            )
        return ensemble

    
    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None
    
    def _to_result(self, doc: Document, rank: int, **extra) -> Dict[str, Any]:
        metadata = doc.metadata or {}
        content = doc.page_content
        
        # Small-to-Big: If this is a summary node, swap with parent (raw table)
        if metadata.get("is_table_summary") and metadata.get("parent_id"):
            parent = self._vector_db.get_parent_node(metadata["parent_id"])
            if parent:
                content = parent.get("content", content)
                # Merge metadata, keeping summary info for debugging
                metadata = {
                    **parent.get("metadata", {}),
                    "original_summary": doc.page_content[:200],
                    "swapped_from_summary": True,
                }
        
        return {
            "id": metadata.get("id"),
            "content": content,
            "metadata": metadata,
            "final_rank": rank,
            **extra,
        }

    
    def vector_search(
        self, text: str, *, k: int | None = None, where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not text.strip():
            return []
        k = k or self._config.top_k
        results = self._vector_db.vectorstore.similarity_search_with_score(text, k=k, filter=where)
        return [self._to_result(doc, i + 1, distance=score) for i, (doc, score) in enumerate(results)]
    
    def bm25_search(self, text: str, *, k: int | None = None) -> List[Dict[str, Any]]:
        if not text.strip():
            return []
        
        bm25 = self._init_bm25()  # Lazy-load BM25
        if not bm25:
            return self.vector_search(text, k=k)
        
        k = k or self._config.top_k
        bm25.k = k
        results = bm25.invoke(text)
        return [self._to_result(doc, i + 1) for i, doc in enumerate(results[:k])]
    
    def hybrid_search(
        self, text: str, *, k: int | None = None, initial_k: int | None = None
    ) -> List[Dict[str, Any]]:
        if not text.strip():
            return []
        
        k = k or self._config.top_k
        if initial_k:
            self._vector_retriever.search_kwargs["k"] = initial_k
            bm25 = self._init_bm25()
            if bm25:
                bm25.k = initial_k
        
        # Dùng ensemble_retriever (lazy-loaded, KHÔNG có reranker)
        ensemble = self._get_ensemble_retriever()
        results = ensemble.invoke(text)
        return [self._to_result(doc, i + 1) for i, doc in enumerate(results[:k])]
    
    def search_with_rerank(
        self,
        text: str,
        *,
        k: int | None = None,
        where: Optional[Dict[str, Any]] = None,
        initial_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        import time
        
        if not text.strip():
            return []
        
        k = k or self._config.top_k
        initial_k = initial_k or self._config.initial_k
        
        # Có filter -> dùng vector search + manual rerank
        if where:
            results = self._vector_db.vectorstore.similarity_search(text, k=initial_k, filter=where)
            if self._reranker:
                results = self._reranker.compress_documents(results, text)
            return [
                self._to_result(doc, i + 1, rerank_score=doc.metadata.get("rerank_score"))
                for i, doc in enumerate(results[:k])
            ]
        
        # Build final retriever (lazy-loaded ensemble + reranker)
        if initial_k:
            self._vector_retriever.search_kwargs["k"] = initial_k
            bm25 = self._init_bm25()
            if bm25:
                bm25.k = initial_k
        
        ensemble = self._get_ensemble_retriever()
        ensemble_results = ensemble.invoke(text)
        
        if self._reranker:
            results = self._reranker.compress_documents(ensemble_results, text)
        else:
            results = ensemble_results
        
        return [
            self._to_result(doc, i + 1, rerank_score=doc.metadata.get("rerank_score"))
            for i, doc in enumerate(results[:k])
        ]


    
    def flexible_search(
        self,
        text: str,
        *,
        mode: RetrievalMode | str = RetrievalMode.HYBRID_RERANK,
        k: int | None = None,
        initial_k: int | None = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not text.strip():
            return []
        
        if isinstance(mode, str):
            try:
                mode = RetrievalMode(mode.lower())
            except ValueError:
                mode = RetrievalMode.HYBRID_RERANK
        
        k = k or self._config.top_k
        initial_k = initial_k or self._config.initial_k
        
        if mode == RetrievalMode.VECTOR_ONLY:
            return self.vector_search(text, k=k, where=where)
        elif mode == RetrievalMode.BM25_ONLY:
            return self.bm25_search(text, k=k)
        elif mode == RetrievalMode.HYBRID:
            if where:
                return self.vector_search(text, k=k, where=where)
            return self.hybrid_search(text, k=k, initial_k=initial_k)
        else:  # HYBRID_RERANK
            return self.search_with_rerank(text, k=k, where=where, initial_k=initial_k)

    # Legacy alias
    query = vector_search
