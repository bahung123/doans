from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from langchain_core.documents import Document
from langchain_chroma import Chroma
from core.hash_file.hash_file import HashProcessor

logger = logging.getLogger(__name__)


@dataclass
class ChromaConfig:
    def _default_persist_dir() -> str:
        repo_root = Path(__file__).resolve().parents[2]
        return str((repo_root / "data" / "chroma").resolve())

    persist_dir: str = field(default_factory=_default_persist_dir)
    collection_name: str = "hust_rag_collection"


class ChromaVectorDB:
    def __init__(
        self,
        embedder: Any,
        config: ChromaConfig | None = None,
    ):
        self.embedder = embedder
        self.config = config or ChromaConfig()
        self._hasher = HashProcessor(verbose=False)
        
        # Storage for parent nodes (not embedded, used for Small-to-Big retrieval)
        # Persist to JSON file in same directory as ChromaDB
        self._parent_nodes_path = Path(self.config.persist_dir) / "parent_nodes.json"
        self._parent_nodes: Dict[str, Dict[str, Any]] = self._load_parent_nodes()

        self._vs = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embedder,
            persist_directory=self.config.persist_dir,
        )
        logger.info(f"ChromaVectorDB initialized: {self.config.collection_name}")
    
    def _load_parent_nodes(self) -> Dict[str, Dict[str, Any]]:
        if self._parent_nodes_path.exists():
            try:
                with open(self._parent_nodes_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} parent nodes from {self._parent_nodes_path}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load parent nodes: {e}")
        return {}
    
    def _save_parent_nodes(self) -> None:
        """Save parent nodes to JSON file."""
        try:
            self._parent_nodes_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._parent_nodes_path, 'w', encoding='utf-8') as f:
                json.dump(self._parent_nodes, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self._parent_nodes)} parent nodes to {self._parent_nodes_path}")
        except Exception as e:
            logger.warning(f"Failed to save parent nodes: {e}")

    @property
    def collection(self):
        return getattr(self._vs, "_collection", None)

    @property
    def vectorstore(self):
        return self._vs

    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (metadata or {}).items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                out[str(k)] = v
            elif isinstance(v, (list, tuple, set, dict)):
                out[str(k)] = json.dumps(v, ensure_ascii=False)
            else:
                out[str(k)] = str(v)
        return out
    
    def _normalize_doc(self, doc: Any) -> Dict[str, Any]:
        # Nếu đã là dict
        if isinstance(doc, dict):
            return doc
        
        # Nếu là TextNode/BaseNode từ llama_index
        if hasattr(doc, "get_content") and hasattr(doc, "metadata"):
            return {
                "content": doc.get_content(),
                "metadata": dict(doc.metadata) if doc.metadata else {},
            }
        
        # Nếu là Document từ langchain
        if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            return {
                "content": doc.page_content,
                "metadata": dict(doc.metadata) if doc.metadata else {},
            }
        
        raise TypeError(f"Unsupported document type: {type(doc)}")

    def _to_documents(self, docs: Sequence[Any], ids: Sequence[str]) -> List[Document]:
        out: List[Document] = []
        for d, doc_id in zip(docs, ids):
            normalized = self._normalize_doc(d)
            md = self._flatten_metadata(normalized.get("metadata", {}) or {})
            md.setdefault("id", doc_id)
            out.append(Document(page_content=normalized.get("content", ""), metadata=md))
        return out

    def _doc_id(self, doc: Any) -> str:
        normalized = self._normalize_doc(doc)
        md = normalized.get("metadata") or {}
        key = {
            "source_file": md.get("source_file"),
            "header_path": md.get("header_path"),
            "chunk_index": md.get("chunk_index"),
            "content": normalized.get("content"),
        }
        return self._hasher.get_string_hash(str(key))

    def add_documents(
        self,
        docs: Sequence[Dict[str, Any]],
        *,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 128,
    ) -> int:
        if not docs:
            return 0

        if ids is not None and len(ids) != len(docs):
            raise ValueError("ids length must match docs length")

        # Separate parent nodes (not embedded) from regular nodes
        regular_docs = []
        regular_ids = []
        parent_count = 0
        
        for i, d in enumerate(docs):
            normalized = self._normalize_doc(d)
            md = normalized.get("metadata", {}) or {}
            doc_id = ids[i] if ids else self._doc_id(d)
            
            if md.get("is_parent"):
                # Store parent node separately (for Small-to-Big retrieval)
                parent_id = md.get("node_id", doc_id)
                self._parent_nodes[parent_id] = {
                    "id": parent_id,
                    "content": normalized.get("content", ""),
                    "metadata": md,
                }
                parent_count += 1
            else:
                regular_docs.append(d)
                regular_ids.append(doc_id)
        
        if parent_count > 0:
            logger.info(f"Stored {parent_count} parent nodes (not embedded)")
            self._save_parent_nodes()  # Persist to disk
        
        if not regular_docs:
            return parent_count

        bs = max(1, batch_size)
        total = 0
        
        for start in range(0, len(regular_docs), bs):
            batch = regular_docs[start : start + bs]
            batch_ids = regular_ids[start : start + bs]
            lc_docs = self._to_documents(batch, batch_ids)

            try:
                self._vs.add_documents(lc_docs, ids=batch_ids)
            except TypeError:
                texts = [d.page_content for d in lc_docs]
                metas = [d.metadata for d in lc_docs]
                self._vs.add_texts(texts=texts, metadatas=metas, ids=batch_ids)
            total += len(batch)

        logger.info(f"Added {total} documents to vector store")
        return total + parent_count

    def upsert_documents(
        self,
        docs: Sequence[Dict[str, Any]],
        *,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 128,
    ) -> int:
        if not docs:
            return 0

        if ids is not None and len(ids) != len(docs):
            raise ValueError("ids length must match docs length")

        # Separate parent nodes (not embedded) from regular nodes
        regular_docs = []
        regular_ids = []
        parent_count = 0
        
        for i, d in enumerate(docs):
            normalized = self._normalize_doc(d)
            md = normalized.get("metadata", {}) or {}
            doc_id = ids[i] if ids else self._doc_id(d)
            
            if md.get("is_parent"):
                # Store parent node separately (for Small-to-Big retrieval)
                parent_id = md.get("node_id", doc_id)
                self._parent_nodes[parent_id] = {
                    "id": parent_id,
                    "content": normalized.get("content", ""),
                    "metadata": md,
                }
                parent_count += 1
            else:
                regular_docs.append(d)
                regular_ids.append(doc_id)
        
        if parent_count > 0:
            logger.info(f"Stored {parent_count} parent nodes (not embedded)")
            self._save_parent_nodes()  # Persist to disk
        
        if not regular_docs:
            return parent_count

        bs = max(1, batch_size)
        col = self.collection
        
        if col is None:
            return self.add_documents(regular_docs, ids=regular_ids, batch_size=bs) + parent_count

        total = 0
        for start in range(0, len(regular_docs), bs):
            batch = regular_docs[start : start + bs]
            batch_ids = regular_ids[start : start + bs]
            lc_docs = self._to_documents(batch, batch_ids)
            texts = [d.page_content for d in lc_docs]
            metas = [d.metadata for d in lc_docs]
            embs = self.embedder.embed_documents(texts)
            col.upsert(ids=batch_ids, documents=texts, metadatas=metas, embeddings=embs)
            total += len(batch)

        logger.info(f"Upserted {total} documents to vector store")
        return total + parent_count

    def count(self) -> int:
        col = self.collection
        return int(col.count()) if col else 0

    def get_all_documents(self, limit: int = 5000) -> List[Dict[str, Any]]:
        col = self.collection
        if col is None:
            return []
        
        result = col.get(limit=limit, include=['documents', 'metadatas'])
        docs = []
        for i, doc_content in enumerate(result.get('documents', [])):
            if doc_content:
                docs.append({
                    'id': result['ids'][i] if result.get('ids') else str(i),
                    'content': doc_content,
                    'metadata': result['metadatas'][i] if result.get('metadatas') else {},
                })
        return docs

    def delete_documents(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        
        col = self.collection
        if col is None:
            return 0
        
        col.delete(ids=list(ids))
        logger.info(f"Deleted {len(ids)} documents from vector store")
        return len(ids)

    def get_parent_node(self, parent_id: str) -> Optional[Dict[str, Any]]:
        return self._parent_nodes.get(parent_id)
    
    @property
    def parent_nodes(self) -> Dict[str, Dict[str, Any]]:
        return self._parent_nodes
