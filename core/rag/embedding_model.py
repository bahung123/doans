from __future__ import annotations
import os
import logging
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
from openai import OpenAI
from langchain_core.embeddings import Embeddings
import time
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    api_base_url: str = "https://api.siliconflow.com/v1"
    model: str = "Qwen/Qwen3-Embedding-4B"
    dimension: int = 2048
    batch_size: int = 16


_embed_config: EmbeddingConfig | None = None

def get_embedding_config() -> EmbeddingConfig:
    global _embed_config
    if _embed_config is None:
        _embed_config = EmbeddingConfig()
    return _embed_config


class QwenEmbeddings(Embeddings):
    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or get_embedding_config()
        
        api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY environment variable not set")
        
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.config.api_base_url,
        )
        logger.info(f"QwenEmbeddings initialized: {self.config.model}")
    
    def embed_query(self, text: str) -> List[float]:
        return self._embed_texts([text])[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts(texts)
    
    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        
        all_embeddings: List[List[float]] = []
        batch_size = self.config.batch_size
        max_retries = 3
        
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            
            for attempt in range(max_retries):
                try:
                    response = self._client.embeddings.create(
                        model=self.config.model,
                        input=batch,
                    )
                    for item in response.data:
                        all_embeddings.append(item.embedding)
                    break
                except Exception as e:
                    if "rate" in str(e).lower() and attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # 1s, 2s, 4s
                        logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
        
        return all_embeddings
    
    def embed_texts_np(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._embed_texts(list(texts)), dtype=np.float32)


# Legacy alias
SiliconFlowConfig = EmbeddingConfig
get_config = get_embedding_config
