from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import gradio as gr
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class GradioConfig:
    server_host: str = "127.0.0.1"
    server_port: int = 7860

def _load_env() -> None:
    dotenv_path = find_dotenv(usecwd=True) or ""
    load_dotenv(dotenv_path=dotenv_path or None, override=False)


from core.rag.embedding_model import EmbeddingConfig, QwenEmbeddings
from core.rag.vector_store import ChromaConfig, ChromaVectorDB
from core.rag.retrival import Retriever, RetrievalMode, get_retrieval_config
from core.rag.generator import RAGContextBuilder, build_context, build_prompt, SYSTEM_PROMPT

_load_env()

RETRIEVAL_MODE = RetrievalMode.HYBRID_RERANK

# LLM Config
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-32b")
LLM_API_BASE = "https://api.groq.com/openai/v1"
LLM_API_KEY_ENV = "GROQ_API_KEY"

# Load retrieval config
GRADIO_CFG = GradioConfig()
RETRIEVAL_CFG = get_retrieval_config()


class AppState:
    def __init__(self) -> None:
        self.db: Optional[ChromaVectorDB] = None
        self.retriever: Optional[Retriever] = None
        self.rag_builder: Optional[RAGContextBuilder] = None
        self.client: Optional[OpenAI] = None


STATE = AppState()


def _init_resources() -> None:
    if STATE.db is not None:
        return

    print(f" Đang khởi tạo Database & Re-ranker...")
    print(f" Retrieval Mode: {RETRIEVAL_MODE.value}")

    emb = QwenEmbeddings(EmbeddingConfig())

    db_cfg = ChromaConfig()
    
    STATE.db = ChromaVectorDB(
        embedder=emb,
        config=db_cfg,
    )
    STATE.retriever = Retriever(vector_db=STATE.db)

    # Initialize LLM Client
    api_key = (os.getenv(LLM_API_KEY_ENV) or "").strip()
    if not api_key:
        raise RuntimeError(f"Missing {LLM_API_KEY_ENV}")
    STATE.client = OpenAI(api_key=api_key, base_url=LLM_API_BASE)
    
    # RAGContextBuilder - only retrieve, no LLM call
    STATE.rag_builder = RAGContextBuilder(retriever=STATE.retriever)
    
    print(" Đã sẵn sàng!")


def rag_chat(message: str, history: List[Dict[str, str]] | None = None):
    _init_resources()

    assert STATE.db is not None
    assert STATE.client is not None
    assert STATE.retriever is not None
    assert STATE.rag_builder is not None

    # Step 1: Retrieve and prepare context
    prepared = STATE.rag_builder.retrieve_and_prepare(
        message,
        k=RETRIEVAL_CFG.top_k, 
        initial_k=RETRIEVAL_CFG.initial_k,
        mode=RETRIEVAL_MODE.value,
    )
    results = prepared["results"]

    if not results:
        yield "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong dữ liệu."
        return

    # Step 2: Call LLM streaming to generate answer
    completion = STATE.client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prepared["prompt"]}],
        temperature=0.0,
        max_tokens=4096,
        stream=True,
    )
    
    acc = ""
    for chunk in completion:
        delta = getattr(chunk.choices[0].delta, "content", "") or ""
        if delta:
            acc += delta
            # Filter out <think>...</think> before yielding
            display_text = _filter_think_tags(acc)
            yield display_text

    # Yield final result (filtered)
    yield _filter_think_tags(acc)


def _filter_think_tags(text: str) -> str:
    import re
    # Remove <think>...</think> blocks (including multi-line)
    filtered = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove leading whitespace
    filtered = filtered.strip()
    return filtered




# Create Gradio interface
demo = gr.ChatInterface(
    fn=rag_chat,
    title=f"HUST RAG Assistant",
    description=f"Trợ lý học vụ Đại học Bách khoa Hà Nội",
    examples=[
        "Sinh viên vi phạm quy chế thi thì bị xử lý như thế nào?",
        "Điều kiện để đổi ngành là gì?",
        "Làm thế nào để đăng ký hoãn thi?",
    ],
)

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Starting HUST RAG Assistant")
    print(f"{'='*60}\n")
    demo.launch(
        server_name=GRADIO_CFG.server_host, 
        server_port=GRADIO_CFG.server_port
    )
