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
    
    # RAGContextBuilder
    STATE.rag_builder = RAGContextBuilder(retriever=STATE.retriever)
    
    print(" Đã sẵn sàng!")


def rag_chat(message: str, history: List[Dict[str, str]] | None = None):
    _init_resources()

    assert STATE.db is not None
    assert STATE.client is not None
    assert STATE.retriever is not None
    assert STATE.rag_builder is not None

    # Retrieve and prepare context
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

    # Call LLM streaming to generate answer
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
            yield acc

    # Debug info with retrieval mode
    debug_info = f"\n\n---\n\n**Retrieved (Top {len(results)} | Mode: {RETRIEVAL_MODE.value})**\n\n"
    for i, r in enumerate(results, 1):
        md = r.get("metadata", {})
        content = r.get("content", "").strip()
        rerank_score = r.get("rerank_score")
        distance = r.get("distance")
        
        # Extract metadata
        source = md.get("source_file", "N/A")
        doc_type = md.get("document_type", "N/A")
        header = md.get("header_path", "")
        cohorts = md.get("applicable_cohorts", "")
        program = md.get("program_name", "")
        issued_year = md.get("issued_year", "")

        # Display scores based on retrieval mode
        score_info = ""
        if rerank_score is not None:
            score_info += f"Rerank: `{rerank_score:.4f}` "
        if distance is not None:
            score_info += f"Distance: `{distance:.4f}`"
        if not score_info:
            score_info = f"Rank: `{r.get('final_rank', i)}`"

        # Build metadata line
        meta_parts = [f"**Nguồn:** {source}", f"**Loại:** {doc_type}"]
        if issued_year:
            meta_parts.append(f"**Năm:** {issued_year}")
        if cohorts:
            meta_parts.append(f"**Áp dụng:** {cohorts}")
        if program:
            meta_parts.append(f"**CTĐT:** {program}")
        
        debug_info += f"**#{i}** | {score_info}\n"
        debug_info += f"   - {' | '.join(meta_parts)}\n"
        if header and header != "/":
            debug_info += f"   - **Mục:** {header[:80]}{'...' if len(header) > 80 else ''}\n"
        debug_info += f"   - **Content:** {content[:200]}{'...' if len(content) > 200 else ''}\n\n"

    yield acc + debug_info




# Create Gradio interface
demo = gr.ChatInterface(
    fn=rag_chat,
    title=f"HUST RAG Assistant",
    description=f"Trợ lý học vụ Đại học Bách khoa Hà Nội",
    examples=[
        "Điều kiện tốt nghiệp đại học là gì?",
        "Yêu cầu TOEIC của ngành Toán tin là bao nhiêu?",
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
