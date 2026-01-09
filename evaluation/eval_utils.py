from __future__ import annotations
import os
import re
import sys
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Any
from dotenv import find_dotenv, load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(find_dotenv(usecwd=True))

from core.rag.embedding_model import SiliconFlowConfig, QwenEmbeddings
from core.rag.vector_store import ChromaConfig, ChromaVectorDB
from core.rag.retrival import Retriever
from core.rag.generator import RAGGenerator


def strip_thinking(text: str) -> str:
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


def load_config() -> dict:
    return {
        # Model for generating RAG answers
        "gen_llm": {
            "model": os.getenv("GEN_LLM_MODEL", "nex-agi/DeepSeek-V3.1-Nex-N1"),
            "temperature": float(os.getenv("GEN_LLM_TEMPERATURE", "0")),
            "max_tokens": int(os.getenv("GEN_LLM_MAX_TOKENS", "4096")),
        },
        # Model for RAGAS evaluation (should be DIFFERENT from gen_llm to avoid bias)
        "eval_llm": {
            "model": os.getenv("EVAL_LLM_MODEL", "nex-agi/DeepSeek-V3.1-Nex-N1"),
            "api_key_env": os.getenv("EVAL_API_KEY_ENV", "SILICONFLOW_API_KEY"),  # Which env var has the API key
            "base_url": os.getenv("EVAL_API_BASE_URL", "https://api.siliconflow.com/v1"),
            "temperature": float(os.getenv("EVAL_LLM_TEMPERATURE", "0")),
        },
        # Legacy compatibility (deprecated, use gen_llm instead)
        "llm": {
            "model": os.getenv("GEN_LLM_MODEL", os.getenv("EVAL_LLM_MODEL", "nex-agi/DeepSeek-V3.1-Nex-N1")),
            "temperature": float(os.getenv("EVAL_LLM_TEMPERATURE", "0")),
            "timeout": int(os.getenv("EVAL_LLM_TIMEOUT", "30")),
            "max_retries": int(os.getenv("EVAL_LLM_MAX_RETRIES", "2")),
            "max_tokens": int(os.getenv("EVAL_LLM_MAX_TOKENS", "4096")),
        },
        "retrieval": {
            "top_k": int(os.getenv("EVAL_TOP_K", "5")),
            "initial_k": int(os.getenv("EVAL_INITIAL_K", "100")),
        },
        "data": {
            "csv_path": os.getenv("EVAL_CSV_PATH", "data/data.csv"),
            "sample_size": int(os.getenv("EVAL_SAMPLE_SIZE", "0")),
        },
        "output": {
            "dir": os.getenv("EVAL_OUTPUT_DIR", "evaluation/results"),
        },
    }


def load_csv_data(csv_path: str, sample_size: int = 0) -> tuple[list, list]:
    questions, ground_truths = [], []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('question') and row.get('ground_truth'):
                questions.append(row['question'])
                ground_truths.append(row['ground_truth'])
    
    if sample_size > 0:
        questions = questions[:sample_size]
        ground_truths = ground_truths[:sample_size]
    
    return questions, ground_truths


def init_rag_components(config: dict) -> tuple:
    from openai import OpenAI
    
    emb_cfg = SiliconFlowConfig()
    qwen_embeddings = QwenEmbeddings(emb_cfg)
    
    db_cfg = ChromaConfig()
    db = ChromaVectorDB(embedder=qwen_embeddings, config=db_cfg)
    retriever = Retriever(vector_db=db)
    
    rag_builder = RAGGenerator(retriever=retriever)
    
    api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing SILICONFLOW_API_KEY")
    
    llm_client = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.com/v1",
    )
    
    info = {
        "gen_model": config["llm"]["model"],
        "eval_model": config["llm"]["model"],  
        "embed_model": emb_cfg.model,
        "collection": db_cfg.collection_name,
        "doc_count": db.count(),
        "embeddings": qwen_embeddings,
        "llm_client": llm_client,
    }
    
    return rag_builder, info


def generate_answers(
    rag_builder: RAGGenerator,
    questions: list,
    retrieval_mode: str = "hybrid_rerank",
    max_workers: int = 8,  # SiliconFlow allows 1000 RPM, safe to use 8 workers
    llm_client = None,
    llm_model: str = "nex-agi/DeepSeek-V3.1-Nex-N1",
    timeout_per_question: int = 120,  # 2 minutes timeout per question
) -> tuple[list, list]:
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
    
    if llm_client is None:
        from openai import OpenAI
        api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing SILICONFLOW_API_KEY")
        llm_client = OpenAI(
            api_key=api_key, 
            base_url="https://api.siliconflow.com/v1",
            timeout=60.0,  # 60 seconds timeout for API call
        )
    
    def process_question(idx_q: tuple) -> tuple:
        """Process a single question and return (idx, answer, contexts, error, retrieval_empty)"""
        idx, q = idx_q
        try:
            prepared = rag_builder.retrieve_and_prepare(q, mode=retrieval_mode)
            
            if not prepared["results"]:
                # Mark retrieval as failed for debugging RAG vs LLM issues
                return idx, "Không tìm thấy thông tin trong dữ liệu hiện có.", [], None, True
            
            completion = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prepared["prompt"]}],
                temperature=0.0,
                max_tokens=4096,
            )
            
            answer = strip_thinking(completion.choices[0].message.content or "")
            return idx, answer, prepared["contexts"], None, False
        except Exception as e:
            return idx, "Không thể trả lời.", [], str(e), False
    
    n = len(questions)
    answers: list[str] = [""] * n
    contexts: list[list] = [[] for _ in range(n)]  # FIXED: [[]] * n creates shared references!
    retrieval_failed: list[bool] = [False] * n  # Track retrieval failures for debugging
    errors: list[str | None] = [None] * n  # Track error messages
    
    print(f"  Generating answers with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_question, (i, q)): i for i, q in enumerate(questions)}
        done_count = 0
        
        for future in as_completed(futures):
            try:
                # Use timeout_per_question for actual timeout enforcement
                idx, answer, ctx, error, is_retrieval_empty = future.result(timeout=timeout_per_question)
                answers[idx] = answer
                contexts[idx] = ctx
                retrieval_failed[idx] = is_retrieval_empty
                errors[idx] = error
                done_count += 1
                
                if error:
                    print(f"  [{done_count}/{n}] Q{idx+1}: Error - {error}")
                elif is_retrieval_empty:
                    print(f"  [{done_count}/{n}] Q{idx+1}: Done (⚠️ no retrieval results)")
                else:
                    print(f"  [{done_count}/{n}] Q{idx+1}: Done")
            except TimeoutError:
                idx = futures[future]
                answers[idx] = "Timeout: Không thể xử lý câu hỏi trong thời gian cho phép."
                contexts[idx] = []
                errors[idx] = f"Timeout after {timeout_per_question}s"
                done_count += 1
                print(f"  [{done_count}/{n}] Q{idx+1}: TIMEOUT")
    
    # Log summary of retrieval failures
    failed_count = sum(retrieval_failed)
    if failed_count > 0:
        print(f"  ⚠️ {failed_count}/{n} questions had empty retrieval results")
    
    return answers, contexts


def save_eval_report(
    output_dir: Path,
    eval_type: str,
    questions: list,
    answers: list,
    ground_truths: list,
    contexts: list,
    scores_per_sample: list,
    avg_scores: dict,
    config: dict,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / f"{eval_type}_eval_{timestamp}.json"
    
    data = {
        "timestamp": timestamp,
        "eval_type": eval_type,
        "config": config,
        "avg_scores": avg_scores,
        "samples": [
            {
                "idx": i + 1,
                "question": q,
                "answer": a,
                "ground_truth": gt[0] if isinstance(gt, list) else gt,
                "contexts": ctx,
                "scores": scores,
            }
            for i, (q, a, gt, ctx, scores) in enumerate(
                zip(questions, answers, ground_truths, contexts, scores_per_sample)
            )
        ],
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f" Saved: {json_path}")
    return json_path


def print_scores(title: str, scores: dict, metrics: list | None = None):
    import math
    print(f"\n[{title}]")
    
    metrics = metrics or list(scores.keys())
    for metric in metrics:
        if metric not in scores:
            print(f"  {metric:25} [N/A]")
            continue
        
        score = float(scores[metric])
        if math.isnan(score):
            print(f"  {metric:25} [FAILED - NaN]")
        else:
            bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
            print(f"  {metric:25} [{bar}] {score:.4f}")
