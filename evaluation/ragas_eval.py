from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import find_dotenv, load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(find_dotenv(usecwd=True))

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, RougeScore 
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from evaluation.eval_utils import load_config, load_csv_data, init_rag_components, generate_answers


def run_ragas_evaluation(sample_size: int = 10, output_dir: Optional[str] = None, retrieval_mode: str = "hybrid_rerank") -> dict:
    print(f"\n{'='*60}\nRAGAS EVALUATION - Mode: {retrieval_mode}\n{'='*60}")
    config = load_config()
    out_path = Path(output_dir) if output_dir else REPO_ROOT / config["output"]["dir"]
    
    # Init components
    rag_gen, info = init_rag_components(config)
    
    eval_config = config.get("eval_llm", config["llm"])  # Fallback to llm for backward compatibility
    gen_config = config.get("gen_llm", config["llm"])
    
    # Get eval API key (may be different from generation API key)
    eval_api_key_env = eval_config.get("api_key_env", "SILICONFLOW_API_KEY")
    eval_api_key = os.getenv(eval_api_key_env, "")
    if not eval_api_key:
        # Fallback to SILICONFLOW_API_KEY
        eval_api_key = os.getenv("SILICONFLOW_API_KEY", "")
    if not eval_api_key:
        raise ValueError(f"Missing API key. Set {eval_api_key_env} or SILICONFLOW_API_KEY")
    
    eval_base_url = eval_config.get("base_url", "https://api.siliconflow.com/v1")
    eval_model = eval_config.get("model", config["llm"]["model"])
    gen_model = gen_config.get("model", config["llm"]["model"])
    
    # ChatOpenAI expects SecretStr or callable for api_key
    from pydantic import SecretStr
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        model=eval_model,
        api_key=SecretStr(eval_api_key),  # Wrap in SecretStr
        base_url=eval_base_url,
        temperature=eval_config.get("temperature", 0),
        timeout=120,  # 2 minutes timeout per request
        max_retries=3,
    ))
    evaluator_embeddings = LangchainEmbeddingsWrapper(info["embeddings"])
    
    # Load data & generate answers
    questions, ground_truths = load_csv_data(
        str(REPO_ROOT / config["data"]["csv_path"]),
        sample_size or config["data"]["sample_size"]
    )
    
    # Use generation model (NOT evaluation model) for RAG answers
    answers, contexts = generate_answers(
        rag_gen, questions,
        retrieval_mode=retrieval_mode,
        llm_client=info["llm_client"],
        llm_model=gen_config.get("model", config["llm"]["model"]),
    )
    
    # Run RAGAS
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    
    # RougeScore - non-LLM metrics (fast & free)
    rouge1_scorer = RougeScore(rouge_type='rouge1', mode='fmeasure')
    rouge2_scorer = RougeScore(rouge_type='rouge2', mode='fmeasure')
    rougeL_scorer = RougeScore(rouge_type='rougeL', mode='fmeasure')
    
    results = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, rouge1_scorer, rouge2_scorer, rougeL_scorer],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
        run_config=RunConfig(
            max_workers=8,  # SiliconFlow allows 1000 RPM, safe to use 8 workers
            timeout=600,    # 10 minutes per job
            max_retries=3,  # Retry on transient failures
            max_wait=120,   # Max wait between retries
        ),
    )
    
    # Save JSON
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_path / f"ragas_{timestamp}.json"
    
    # Convert results to dict - handle both per-sample and aggregate scores
    scores_per_sample: list = []
    avg_scores: dict = {}
    
    # Get per-sample scores from results DataFrame
    if hasattr(results, "to_pandas"):
        df = results.to_pandas()  # type: ignore[union-attr]
        metric_cols = [col for col in df.columns if col not in ("question", "answer", "contexts", "ground_truth", "user_input", "response", "reference", "retrieved_contexts")]
        
        for _, row in df.iterrows():
            sample_scores = {}
            for col in metric_cols:
                val = row[col]
                if val is not None:
                    try:
                        sample_scores[col] = float(val)
                    except (ValueError, TypeError):
                        pass
            scores_per_sample.append(sample_scores)
        
        # Calculate average scores
        import numpy as np
        for col in metric_cols:
            values = [s.get(col) for s in scores_per_sample if s.get(col) is not None]
            if values:
                valid_values = [v for v in values if not np.isnan(v)]
                if valid_values:
                    avg_scores[col] = float(np.mean(valid_values))
    else:
        # Fallback for older ragas versions
        if hasattr(results, "scores"):
            avg_scores = results.scores  # type: ignore
        elif hasattr(results, "__getitem__"):
            for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "rouge1_score", "rouge2_score", "rougeL_score"]:
                try:
                    avg_scores[k] = float(results[k])  # type: ignore
                except (KeyError, TypeError):
                    pass
    
    # Save JSON with both per-sample and average scores
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "retrieval_mode": retrieval_mode,
            "config": {"sample_size": len(questions), **{k: v for k, v in info.items() if k not in ("embeddings", "llm_client")}},
            "avg_scores": avg_scores,
            "scores_per_sample": scores_per_sample,
            "samples": [
                {"question": q, "answer": a, "ground_truth": gt, "contexts": ctx, "scores": sc}
                for q, a, gt, ctx, sc in zip(questions, answers, ground_truths, contexts, scores_per_sample or [{}]*len(questions))
            ]
        }, f, ensure_ascii=False, indent=2)
    
    # Save CSV summary (only average scores, not per-sample)
    csv_path = out_path / f"ragas_{retrieval_mode}_{timestamp}.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        # Header: metric names
        f.write("retrieval_mode,sample_size," + ",".join(avg_scores.keys()) + "\n")
        # Data: average values
        f.write(f"{retrieval_mode},{len(questions)}," + ",".join([f"{v:.4f}" for v in avg_scores.values()]) + "\n")
    
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY - {retrieval_mode} ({len(questions)} samples)")
    print(f"{'='*60}")
    for metric, score in avg_scores.items():
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        print(f"  {metric:25} [{bar}] {score:.4f}")
    
    return avg_scores


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAGAS Evaluation - Compare retrieval modes")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["all", "vector_only", "bm25_only", "hybrid", "hybrid_rerank"],
                        help="Retrieval mode to evaluate (default: all)")
    args = parser.parse_args()
    
    ALL_MODES = ["vector_only", "bm25_only", "hybrid", "hybrid_rerank"]
    
    # Single mode or all modes
    if args.mode != "all":
        print(f"\n{'='*60}")
        print(f"RAGAS EVALUATION - Single Mode: {args.mode}")
        print(f"{'='*60}")
        scores = run_ragas_evaluation(args.samples, args.output, args.mode)
        print(f"\nScores: {scores}")
    else:
        # Run all modes and compare
        all_results = {}
        
        for mode in ALL_MODES:
            print(f"\n{'#'*60}")
            print(f"# Running mode: {mode}")
            print(f"{'#'*60}")
            try:
                scores = run_ragas_evaluation(args.samples, args.output, mode)
                all_results[mode] = scores
            except Exception as e:
                print(f"Error in {mode}: {e}")
                all_results[mode] = {"error": str(e)}
    
        # Print comparison table
        print(f"\n{'='*80}")
        print("COMPARISON REPORT - All Retrieval Modes")
        print(f"{'='*80}")
        
        # Get all metric names
        metrics = set()
        for scores in all_results.values():
            if isinstance(scores, dict) and "error" not in scores:
                metrics.update(scores.keys())
        metrics = sorted(metrics)
        
        # Print header
        header = f"{'Metric':<25} | " + " | ".join(f"{m:<15}" for m in ALL_MODES)
        print(header)
        print("-" * len(header))
        
        # Print each metric
        for metric in metrics:
            row = f"{metric:<25} | "
            for mode in ALL_MODES:
                scores = all_results.get(mode, {})
                if isinstance(scores, dict) and metric in scores:
                    val = scores[metric]
                    if isinstance(val, (int, float)):
                        row += f"{val:<15.4f} | "
                    else:
                        row += f"{'N/A':<15} | "
                else:
                    row += f"{'N/A':<15} | "
            print(row)
        
        # Save comparison JSON and CSV
        config = load_config()
        out_path = Path(args.output) if args.output else REPO_ROOT / config["output"]["dir"]
        out_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        comparison_path = out_path / f"comparison_{timestamp}.json"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "sample_size": args.samples,
                "results": all_results,
            }, f, ensure_ascii=False, indent=2)
        
        # Save CSV summary
        csv_path = out_path / f"comparison_{timestamp}.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("Metric," + ",".join(ALL_MODES) + "\n")
            # Data rows
            for metric in metrics:
                row_values = [metric]
                for mode in ALL_MODES:
                    scores = all_results.get(mode, {})
                    if isinstance(scores, dict) and metric in scores:
                        val = scores[metric]
                        if isinstance(val, (int, float)):
                            row_values.append(f"{val:.4f}")
                        else:
                            row_values.append("N/A")
                    else:
                        row_values.append("N/A")
                f.write(",".join(row_values) + "\n")
        
        print(f"\nComparison saved:")
        print(f"  - JSON: {comparison_path}")
        print(f"  - CSV:  {csv_path}")