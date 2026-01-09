from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def print_header(mode: str):
    """Print evaluation header."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 70)
    print(f"{'RAG EVALUATION':^70}")
    print(f"{'Mode: ' + mode.upper():^70}")
    print(f"{timestamp:^70}")
    print("=" * 70)

def run_rouge_only(samples: int, output_dir: str):
    from evaluation.rouge_eval import run_rouge_evaluation
    return run_rouge_evaluation(sample_size=samples, output_dir=output_dir)


def run_ragas_only(samples: int, output_dir: str):
    from evaluation.ragas_eval import run_ragas_evaluation
    return run_ragas_evaluation(sample_size=samples, output_dir=output_dir)


def run_all(samples: int, output_dir: str):
    print("\n" + "-" * 70)
    print("STEP 1/2: ROUGE EVALUATION")
    print("-" * 70)
    rouge_scores = run_rouge_only(samples, output_dir)
    
    print("\n" + "-" * 70)
    print("STEP 2/2: RAGAS EVALUATION")
    print("-" * 70)
    ragas_scores = run_ragas_only(samples, output_dir)
    
    # Combine and print final summary
    print("\n" + "=" * 70)
    print(f"{'FINAL EVALUATION SUMMARY':^70}")
    print("=" * 70)
    
    if rouge_scores:
        print("\n[ROUGE Metrics - Text-based]")
        for metric, score in rouge_scores.items():
            bar = "#" * int(float(score) * 20) + "-" * (20 - int(float(score) * 20))
            print(f"  {metric:25} [{bar}] {score:.4f}")
    
    if ragas_scores:
        print("\n[RAGAS Metrics - LLM-based]")
        import math
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric in ragas_scores:
                score = float(ragas_scores[metric])
                if not math.isnan(score):
                    bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
                    print(f"  {metric:25} [{bar}] {score:.4f}")
                else:
                    print(f"  {metric:25} [FAILED]")
    
    print("\n" + "=" * 70)
    print(" All evaluations complete!")
    
    return {"rouge": rouge_scores, "ragas": ragas_scores}


def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Script - Run ROUGE and/or RAGAS metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/run_eval.py --mode all --samples 10
  python evaluation/run_eval.py --mode rouge --samples 20
  python evaluation/run_eval.py --mode ragas --samples 5
  python evaluation/run_eval.py --mode all --samples 0  # All samples

Evaluation modes:
  rouge   - Text-based metrics (ROUGE-1, ROUGE-2, ROUGE-L)
  ragas   - LLM-based metrics (Faithfulness, Answer Relevancy, etc.)
  all     - Both ROUGE and RAGAS
        """
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="all",
        choices=["rouge", "ragas", "all"],
        help="Evaluation mode: rouge, ragas, or all (default: all)"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=10,
        help="Number of samples to evaluate (0 = all samples, default: 10)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output directory for results (default: evaluation/results)"
    )
    
    args = parser.parse_args()
    
    print_header(args.mode)
    print(f"\n[Configuration]")
    print(f"  Mode: {args.mode}")
    print(f"  Samples: {args.samples if args.samples > 0 else 'ALL'}")
    print(f"  Output: {args.output or 'evaluation/results'}")
    
    if args.mode == "rouge":
        run_rouge_only(args.samples, args.output)
    elif args.mode == "ragas":
        run_ragas_only(args.samples, args.output)
    else:
        run_all(args.samples, args.output)


if __name__ == "__main__":
    main()
