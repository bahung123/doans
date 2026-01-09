import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Load .env file
load_dotenv(find_dotenv(usecwd=True))

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.rag.chunk import chunk_markdown_file
from core.rag.embedding_model import EmbeddingConfig, QwenEmbeddings
from core.rag.vector_store import ChromaConfig, ChromaVectorDB


def get_existing_source_files(db: ChromaVectorDB) -> set:
    docs = db.get_all_documents()
    existing = set()
    for d in docs:
        meta = d.get("metadata", {})
        source = meta.get("source_basename") or meta.get("source_file")
        if source:
            existing.add(source)
    return existing


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force rebuild all files")
    args = parser.parse_args()
    
    print("=" * 60)
    print("BUILD HUST RAG DATABASE")
    print("=" * 60)
    
    print("\n[1/4] Initializing embedder...")
    emb_cfg = EmbeddingConfig()
    emb = QwenEmbeddings(emb_cfg)
    print(f"  ✓ Model: {emb_cfg.model}")
    print(f"  ✓ API: {emb_cfg.api_base_url}")
    
    print("\n[2/4] Initializing ChromaDB...")
    db_cfg = ChromaConfig()
    db = ChromaVectorDB(embedder=emb, config=db_cfg)
    old_count = db.count()
    print(f"  ✓ Collection: {db_cfg.collection_name}")
    print(f"  ✓ Current docs: {old_count}")
    
    # Get existing files to skip
    existing_files = set()
    if not args.force and old_count > 0:
        print("\n  Checking existing files...")
        existing_files = get_existing_source_files(db)
        print(f"  ✓ Found {len(existing_files)} files already in DB")
    
    print("\n[3/4] Processing markdown files...")
    root = REPO_ROOT / "data" / "data_process"
    md_files = sorted(root.rglob("*.md"))
    print(f"  Found {len(md_files)} markdown files")
    
    total = 0
    skipped = 0
    for i, f in enumerate(md_files, 1):
        # Skip if already exists
        if f.name in existing_files:
            print(f"  [{i}/{len(md_files)}] {f.name}: SKIP (already in DB)")
            skipped += 1
            continue
            
        try:
            docs = chunk_markdown_file(f)
            if docs:
                n = db.upsert_documents(docs)
                total += n
                print(f"  [{i}/{len(md_files)}] {f.name}: {n} chunks")
            else:
                print(f"  [{i}/{len(md_files)}] {f.name}: SKIP (no chunks)")
        except Exception as e:
            print(f"  [{i}/{len(md_files)}] {f.name}: ERROR - {e}")
    
    new_count = db.count()
    print(f"\n[4/4] Database rebuilt successfully!")
    print(f"  Skipped: {skipped} files (already in DB)")
    print(f"  Total upserted: {total}")
    print(f"  Old DB count: {old_count}")
    print(f"  New DB count: {new_count}")
    print(f"  Delta: {new_count - old_count:+d}")
    
    print("\n" + "=" * 60)
    print("TESTING QUERY")
    print("=" * 60)
    
    from core.rag.retrival import Retriever, RetrievalMode
    
    # Test với mode VECTOR_ONLY
    test_mode = RetrievalMode.VECTOR_ONLY
    retriever = Retriever(vector_db=db, use_reranker=False)
    
    test_query = "Yêu cầu TOEIC của ngành Toán tin là bao nhiêu?"
    print(f"Query: {test_query}")
    print(f"Mode: {test_mode.value}")
    
    results = retriever.flexible_search(test_query, mode=test_mode, k=3)
    
    if results:
        print(f"\nTop {len(results)} results:")
        for i, r in enumerate(results, 1):
            score = r.get('distance') or r.get('rerank_score') or r.get('final_rank')
            print(f"\n[{i}] Score: {score}")
            print(f"  Source: {r['metadata'].get('source_file', 'N/A')}")
            print(f"  Section: {r['metadata'].get('section', 'N/A')}")
            print(f"  Content: {r['content'][:150]}...")
    else:
        print("No results found!")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()