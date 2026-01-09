#!/usr/bin/env python
"""Test Small-to-Big table summarization."""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from core.rag.chunk import chunk_markdown_file

def test_chunk_with_summary():
    """Test chunking a file with tables to verify summary generation."""
    
    # Use the K70 English requirements file which has many tables
    test_file = REPO_ROOT / "data/data_process/chuong_trinh_dao_tao/1.1. Kỹ thuật Cơ điện tử.md"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    print(f"📄 Processing: {test_file.name}")
    print("=" * 60)
    
    nodes = chunk_markdown_file(test_file)
    
    print(f"\n📊 Total nodes: {len(nodes)}")
    
    # Count different types
    parent_nodes = [n for n in nodes if n.metadata.get("is_parent")]
    summary_nodes = [n for n in nodes if n.metadata.get("is_table_summary")]
    table_nodes = [n for n in nodes if n.metadata.get("is_table") and not n.metadata.get("is_parent")]
    text_nodes = [n for n in nodes if not n.metadata.get("is_table") and not n.metadata.get("is_table_summary")]
    
    print(f"   - Parent nodes (raw tables, NOT embedded): {len(parent_nodes)}")
    print(f"   - Summary nodes (embedded for search): {len(summary_nodes)}")
    print(f"   - Small table nodes (embedded directly): {len(table_nodes)}")
    print(f"   - Text nodes: {len(text_nodes)}")
    
    # Debug: Show sample metadata
    if nodes:
        print("\n🔍 Sample metadata from first node:")
        sample = nodes[0].metadata
        for k, v in sample.items():
            print(f"   - {k}: {v}")
    
    # Export to markdown
    output_file = REPO_ROOT / "test" / "chunk_results.md"
    export_to_markdown(nodes, output_file, test_file.name)
    print(f"\n📝 Exported detailed results to: {output_file}")


def export_to_markdown(nodes, output_path: Path, source_name: str):
    """Export all chunks to a markdown file for review."""
    
    lines = [
        f"# Chunk Results: {source_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Summary",
        f"",
        f"| Type | Count |",
        f"|------|-------|",
    ]
    
    # Count types
    parent_nodes = [n for n in nodes if n.metadata.get("is_parent")]
    summary_nodes = [n for n in nodes if n.metadata.get("is_table_summary")]
    table_nodes = [n for n in nodes if n.metadata.get("is_table") and not n.metadata.get("is_parent")]
    text_nodes = [n for n in nodes if not n.metadata.get("is_table") and not n.metadata.get("is_table_summary")]
    
    lines.extend([
        f"| Parent nodes (raw tables, NOT embedded) | {len(parent_nodes)} |",
        f"| Summary nodes (embedded for search) | {len(summary_nodes)} |",
        f"| Small table nodes (embedded directly) | {len(table_nodes)} |",
        f"| Text nodes | {len(text_nodes)} |",
        f"| **Total** | **{len(nodes)}** |",
        f"",
        f"---",
        f"",
    ])
    
    # Group: Summary nodes with their parents
    lines.append("## 📝 Summary Nodes (with Parent Tables)")
    lines.append("")
    
    parent_map = {n.metadata.get("node_id"): n for n in parent_nodes}
    
    for i, node in enumerate(summary_nodes, 1):
        parent_id = node.metadata.get("parent_id", "")
        parent = parent_map.get(parent_id)
        meta = node.metadata
        
        # Build table identifier for title
        table_num = meta.get('table_number', '')
        table_title = meta.get('table_title', '')
        title_suffix = ""
        if table_num:
            title_suffix = f" (Bảng {table_num})"
        
        lines.append(f"### Summary #{i}{title_suffix}")
        lines.append(f"")
        lines.append(f"**Metadata:**")
        lines.append(f"- is_table_summary: True")
        lines.append(f"- parent_id: `{parent_id}`")
        if table_num:
            lines.append(f"- table_number: {table_num}")
        if table_title:
            lines.append(f"- table_title: {table_title}")
        if meta.get("source_file"):
            lines.append(f"- source_file: {meta.get('source_file')}")
        if meta.get("applicable_cohorts"):
            lines.append(f"- applicable_cohorts: {meta.get('applicable_cohorts')}")
        lines.append(f"")
        lines.append(f"**Summary Text (embedded for search):**")
        lines.append(f"")
        lines.append(f"> {node.get_content()}")
        lines.append(f"")
        
        if parent:
            lines.append(f"**Parent Table (raw, NOT embedded):**")
            lines.append(f"")
            lines.append(f"```markdown")
            lines.append(parent.get_content())
            lines.append(f"```")
            lines.append(f"")
        
        lines.append(f"---")
        lines.append(f"")

    
    # Small tables (embedded directly)
    if table_nodes:
        lines.append("## 📋 Small Tables (embedded directly)")
        lines.append("")
        
        for i, node in enumerate(table_nodes, 1):
            meta = node.metadata
            table_num = meta.get('table_number', '')
            table_title = meta.get('table_title', '')
            title_suffix = ""
            if table_num:
                title_suffix = f" (Bảng {table_num})"
            
            lines.append(f"### Small Table #{i}{title_suffix}")
            lines.append(f"")
            lines.append(f"**Metadata:**")
            lines.append(f"- is_table: True")
            if table_num:
                lines.append(f"- table_number: {table_num}")
            if table_title:
                lines.append(f"- table_title: {table_title}")
            if meta.get("table_part"):
                lines.append(f"- table_part: {meta.get('table_part')}")
            if meta.get("source_file"):
                lines.append(f"- source_file: {meta.get('source_file')}")
            if meta.get("applicable_cohorts"):
                lines.append(f"- applicable_cohorts: {meta.get('applicable_cohorts')}")
            if meta.get("chunk_index") is not None:
                lines.append(f"- chunk_index: {meta.get('chunk_index')}")
            lines.append(f"")
            lines.append(f"```markdown")
            lines.append(node.get_content())
            lines.append(f"```")
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

    
    # Text nodes
    lines.append("## 📄 Text Nodes")
    lines.append("")
    
    for i, node in enumerate(text_nodes, 1):
        content = node.get_content()
        meta = node.metadata
        
        lines.append(f"### Text #{i}")
        lines.append(f"")
        lines.append(f"**Metadata:**")
        if meta.get("document_type"):
            lines.append(f"- document_type: {meta.get('document_type')}")
        if meta.get("title"):
            lines.append(f"- title: {meta.get('title')}")
        if meta.get("applicable_cohorts"):
            lines.append(f"- applicable_cohorts: {meta.get('applicable_cohorts')}")
        if meta.get("source_file"):
            lines.append(f"- source_file: {meta.get('source_file')}")
        if meta.get("header_path"):
            lines.append(f"- header_path: {meta.get('header_path')}")
        if meta.get("Header 1"):
            lines.append(f"- Header 1: {meta.get('Header 1')}")
        if meta.get("chunk_index") is not None:
            lines.append(f"- chunk_index: {meta.get('chunk_index')}")
        lines.append(f"")
        
        lines.append(f"**Content:**")
        lines.append(f"")
        lines.append(content)
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
    
    # Write to file
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    test_chunk_with_summary()
