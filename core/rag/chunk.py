from __future__ import annotations
import os
import re
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import yaml
from openai import OpenAI
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode

# Cấu hình
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 200
TABLE_ROWS_PER_CHUNK = 15

# Cấu hình Small-to-Big
ENABLE_TABLE_SUMMARY = True
MIN_TABLE_ROWS_FOR_SUMMARY = 0  # Summarize ALL tables regardless of size
SUMMARY_MODEL = "nex-agi/DeepSeek-V3.1-Nex-N1"
SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"

# Regex
COURSE_PATTERN = re.compile(r"Học\s*phần\s+(.+?)\s*\(\s*m[ãa]\s+([^\)]+)\)", re.I | re.DOTALL)
TABLE_PLACEHOLDER = re.compile(r"__TBL_(\d+)__")
HEADER_KEYWORDS = {'TT', 'STT', 'MÃ', 'TÊN', 'KHỐI', 'SỐ', 'ID', 'NO', '#'}
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Pattern để trích xuất số bảng và tiêu đề
TABLE_TITLE_PATTERN = re.compile(r"(?:^|\n)#+\s*(?:Bảng|BẢNG)\s*(\d+(?:\.\d+)?)\s*[.:]*\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and s.count("|") >= 2

def _is_separator(line: str) -> bool:
    if not _is_table_row(line):
        return False
    return not line.strip().replace("|", "").replace("-", "").replace(":", "").replace(" ", "")

def _is_header(line: str) -> bool:
    if not _is_table_row(line):
        return False
    cells = [c.strip() for c in line.split("|") if c.strip()]
    if not cells or cells[0].isdigit():
        return False
    return any(k in cells[0].upper() for k in HEADER_KEYWORDS) or len(cells[0].split()) <= 3


def _extract_tables(text: str) -> Tuple[List[Tuple[str, List[str]]], str]:
    lines, tables, last_header, i = text.split("\n"), [], None, 0
    
    while i < len(lines) - 1:
        if _is_table_row(lines[i]) and _is_separator(lines[i + 1]):
            if _is_header(lines[i]):
                header = f"{lines[i]}\n{lines[i + 1]}\n"
                last_header, start = header, i + 2
            else:
                header = last_header or f"| {'|'.join(['Col'] * (lines[i].count('|') - 1))} |\n|{'|'.join(['---'] * (lines[i].count('|') - 1))}|\n"
                start = i
            
            rows, j = [], start
            while j < len(lines) and (_is_table_row(lines[j]) or _is_separator(lines[j])):
                if not _is_separator(lines[j]):
                    rows.append(lines[j])
                j += 1
            
            if rows:
                tables.append((header, rows))
            i = j
        else:
            i += 1
    
    # Thay thế bảng bằng placeholder
    result, tbl_idx, i = [], 0, 0
    while i < len(lines):
        if tbl_idx < len(tables) and i < len(lines) - 1 and _is_table_row(lines[i]) and _is_separator(lines[i + 1]):
            j = i
            while j < len(lines) and (_is_table_row(lines[j]) or _is_separator(lines[j])):
                j += 1
            result.append(f"__TBL_{tbl_idx}__")
            tbl_idx, i = tbl_idx + 1, j
        else:
            result.append(lines[i])
            i += 1
    
    return tables, "\n".join(result)


def _split_table(header: str, rows: List[str], max_rows: int = TABLE_ROWS_PER_CHUNK) -> List[str]:
    if len(rows) <= max_rows:
        return [header + "\n".join(rows)]
    
    chunks = []
    for i in range(0, len(rows), max_rows):
        chunk_rows = rows[i:i + max_rows]
        chunks.append(chunk_rows)
    
    # Gộp chunk cuối nếu quá nhỏ (< 5 dòng)
    if len(chunks) > 1 and len(chunks[-1]) < 5:
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    
    return [header + "\n".join(r) for r in chunks]

_summary_client: Optional[OpenAI] = None

def _get_summary_client() -> Optional[OpenAI]:
    global _summary_client
    if _summary_client is not None:
        return _summary_client
    
    api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        print("SILICONFLOW_API_KEY chưa được thiết lập. Tóm tắt bảng bị vô hiệu hóa.")
        return None
    
    _summary_client = OpenAI(api_key=api_key, base_url=SILICONFLOW_BASE_URL)
    return _summary_client


def _summarize_table(
    table_text: str, 
    context_hint: str = "",
    table_number: str = "",
    table_title: str = "",
    source_file: str = "",
    max_retries: int = 5,
    base_delay: float = 2.0
) -> str:

    import time
    
    if not ENABLE_TABLE_SUMMARY:
        raise RuntimeError("Table summarization is disabled but required. Set ENABLE_TABLE_SUMMARY = True")
    
    client = _get_summary_client()
    if client is None:
        raise RuntimeError("SILICONFLOW_API_KEY not set. Cannot summarize tables.")
    
    # Tạo chuỗi định danh bảng
    table_id_parts = []
    if table_number:
        table_id_parts.append(f"Bảng {table_number}")
    if table_title:
        table_id_parts.append(f'"{table_title}"')
    if source_file:
        table_id_parts.append(f"từ file {source_file}")
    
    table_identifier = " - ".join(table_id_parts) if table_id_parts else "Bảng không xác định"
    
    prompt = f"""Tóm tắt ngắn gọn nội dung bảng sau bằng tiếng Việt.

{f"**Thông tin bảng:** {table_identifier}" if table_identifier else ""}
{f"**Ngữ cảnh:** {context_hint}" if context_hint else ""}

**YÊU CẦU QUAN TRỌNG:**
- Bắt đầu tóm tắt bằng việc nêu rõ đây là {f"Bảng {table_number}" if table_number else "bảng nào"}{f' với tiêu đề "{table_title}"' if table_title else ""}{f" thuộc file {source_file}" if source_file else ""}
- Ghi rõ bảng này liệt kê/quy định về cái gì
- Nêu các cột chính trong bảng
- Thông tin quan trọng (nếu có số liệu cụ thể thì nêu ví dụ)

Bảng:
{table_text[:3000]}  
"""
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            summary = response.choices[0].message.content or ""
            if summary.strip():
                return summary.strip()
            else:
                raise ValueError("Empty summary returned from API")
                
        except Exception as e:
            last_error = e
            delay = base_delay * (2 ** attempt)  # Exponential backoff: 2, 4, 8, 16, 32 seconds
            print(f"Thử lại {attempt + 1}/{max_retries} cho {table_identifier}: {e}")
            print(f"   Chờ {delay:.1f}s trước khi thử lại...")
            time.sleep(delay)
    
    # Tất cả các lần thử đều thất bại
    raise RuntimeError(f"Không thể tóm tắt {table_identifier} sau {max_retries} lần thử. Lỗi cuối: {last_error}")





def _create_table_nodes(
    table_text: str, 
    metadata: dict, 
    context_hint: str = "",
    table_number: str = "",
    table_title: str = "",
    source_file: str = ""
) -> List[TextNode]:
    # Đếm số dòng để quyết định có nên tóm tắt không
    row_count = table_text.count("\n")
    
    # Thêm thông tin bảng vào metadata
    table_meta = {**metadata}
    if table_number:
        table_meta["table_number"] = table_number
    if table_title:
        table_meta["table_title"] = table_title
    
    if row_count < MIN_TABLE_ROWS_FOR_SUMMARY:
        # Bảng quá nhỏ, giữ nguyên (không cần tóm tắt)
        return [TextNode(text=table_text, metadata={**table_meta, "is_table": True})]
    
    # Tạo tóm tắt với logic thử lại
    summary = _summarize_table(
        table_text, 
        context_hint,
        table_number=table_number,
        table_title=table_title,
        source_file=source_file
    )
    
    # Tạo node cha
    parent_id = str(uuid.uuid4())
    parent_node = TextNode(
        text=table_text,
        metadata={
            **table_meta,
            "is_table": True,
            "is_parent": True,  # Cờ để bỏ qua embedding
            "node_id": parent_id,
        }
    )
    parent_node.id_ = parent_id
    
    # Tạo node tóm tắt
    summary_node = TextNode(
        text=summary,
        metadata={
            **table_meta,
            "is_table_summary": True,
            "parent_id": parent_id,  # Liên kết với node cha
        }
    )
    
    table_id = f"Bảng {table_number}" if table_number else "table"
    print(f"Đã tạo tóm tắt cho {table_id} ({row_count} dòng)")
    return [parent_node, summary_node]




def _enrich_metadata(node: BaseNode, source_path: Path | None) -> None:
    if source_path:
        node.metadata.update({"source_path": str(source_path), "source_file": source_path.name})
    if "Học phần" in (text := node.get_content()) and (m := COURSE_PATTERN.search(text)):
        node.metadata.update({"course_name": " ".join(m.group(1).split()), "course_code": " ".join(m.group(2).split())})


def _chunk_text(text: str, metadata: dict) -> List[BaseNode]:
    if len(text) <= CHUNK_SIZE:
        return [TextNode(text=text, metadata=metadata.copy())]
    return SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).get_nodes_from_documents(
        [Document(text=text, metadata=metadata.copy())]
    )


def _extract_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    match = FRONTMATTER_PATTERN.match(text)
    if not match:
        return {}, text
    
    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
        remaining_text = text[match.end():].lstrip()
        return frontmatter, remaining_text
    except yaml.YAMLError:
        return {}, text


def chunk_markdown(text: str, source_path: str | Path | None = None) -> List[BaseNode]:
    if not text or not text.strip():
        return []
    
    path = Path(source_path) if source_path else None
    
    # Trích xuất YAML frontmatter làm metadata (không chunk)
    frontmatter_meta, text = _extract_frontmatter(text)
    
    tables, text_with_placeholders = _extract_tables(text)
    
    # Metadata cơ bản từ frontmatter + đường dẫn nguồn
    base_meta = {**frontmatter_meta}
    if path:
        base_meta.update({"source_path": str(path), "source_file": path.name})
    
    # Phân tích theo tiêu đề
    doc = Document(text=text_with_placeholders, metadata=base_meta.copy())
    heading_nodes = MarkdownNodeParser().get_nodes_from_documents([doc])
    
    nodes: List[BaseNode] = []
    for node in heading_nodes:
        content, meta = node.get_content(), node.metadata.copy()
        matches = list(TABLE_PLACEHOLDER.finditer(content))
        
        if not matches:
            nodes.extend(_chunk_text(content, meta) if len(content) > CHUNK_SIZE else [TextNode(text=content, metadata=meta)])
            continue
        
        last_end = 0
        for match in matches:
            # Văn bản trước bảng
            before_text = content[last_end:match.start()].strip()
            
            # Trích xuất số bảng và tiêu đề từ văn bản trước bảng
            table_number = ""
            table_title = ""
            if before_text:
                # Tìm mẫu như "## Bảng 3.1 Danh mục các học phần..."
                title_match = TABLE_TITLE_PATTERN.search(before_text)
                if title_match:
                    table_number = title_match.group(1).strip()
                    table_title = title_match.group(2).strip()
            
            if before_text and len(before_text) >= MIN_CHUNK_SIZE:
                nodes.extend(_chunk_text(before_text, meta) if len(before_text) > CHUNK_SIZE else [TextNode(text=before_text, metadata=meta.copy())])
            
            # Các chunk bảng - sử dụng mẫu Small-to-Big
            if (idx := int(match.group(1))) < len(tables):
                header, rows = tables[idx]
                table_chunks = _split_table(header, rows)
                
                # Lấy gợi ý ngữ cảnh từ đường dẫn tiêu đề
                context_hint = meta.get("Header 1", "") or meta.get("section", "")
                
                # Lấy tên file nguồn cho tóm tắt
                source_file = meta.get("source_file", "") or (path.name if path else "")
                
                for i, chunk in enumerate(table_chunks):
                    chunk_meta = {**meta}
                    if len(table_chunks) > 1:
                        chunk_meta["table_part"] = f"{i+1}/{len(table_chunks)}"
                    
                    # Tạo các node cha + tóm tắt nếu áp dụng được
                    table_nodes = _create_table_nodes(
                        chunk, 
                        chunk_meta, 
                        context_hint,
                        table_number=table_number,
                        table_title=table_title,
                        source_file=source_file
                    )
                    nodes.extend(table_nodes)
                    
            last_end = match.end()
        
        # Văn bản sau bảng
        if (after := content[last_end:].strip()) and len(after) >= MIN_CHUNK_SIZE:
            nodes.extend(_chunk_text(after, meta) if len(after) > CHUNK_SIZE else [TextNode(text=after, metadata=meta.copy())])

    
    final: List[BaseNode] = []
    i = 0
    while i < len(nodes):
        curr = nodes[i]
        curr_content = curr.get_content()
        curr_is_table = curr.metadata.get("is_table")
        
        # Bỏ qua các node rỗng hoặc chỉ có khoảng trắng
        if not curr_content.strip():
            i += 1
            continue
        
        # Nếu node hiện tại là non-table nhỏ và có node tiếp theo
        if not curr_is_table and len(curr_content) < MIN_CHUNK_SIZE and i + 1 < len(nodes):
            next_node = nodes[i + 1]
            next_is_table = next_node.metadata.get("is_table")
            
            if next_is_table:
                merged_text = curr_content.strip() + "\n\n" + next_node.get_content()
                merged_meta = {**curr.metadata, **next_node.metadata}
                final.append(TextNode(text=merged_text, metadata=merged_meta))
                i += 2
            else:
                merged_text = curr_content + "\n\n" + next_node.get_content()
                merged_meta = {**curr.metadata, **next_node.metadata}
                final.append(TextNode(text=merged_text, metadata=merged_meta))
                i += 2
        else:
            final.append(curr)
            i += 1
    
    for idx, node in enumerate(final):
        _enrich_metadata(node, path)
        node.metadata["chunk_index"] = idx
    
    return final


def chunk_markdown_file(path: str | Path) -> List[BaseNode]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {p}")
    return chunk_markdown(p.read_text(encoding="utf-8"), source_path=p)
