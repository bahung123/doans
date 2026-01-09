from __future__ import annotations
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.rag.retrival import Retriever


# System prompt để sử dụng khi gọi LLM (export cho gradio/eval dùng)
SYSTEM_PROMPT = """Bạn là Trợ lý học vụ Đại học Bách khoa Hà Nội.

## NGUYÊN TẮC:
1. Chỉ được đưa ra câu trả lời dựa trên CONTEXT được cung cấp. Không suy đoán, không bổ sung thông tin ngoài CONTEXT.
2. Nếu CONTEXT chứa nhiều văn bản khác nhau, ưu tiên nội dung mới nhất, TRỪ KHI có điều khoản chuyển tiếp nói khác.
3. Nếu không tìm thấy thông tin trong CONTEXT, trả lời: "Không tìm thấy thông tin trong dữ liệu hiện có."
"""


def build_context(results: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        source = meta.get("source_file", "N/A")
        header = meta.get("header_path", "")
        doc_type = meta.get("document_type", "")
        cohorts = meta.get("applicable_cohorts", "")
        program = meta.get("program_name", "")
        program_code = meta.get("program_code", "")
        faculty = meta.get("faculty", "")
        degree_levels = meta.get("degree_levels", [])
        issued_year = meta.get("issued_year", "")
        content = r.get("content", "").strip()
        
        # Build metadata line
        meta_info = f"Nguồn: {source}"
        if header and header != "/":
            meta_info += f" | Mục: {header}"
        if doc_type:
            meta_info += f" | Loại: {doc_type}"
        if issued_year:
            meta_info += f" | Năm: {issued_year}"
        if cohorts:
            meta_info += f" | Áp dụng: {cohorts}"
        if program:
            meta_info += f" | CTĐT: {program}"
        if program_code:
            meta_info += f" | Mã: {program_code}"
        if faculty:
            meta_info += f" | Khoa: {faculty}"
        if degree_levels:
            levels = ", ".join(degree_levels) if isinstance(degree_levels, list) else degree_levels
            meta_info += f" | Bậc: {levels}"
        
        parts.append(f"[TÀI LIỆU {i}]\n{meta_info}\n{content}")

    context = "\n---\n".join(parts)
    return context[:max_chars] if len(context) > max_chars else context


def build_prompt(question: str, context: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n## CONTEXT:\n{context}\n\n## CÂU HỎI: {question}\n\n## TRẢ LỜI:"


class RAGContextBuilder:
    
    def __init__(self, retriever: "Retriever", max_context_chars: int = 8000):
        self._retriever = retriever
        self._max_context_chars = max_context_chars

    def retrieve_and_prepare(
        self, 
        question: str, 
        k: int = 5, 
        initial_k: int = 20, 
        mode: str = "hybrid_rerank"
    ) -> Dict[str, Any]:

        results = self._retriever.flexible_search(question, k=k, initial_k=initial_k, mode=mode)

        if not results:
            return {
                "results": [],
                "contexts": [],
                "context_text": "",
                "prompt": "",
            }

        context_text = build_context(results, self._max_context_chars)
        prompt = build_prompt(question, context_text)

        return {
            "results": results,
            "contexts": [r.get("content", "")[:1000] for r in results],
            "context_text": context_text,
            "prompt": prompt,
        }


RAGGenerator = RAGContextBuilder
