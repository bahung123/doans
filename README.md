# HUST RAG - Hệ thống Hỏi đáp Quy chế Sinh viên

Hệ thống RAG hỗ trợ sinh viên tra cứu quy chế, quy định tại Đại học Bách khoa Hà Nội.

## Tính năng

- Hybrid Search (Vector + BM25)
- Reranking với Qwen3-Reranker
- Small-to-Big Retrieval cho bảng biểu
- Giao diện chat Gradio

## Cài đặt

**Yêu cầu:** Python 3.10+

**Bước 1:** Chạy setup script

- **Linux/Mac:** `bash setup.sh`
- **Windows:** nhấp đúp `setup.bat` hoặc gõ `setup.bat` trong cmd

> Script sẽ: tạo venv → cài dependencies → tải data → tạo .env

**Bước 2:** Cấu hình API keys

Sửa file `.env`:
```
SILICONFLOW_API_KEY=your_key    # Embedding & Reranking
GROQ_API_KEY=your_key           # LLM Generation
```
Lấy API keys tại: [SiliconFlow](https://siliconflow.ai/) | [Groq](https://groq.com/)

**Bước 3:** Chạy ứng dụng
```
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

python scripts/run_app.py
```

Truy cập: http://127.0.0.1:7860

## Data

Data trên HuggingFace: [hungnha/do_an_tot_nghiep](https://huggingface.co/datasets/hungnha/do_an_tot_nghiep)

Tải thủ công:
```
huggingface-cli download hungnha/do_an_tot_nghiep --repo-type dataset --local-dir ./data
```

## Tác giả

**Nguyễn Hùng Ba** - Đại học Bách khoa Hà Nội
