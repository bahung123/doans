import os
import sys
import random
import shutil
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.preprocessing.docling_processor import DoclingProcessor

def get_random_local_pdf(source_dir: str):
    """Lấy ngẫu nhiên 1 file PDF từ thư mục local."""
    if not os.path.exists(source_dir):
        return None
    
    files = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]
    if not files:
        return None
    
    return os.path.join(source_dir, random.choice(files))

def main(output_dir=None, use_ocr=False):
    """Test Docling với 1 file PDF ngẫu nhiên."""
    
    # Setup paths
    source_dir = os.path.join(_PROJECT_ROOT, "data", "files")
    if output_dir is None:
        output_dir = os.path.join(_PROJECT_ROOT, "data", "test_output")
    
    # Clean up old test output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔍 Đang tìm file PDF để test...")
    
    # 1. Thử lấy từ local data/files
    file_path = get_random_local_pdf(source_dir)
    
    if not file_path:
        print(f"❌ Không tìm thấy file PDF nào trong {source_dir}")
        print("💡 Hãy chạy 'python core/hash_file/hash_data_goc.py' để tải dữ liệu trước.")
        return 1
        
    filename = os.path.basename(file_path)
    print(f"🎯 Đã chọn file test: {filename}")
    print(f"📂 Đường dẫn: {file_path}")
    
    try:
        # Khởi tạo processor
        print("\n⚙️  Khởi tạo DoclingProcessor...")
        processor = DoclingProcessor(
            output_dir=output_dir,
            use_ocr=use_ocr,
            timeout=None
        )
        
        # Parse file
        print(f"🚀 Bắt đầu parse...")
        result = processor.parse_document(file_path)
        
        if result:
            print(f"\n✅ Test thành công!")
            
            # Kiểm tra kết quả
            output_files = os.listdir(output_dir)
            md_files = [f for f in output_files if f.endswith('.md')]
            
            if md_files:
                print(f"📄 File output: {md_files[0]}")
                print(f"📁 Thư mục output: {output_dir}")
                
                # In thống kê sơ bộ cho Markdown
                content_len = len(result)
                preview = result[:200].replace('\n', ' ') + "..."
                print(f" Kích thước: {content_len} ký tự")
                print(f" Preview: {preview}")
            else:
                print("  Không tìm thấy file Markdown output dù hàm trả về kết quả.")
        else:
            print(f"\n❌ Test thất bại: Hàm parse trả về None")
            return 1
            
        return 0
        
    except Exception as e:
        print(f"\n❌ Lỗi ngoại lệ: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Docling với 1 file PDF ngẫu nhiên từ data/files")
    parser.add_argument("--output", help="Thư mục output cho test (mặc định: data/test_output)")
    parser.add_argument("--ocr", action="store_true", help="Bật OCR")
    args = parser.parse_args()

    sys.exit(main(
        output_dir=args.output,
        use_ocr=args.ocr
    ))
