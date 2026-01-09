from docling_processor import DoclingProcessor

PDF_FILE = "data/data_raw/quyet_dinh/quy-dinh-chuan-ngoai-ngu-2021.pdf"
SOURCE_DIR = "data/data_raw"
OUTPUT_DIR = "data"
USE_OCR = False

if __name__ == "__main__":
    processor = DoclingProcessor(OUTPUT_DIR, use_ocr=USE_OCR)
    
    if PDF_FILE:
        print(f"Đang xử lý: {PDF_FILE}")
        result = processor.parse_document(PDF_FILE)
        print(f"Hoàn tất: {result}" if result else "Bỏ qua/lỗi")
    else:
        print(f"Đang xử lý thư mục: {SOURCE_DIR}")
        r = processor.parse_directory(SOURCE_DIR)
        print(f"Tổng: {r['total']} | Thành công: {r['parsed']} | Bỏ qua: {r['skipped']} | Lỗi: {r['errors']}")
