import os
import re
import gc
import signal
import logging
from datetime import datetime
from pathlib import Path

from docling.document_converter import DocumentConverter, FormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions, EasyOcrOptions, TableFormerMode
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


class DoclingProcessor:
    def __init__(self, output_dir: str, use_ocr: bool = True, timeout: int = 300, images_scale: float = 3.0):
        self.output_dir = output_dir
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
        
        # Cấu hình pipeline
        opts = PdfPipelineOptions(do_ocr=use_ocr, do_table_structure=True)
        opts.table_structure_options = TableStructureOptions(do_cell_matching=True, mode=TableFormerMode.ACCURATE)
        opts.images_scale = images_scale
        
        if use_ocr:
            ocr = EasyOcrOptions()
            ocr.lang = ["vi"]  # EasyOCR dùng "vi" cho tiếng Việt
            ocr.force_full_page_ocr = False
            opts.ocr_options = ocr

        self.converter = DocumentConverter(format_options={
            InputFormat.PDF: FormatOption(backend=PyPdfiumDocumentBackend, pipeline_cls=StandardPdfPipeline, pipeline_options=opts)
        })
        self.logger.info(f"Docling | OCR={use_ocr} | Bảng=chính xác | Tỷ lệ={images_scale} | Timeout={timeout}s")
    
    def clean_markdown(self, text: str) -> str:
        text = re.sub(r'\n\s*Trang\s+\d+\s*\n', '\n', text)
        return re.sub(r'\n{3,}', '\n\n', text).strip()
    
    def parse_document(self, file_path: str) -> str | None:
        if not os.path.exists(file_path):
            return None
        filename = os.path.basename(file_path)
        try:
            signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(self.timeout)
            result = self.converter.convert(file_path)
            md = result.document.export_to_markdown(image_placeholder="")
            signal.alarm(0)
            md = self.clean_markdown(md)
            return f"---\nfilename: {filename}\nfilepath: {file_path}\npage_count: {len(result.document.pages)}\nprocessed_at: {datetime.now().isoformat()}\n---\n\n{md}"
        except TimeoutError:
            self.logger.warning(f"Quá thời gian: {filename}")
            signal.alarm(0)
            return None
        except Exception as e:
            self.logger.error(f"Lỗi xử lý: {filename}: {e}")
            signal.alarm(0)
            return None
    
    def parse_directory(self, source_dir: str) -> dict:
        source_path = Path(source_dir)
        pdf_files = list(source_path.rglob("*.pdf"))
        self.logger.info(f"Tìm thấy {len(pdf_files)} file PDF trong {source_dir}")
        
        results = {"total": len(pdf_files), "parsed": 0, "skipped": 0, "errors": 0}
        for i, fp in enumerate(pdf_files):
            try:
                rel = fp.relative_to(source_path)
            except ValueError:
                rel = Path(fp.name)
            out = Path(self.output_dir) / rel.with_suffix(".md")
            out.parent.mkdir(parents=True, exist_ok=True)
            
            if out.exists():
                results["skipped"] += 1
                continue
            
            md = self.parse_document(str(fp))
            if md:
                out.write_text(md, encoding="utf-8")
                results["parsed"] += 1
            else:
                results["errors"] += 1
            
            if (i + 1) % 10 == 0:
                gc.collect()
                self.logger.info(f"Tiến độ: {i+1}/{len(pdf_files)} (bỏ qua: {results['skipped']})")
        
        self.logger.info(f"Hoàn tất: {results['parsed']} đã xử lý, {results['skipped']} bỏ qua, {results['errors']} lỗi")
        return results
