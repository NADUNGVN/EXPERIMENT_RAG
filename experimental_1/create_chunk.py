import os
import json
import logging
import fitz  # PyMuPDF
import re
from datetime import datetime
from typing import Dict, List
from pathlib import Path

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Định nghĩa thư mục chứa file PDF
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = os.path.join(BASE_DIR, "data", "pdf")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "chunks")

# Cấu hình chunk
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100

# Mapping ý định với từng loại mục
INTENT_MAPPING = {
    "Trình tự thực hiện": "trình tự",
    "Cách thức thực hiện": "cách thức",
    "Thành phần hồ sơ": "hồ sơ",
    "Thời hạn giải quyết": "thời hạn",
    "Phí, lệ phí": "phí",
    "Căn cứ pháp lý": "pháp lý",
    "Yêu cầu, điều kiện thực hiện": "điều kiện"
}

class TextProcessor:
    def split_into_sections(self, text: str) -> Dict[str, str]:
        """Tách văn bản thành các phần khác nhau dựa trên tiêu đề."""
        sections = {}
        section_headers = list(INTENT_MAPPING.keys())

        current_section = "Thông tin chung"
        current_content = []

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            for header in section_headers:
                if re.match(header, line, re.IGNORECASE):
                    if current_content:
                        sections[current_section] = "\n".join(current_content)
                    current_section = header
                    current_content = []
                    break
            else:
                current_content.append(line)

        if current_content:
            sections[current_section] = "\n".join(current_content)

        return sections

    def process_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Xử lý văn bản và chia nhỏ thành các chunk."""
        sections = self.split_into_sections(text)
        chunks = []

        for section_name, content in sections.items():
            intent = INTENT_MAPPING.get(section_name, "khác")

            chunk_metadata = {**metadata, "section_name": section_name, "intent": intent}
            chunks.append({"content": content.strip(), "metadata": chunk_metadata})

        return chunks

def read_pdf_files(pdf_dir: str) -> Dict[str, str]:
    """Đọc nội dung PDF từ thư mục."""
    pdf_contents = {}

    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_dir, filename)
            with fitz.open(file_path) as doc:
                text = "\n".join([page.get_text() for page in doc])
                pdf_contents[filename] = text
                logger.info(f"Đã đọc file: {filename}")

    return pdf_contents

def main():
    pdf_contents = read_pdf_files(PDF_DIR)
    processor = TextProcessor()

    for file_name, content in pdf_contents.items():
        metadata = {"file_name": file_name, "processed_date": datetime.now().isoformat()}
        chunks = processor.process_text(content, metadata)

        output_file = os.path.join(OUTPUT_DIR, f"{file_name}_chunks.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"Đã lưu {len(chunks)} chunks vào {output_file}")

if __name__ == "__main__":
    main()
