# PDF Processor
from utility.custom_thread import ThreadWithReturnValue
import pdfplumber
import fitz  # PyMuPDF
import re

class PDFProcessor:
    def __init__(self):
        pass
    

    def extract_text_blocks(self, pdf_path):
        print("extract text started")
        """Extract text blocks and detect headings using PyMuPDF."""
        doc = fitz.open(pdf_path)
        blocks = []
        for page_num, page in enumerate(doc, start=1):
            for b in page.get_text("dict")["blocks"]:
                if "lines" not in b:
                    continue
                text = " ".join(span["text"] for line in b["lines"] for span in line["spans"]).strip()
                if not text:
                    continue
                font_sizes = [span["size"] for line in b["lines"] for span in line["spans"]]
                avg_font = sum(font_sizes) / len(font_sizes)
                blocks.append({
                    "page": page_num,
                    "text": text,
                    "font": avg_font
                })
        print("extract text ended")
        return blocks


    def extract_tables(self, pdf_path):
        """Extract tables as markdown chunks using pdfplumber."""
        print("extract table started")
        tables_md = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for t in tables:
                    md = self.table_to_markdown(t)
                    if md:
                        tables_md.append({
                            "heading": f"Table (Page {page_num})",
                            "content": md,
                            "type": "table"
                        })
        print("extract table ended")
        return tables_md


    def process_file(self, pdf_path, max_words=150, overlap=50):
        """Combine text + tables into context-aware chunks."""
        thread1 = ThreadWithReturnValue(target=self.extract_text_blocks, args=(pdf_path,))
        thread2 = ThreadWithReturnValue(target=self.extract_tables, args=(pdf_path,))

        thread1.start()
        thread2.start()

        blocks = thread1.join()
        tables = thread2.join()

        avg_font = sum(b["font"] for b in blocks) / len(blocks)
        chunks, buffer, heading = [], [], "Introduction"

        def add_chunk():
            text = " ".join(buffer).strip()
            if text:
                words = text.split()
                for i in range(0, len(words), max_words - overlap):
                    segment = " ".join(words[i:i + max_words])
                    chunks.append({
                        "heading": heading,
                        "content": segment,
                        "type": "text"
                    })

        for blk in blocks:
            if blk["font"] > avg_font * 1.2:  # detect heading
                add_chunk()
                heading = blk["text"]
                buffer = []
            else:
                buffer.append(blk["text"])
        add_chunk()

        chunks.extend(tables)
        return chunks
    
    
    def table_to_markdown(self, table_blocks):
        rows = []
        for blk in table_blocks:
            row = re.split(r'\s{2,}', blk["text"].strip())  # split on large gaps
            rows.append("| " + " | ".join(row) + " |")
        return "\n".join(rows)
