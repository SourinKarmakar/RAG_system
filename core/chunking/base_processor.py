import re
import os
from .docx_processing import DocxProcessor
from .pdf_processing import PDFProcessor
from .text_processing import TextProcessor


class BaseProcessor:
    def __init__(self):
        self.processors = {
            "txt": TextProcessor(),
            "docx": DocxProcessor(),
            "pdf": PDFProcessor(),
        }


    def normalize_blocks(self, blocks):
        """
        Ensure all incoming data follows a uniform structure:
        [
        {"heading": str, "content": str, "type": "text"|"table"|"list"},
        ...
        ]
        """
        normalized = []
        for b in blocks:
            normalized.append({
                "heading": b.get("heading", "Document"),
                "content": b.get("content", ""),
                "type": b.get("type", "text")
            })
        return normalized


    def sentence_splitter(self, text):
        """Split text by sentence endings while keeping punctuation."""
        return re.split(r'(?<=[.!?])\s+', text.strip())


    def chunk_block(self, block, max_words = 150, overlap = 50):
        """
        Chunk a single block of text (or keep table/list intact).
        """
        if block["type"] != "text":
            # Non-text content (e.g., table) should be kept as-is
            return [block]

        sentences = self.sentence_splitter(block["content"])
        chunks = []
        current = []

        for sent in sentences:
            sent_words = sent.split()
            current_len = sum(len(s.split()) for s in current)
            if current_len + len(sent_words) <= max_words:
                current.append(sent)
            else:
                # finalize chunk
                if current:
                    chunk_text = " ".join(current)
                    chunks.append({
                        "heading": block["heading"],
                        "content": chunk_text,
                        "type": block["type"]
                    })
                    # create overlap (keep last few sentences)
                    current = current[-(overlap // 15):] + [sent]

        if current:
            chunks.append({
                "heading": block["heading"],
                "content": " ".join(current),
                "type": block["type"]
            })

        return chunks


    def merge_small_chunks(self, chunks, min_words = 50):
        """
        Merge chunks that are too small with their neighbors.
        """
        merged = []
        buffer = None

        for c in chunks:
            word_count = len(c["content"].split())
            if word_count < min_words and merged:
                # merge with previous chunk
                merged[-1]["content"] += " " + c["content"]
            else:
                merged.append(c)
        return merged


    def unified_context_chunker(self, blocks, max_words = 150, overlap = 50):
        """
        Core unified algorithm for grouping headers and chunking across formats.
        """
        normalized = self.normalize_blocks(blocks)
        all_chunks = []

        for block in normalized:
            sub_chunks = self.chunk_block(block, max_words=max_words, overlap=overlap)
            all_chunks.extend(sub_chunks)

        final_chunks = self.merge_small_chunks(all_chunks)
        return final_chunks
    
    
    def _detect_file_type(self, file_path):
        """Detect file type based on extension."""
        ext = os.path.splitext(file_path)[-1].lower().replace(".", "")
        return ext
    
    
    def process_file(self, file_path):
        """
        Detect file type and process using the corresponding processor.
        Returns: List of {heading, content, type} dicts.
        """
        file_type = self._detect_file_type(file_path)

        if file_type not in self.processors:
            raise ValueError(f"Unsupported file type: {file_type}")

        processor = self.processors[file_type]

        try:
            sections = processor.process_file(file_path)
        except Exception as e:
            raise RuntimeError(f"Error processing {file_path}: {e}")

        return sections
    
    
    
