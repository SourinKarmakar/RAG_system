import re


class TextProcessor:
    def detect_sections(self, lines):
        """
        Identify sections based on uppercase headings, numbered headings, or blank lines.
        """
        sections = []
        current_heading = "Introduction"
        buffer = []

        for line in lines:
            clean_line = line.strip()
            for n in range(30, 4, -1):
                clean_line = clean_line.replace("-"*n, "")
            # Detect heading-like patterns
            if re.match(r"^[A-Z0-9 ._-]{4,}$", clean_line) and len(clean_line.split()) < 10:
                # Save previous section
                if buffer:
                    sections.append({"heading": current_heading, "content": "\n".join(buffer).strip(), "type": "text"})
                    buffer = []
                current_heading = clean_line
            elif clean_line == "":
                if buffer:
                    sections.append({"heading": current_heading, "content": "\n".join(buffer).strip(), "type": "text"})
                    buffer = []
            else:
                buffer.append(clean_line)

        # Add remaining text
        if buffer:
            sections.append({"heading": current_heading, "content": "\n".join(buffer).strip(), "type": "text"})
        return sections


    def process_file(self, file_path):
        """
        Perform context-aware chunking for plain text files.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        sections = self.detect_sections(lines)

        return sections
