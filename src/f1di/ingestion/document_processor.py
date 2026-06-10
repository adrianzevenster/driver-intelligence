from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("f1di.ingestion.document_processor")


@dataclass
class KnowledgeDocument:
    source_id: str
    title: str
    text: str
    metadata: dict


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text.strip()


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


class DocumentProcessor:
    def process(self, content: bytes, filename: str, metadata: dict | None = None) -> list[KnowledgeDocument]:
        meta = metadata or {}
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return self._process_pdf(content, filename, meta)
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
            return self._process_image(content, filename, meta)
        else:
            return self._process_text(content, filename, meta)

    def _process_pdf(self, content: bytes, filename: str, meta: dict) -> list[KnowledgeDocument]:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pip install 'f1-driver-intelligence[ocr]'") from None

        uid = _content_hash(content)
        stem = Path(filename).stem
        parts: list[str] = []

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    parts.append(_clean(page_text))
                for table in page.extract_tables():
                    if not table:
                        continue
                    rows = [" | ".join(str(c or "").strip() for c in row) for row in table if any(row)]
                    if rows:
                        parts.append("\n".join(rows))

        full_text = "\n\n".join(p for p in parts if p)
        if not full_text.strip():
            return []

        chunks = self._chunk(full_text)
        return [
            KnowledgeDocument(
                source_id=f"{stem}_{uid}_p{i}",
                title=f"{stem} ({i + 1}/{len(chunks)})" if len(chunks) > 1 else stem,
                text=chunk,
                metadata={"source": "uploaded_pdf", "filename": filename, **meta},
            )
            for i, chunk in enumerate(chunks)
        ]

    def _process_image(self, content: bytes, filename: str, meta: dict) -> list[KnowledgeDocument]:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError(
                "pip install pytesseract Pillow && sudo apt install tesseract-ocr"
            ) from None

        img = Image.open(io.BytesIO(content))
        if max(img.size) < 1000:
            scale = 1000 / max(img.size)
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

        text = pytesseract.image_to_string(img, config="--psm 6")
        if not text.strip():
            return []

        uid = _content_hash(content)
        stem = Path(filename).stem
        chunks = self._chunk(_clean(text))
        return [
            KnowledgeDocument(
                source_id=f"{stem}_{uid}_c{i}",
                title=f"{stem} ({i + 1})" if len(chunks) > 1 else stem,
                text=chunk,
                metadata={"source": "uploaded_image", "filename": filename, **meta},
            )
            for i, chunk in enumerate(chunks)
        ]

    def _process_text(self, content: bytes, filename: str, meta: dict) -> list[KnowledgeDocument]:
        text = content.decode("utf-8", errors="replace")
        uid = _content_hash(content)
        stem = Path(filename).stem
        chunks = self._chunk(_clean(text))
        return [
            KnowledgeDocument(
                source_id=f"{stem}_{uid}_c{i}",
                title=f"{stem} ({i + 1})" if len(chunks) > 1 else stem,
                text=chunk,
                metadata={"source": "uploaded_text", "filename": filename, **meta},
            )
            for i, chunk in enumerate(chunks)
        ]

    @staticmethod
    def _chunk(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            boundary = chunk.rfind(". ")
            if boundary > chunk_size // 2:
                chunk = text[start : start + boundary + 1]
                end = start + boundary + 1
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - overlap
        return chunks
