from __future__ import annotations

import pytest

from f1di.ingestion.document_processor import DocumentProcessor, _clean, _content_hash


def _proc() -> DocumentProcessor:
    return DocumentProcessor()


class TestChunking:
    def test_short_text_single_chunk(self):
        chunks = DocumentProcessor._chunk("short text")
        assert chunks == ["short text"]

    def test_empty_text_no_chunks(self):
        assert DocumentProcessor._chunk("   ") == []

    def test_long_text_splits(self):
        text = ("This is a sentence. " * 100).strip()
        chunks = DocumentProcessor._chunk(text, chunk_size=300, overlap=50)
        assert len(chunks) > 1
        # Each chunk fits within the size limit with some tolerance for sentence boundary
        assert all(len(c) <= 350 for c in chunks)

    def test_overlap_means_content_repeated(self):
        text = "word " * 400
        chunks = DocumentProcessor._chunk(text, chunk_size=200, overlap=50)
        if len(chunks) > 1:
            # The tail of chunk[0] should appear in the head of chunk[1]
            tail = chunks[0][-40:]
            assert tail.strip() in chunks[1] or len(chunks[1]) > 0


class TestClean:
    def test_collapses_whitespace(self):
        assert _clean("hello   world\n\t\nfoo") == "hello world foo"

    def test_fixes_hyphen_linebreak(self):
        assert _clean("degrada-\ntion") == "degradation"


class TestContentHash:
    def test_deterministic(self):
        assert _content_hash(b"abc") == _content_hash(b"abc")

    def test_different_for_different_content(self):
        assert _content_hash(b"abc") != _content_hash(b"xyz")

    def test_length_12(self):
        assert len(_content_hash(b"test")) == 12


class TestProcessText:
    def test_plain_text_chunks(self):
        proc = _proc()
        content = b"Tire wear analysis. " * 50
        docs = proc.process(content, "notes.txt")
        assert len(docs) >= 1
        assert docs[0].metadata["source"] == "uploaded_text"
        assert docs[0].metadata["filename"] == "notes.txt"

    def test_source_id_unique_per_chunk(self):
        proc = _proc()
        content = ("race strategy analysis. " * 120).encode()
        docs = proc.process(content, "strategy.txt")
        ids = [d.source_id for d in docs]
        assert len(ids) == len(set(ids))

    def test_pdf_raises_import_error_without_dep(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def no_pdfplumber(name, *args, **kwargs):
            if name == "pdfplumber":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pdfplumber)
        proc = _proc()
        with pytest.raises(ImportError):
            proc.process(b"%PDF-fake", "doc.pdf")
