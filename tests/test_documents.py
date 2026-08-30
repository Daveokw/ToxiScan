import zipfile
from io import BytesIO
from types import SimpleNamespace

import pytest

from toxiscan_core.config import DocumentLimits, TextLimits
from toxiscan_core.documents import (
    DocumentExtractionError,
    DocumentTooLargeError,
    UnsupportedDocumentError,
    extract_document,
)


class StubConverter:
    def __init__(self, text: str = "Converted text") -> None:
        self.text = text
        self.calls: list[tuple[bytes, str | None]] = []

    def convert_stream(self, stream, *, file_extension=None):
        self.calls.append((stream.read(), file_extension))
        return SimpleNamespace(text_content=self.text)


def test_extract_document_uses_bounded_stream_conversion(monkeypatch):
    converter = StubConverter("First paragraph\n\nSecond paragraph")
    monkeypatch.setattr("toxiscan_core.documents._get_converter", lambda: converter)

    result = extract_document("notes.txt", b"source bytes")

    assert result.blocks == ("First paragraph", "Second paragraph")
    assert converter.calls == [(b"source bytes", ".txt")]


def test_extract_document_rejects_unknown_type_before_conversion(monkeypatch):
    converter = StubConverter()
    monkeypatch.setattr("toxiscan_core.documents._get_converter", lambda: converter)

    with pytest.raises(UnsupportedDocumentError):
        extract_document("program.exe", b"MZ data")

    assert converter.calls == []


def test_extract_document_rejects_empty_and_oversized_uploads():
    limits = DocumentLimits(max_upload_bytes=3, text=TextLimits())

    with pytest.raises(DocumentExtractionError, match="empty"):
        extract_document("notes.txt", b"", limits)
    with pytest.raises(DocumentTooLargeError, match="exceeds"):
        extract_document("notes.txt", b"four", limits)


def test_extract_document_rejects_mismatched_pdf_signature():
    with pytest.raises(DocumentExtractionError, match="valid PDF"):
        extract_document("report.pdf", b"not a pdf")


def test_extract_document_rejects_malformed_office_archive():
    with pytest.raises(DocumentExtractionError, match="archive is malformed"):
        extract_document("report.docx", b"PK invalid archive")


def test_extract_document_rejects_high_compression_ratio_archive():
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("content.xml", "0" * 100_000)

    with pytest.raises(DocumentTooLargeError, match="unsafe compressed entry"):
        extract_document("report.docx", payload.getvalue())


def test_extract_document_rejects_conversion_without_readable_text(monkeypatch):
    monkeypatch.setattr(
        "toxiscan_core.documents._get_converter",
        lambda: StubConverter(" \n\t "),
    )

    with pytest.raises(DocumentExtractionError, match="No readable text"):
        extract_document("notes.txt", b"content")
