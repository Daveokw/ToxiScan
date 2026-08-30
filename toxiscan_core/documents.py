"""Safe, bounded text extraction from uploaded documents."""

from __future__ import annotations

import logging
import zipfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from .config import DOCUMENT_LIMITS, DocumentLimits
from .text import ExtractedText, prepare_text

logger = logging.getLogger(__name__)

# This is intentionally an allow-list: converters should never guess how to open
# executable or unknown binary input supplied to the hosted application.
SUPPORTED_DOCUMENT_EXTENSIONS = frozenset(
    {
        ".csv",
        ".docx",
        ".epub",
        ".htm",
        ".html",
        ".json",
        ".jsonl",
        ".md",
        ".msg",
        ".pdf",
        ".pptx",
        ".rst",
        ".tsv",
        ".txt",
        ".xls",
        ".xlsx",
        ".xml",
        ".yaml",
        ".yml",
    }
)

_ZIP_BASED_EXTENSIONS = frozenset({".docx", ".epub", ".pptx", ".xlsx"})
_OLE_BASED_EXTENSIONS = frozenset({".msg", ".xls"})
_MAX_ARCHIVE_ENTRIES = 2_000
_MAX_ARCHIVE_EXPANDED_BYTES = 50 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 100


class DocumentExtractionError(ValueError):
    """Raised when an upload cannot be converted safely."""


class UnsupportedDocumentError(DocumentExtractionError):
    """Raised when the uploaded document type is not accepted."""


class DocumentTooLargeError(DocumentExtractionError):
    """Raised when upload or archive limits are exceeded."""


@lru_cache(maxsize=1)
def _get_converter():
    # Importing all built-in converters is relatively expensive, so defer it until
    # a user actually submits a document and then reuse the instance.
    from markitdown import MarkItDown

    return MarkItDown(enable_builtins=True, enable_plugins=False)


def _validate_archive(data: bytes) -> None:
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > _MAX_ARCHIVE_ENTRIES:
                raise DocumentTooLargeError("The document contains too many embedded files.")

            expanded_size = 0
            for entry in entries:
                expanded_size += entry.file_size
                if expanded_size > _MAX_ARCHIVE_EXPANDED_BYTES:
                    raise DocumentTooLargeError(
                        "The document expands beyond the safe processing limit."
                    )
                if entry.compress_size == 0:
                    ratio = entry.file_size if entry.file_size else 1
                else:
                    ratio = entry.file_size / entry.compress_size
                if ratio > _MAX_COMPRESSION_RATIO:
                    raise DocumentTooLargeError(
                        "The document contains an unsafe compressed entry."
                    )
    except zipfile.BadZipFile as error:
        raise DocumentExtractionError("The document archive is malformed.") from error


def _validate_signature(extension: str, data: bytes) -> None:
    if extension == ".pdf" and not data.startswith(b"%PDF-"):
        raise DocumentExtractionError("The uploaded file is not a valid PDF document.")
    if extension in _ZIP_BASED_EXTENSIONS:
        if not data.startswith(b"PK"):
            raise DocumentExtractionError("The uploaded document has an invalid signature.")
        _validate_archive(data)
    if extension in _OLE_BASED_EXTENSIONS and not data.startswith(
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    ):
        raise DocumentExtractionError("The uploaded document has an invalid signature.")


def extract_document(
    filename: str,
    data: bytes,
    limits: DocumentLimits = DOCUMENT_LIMITS,
) -> ExtractedText:
    """Convert one validated in-memory upload into bounded text blocks."""

    extension = Path(filename).suffix.casefold()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise UnsupportedDocumentError(
            "This document type is not supported. Please choose another document."
        )
    if not data:
        raise DocumentExtractionError("The uploaded document is empty.")
    if len(data) > limits.max_upload_bytes:
        raise DocumentTooLargeError("The uploaded document exceeds the processing limit.")

    _validate_signature(extension, data)

    try:
        converted = _get_converter().convert_stream(
            BytesIO(data),
            file_extension=extension,
        )
        text = converted.text_content
    except DocumentExtractionError:
        raise
    except Exception as error:
        logger.exception(
            "Document conversion failed",
            extra={"document_extension": extension, "document_size": len(data)},
        )
        raise DocumentExtractionError(
            "The document could not be read safely. It may be damaged or protected."
        ) from error

    result = prepare_text(text, limits.text)
    if not result.blocks:
        raise DocumentExtractionError("No readable text was found in the document.")
    return result
