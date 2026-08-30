"""Streamlit interface for ToxiScan."""

from __future__ import annotations

import logging
from collections import Counter

import streamlit as st

from toxiscan_core.classifier import BlockAnalysis, ClassificationError, analyse_blocks
from toxiscan_core.config import DOCUMENT_LIMITS
from toxiscan_core.documents import (
    DocumentExtractionError,
    extract_document,
)
from toxiscan_core.text import ExtractedText, prepare_text
from toxiscan_core.web import WebExtractionError, scrape_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _render_results(results: tuple[BlockAnalysis, ...]) -> None:
    flagged = [result for result in results if result.flagged]
    category_counts = Counter(
        category.label for result in flagged for category in result.categories
    )

    first_metric, second_metric, third_metric = st.columns(3)
    first_metric.metric("Blocks analysed", len(results))
    second_metric.metric("Blocks flagged", len(flagged))
    third_metric.metric("Categories detected", len(category_counts))

    if not flagged:
        st.success("No content exceeded the selected detection threshold.")
        return

    st.warning(
        f"{len(flagged)} of {len(results)} analysed blocks exceeded the selected threshold."
    )
    for index, result in enumerate(flagged, start=1):
        heading = ", ".join(category.label for category in result.categories)
        with st.expander(f"{index}. {heading}", expanded=index == 1):
            st.write(result.text)
            st.markdown("**Category confidence**")
            for category in result.categories:
                st.write(f"{category.label}: {category.score:.1%}")

            if result.evidence:
                st.markdown("**Contextual evidence**")
                for evidence in result.evidence:
                    labels = ", ".join(
                        f"{category.label} ({category.score:.1%})"
                        for category in evidence.categories
                    )
                    st.info(f"{evidence.text}\n\n{labels}")


def _analyse(extracted: ExtractedText, threshold: float) -> None:
    if extracted.truncated:
        st.warning("The safe processing limit was reached; the available content was analysed.")

    with st.spinner("Loading the model and analysing context…"):
        results = analyse_blocks(extracted.blocks, threshold)
    _render_results(results)


def _render_web_input(threshold: float) -> None:
    with st.form("web-page-form"):
        url = st.text_input(
            "Public web page URL",
            placeholder="https://example.com/discussion",
            help="Public pages and accessible comment sections are analysed on a best-effort basis.",
        )
        submitted = st.form_submit_button("Scan page", type="primary")

    if not submitted:
        return
    if not url.strip():
        st.warning("Enter a public web page URL.")
        return

    with st.spinner("Loading the page and accessible comments within safe limits…"):
        scraped = scrape_url(url)
    for warning in scraped.warnings:
        st.warning(warning)
    rendering_note = " with browser rendering" if scraped.used_browser else ""
    st.caption(
        f"Extracted {len(scraped.text.blocks)} blocks from the public page{rendering_note}."
    )
    _analyse(scraped.text, threshold)


def _render_text_input(threshold: float) -> None:
    with st.form("pasted-text-form"):
        value = st.text_area(
            "Text to analyse",
            height=240,
            placeholder="Paste comments, messages, or longer text here…",
        )
        submitted = st.form_submit_button("Analyse text", type="primary")

    if not submitted:
        return
    extracted = prepare_text(value, DOCUMENT_LIMITS.text)
    if not extracted.blocks:
        st.warning("Enter some readable text to analyse.")
        return
    _analyse(extracted, threshold)


def _render_document_input(threshold: float) -> None:
    with st.form("document-form"):
        upload = st.file_uploader(
            "Upload a document",
            max_upload_size=DOCUMENT_LIMITS.max_upload_bytes // (1024 * 1024),
            help="ToxiScan accepts a wide range of documents and extracts readable text locally.",
        )
        submitted = st.form_submit_button("Analyse document", type="primary")

    if not submitted:
        return
    if upload is None:
        st.warning("Choose a document to analyse.")
        return

    with st.spinner("Reading and validating the document…"):
        extracted = extract_document(upload.name, upload.getvalue())
    st.caption(f"Extracted {len(extracted.blocks)} readable blocks from the document.")
    _analyse(extracted, threshold)


def main() -> None:
    st.set_page_config(page_title="ToxiScan", page_icon="🛡️", layout="wide")
    st.title("🛡️ ToxiScan")
    st.write(
        "Analyse pasted text, public web pages, accessible comments, and a wide range "
        "of documents for potentially toxic language."
    )

    with st.sidebar:
        st.header("Detection settings")
        threshold = st.slider(
            "Detection threshold",
            min_value=0.30,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Lower values flag more content; higher values require stronger model confidence.",
        )
        st.caption(
            "Automated scores are indicators, not definitive judgements. Review context before acting."
        )

    mode = st.radio(
        "Choose an input source",
        ("Web page", "Pasted text", "Document"),
        horizontal=True,
    )

    try:
        if mode == "Web page":
            _render_web_input(threshold)
        elif mode == "Pasted text":
            _render_text_input(threshold)
        else:
            _render_document_input(threshold)
    except (DocumentExtractionError, WebExtractionError, ClassificationError) as error:
        st.error(str(error))
    except Exception:
        logger.exception("Unexpected application failure", extra={"input_mode": mode})
        st.error("ToxiScan could not complete the analysis. Please try again later.")


if __name__ == "__main__":
    main()
