# ToxiScan

ToxiScan analyses pasted text, public web pages, accessible comments, and a wide range of documents for potentially toxic language. It is built with PyTorch, Hugging Face Transformers, Streamlit, Selenium, Beautiful Soup, and MarkItDown.

## Live demo

Try [ToxiScan on Streamlit](https://toxiscan.streamlit.app/).

## What it does

- Accepts pasted text and long-form content.
- Extracts readable text from a wide range of uploaded documents.
- Renders public pages, performs bounded scrolling, and loads accessible comment controls where possible.
- Falls back to the page's initial content if browser rendering is unavailable.
- Analyses long passages using overlapping token windows rather than silently discarding text after the model limit.
- Reports seven moderation categories with confidence scores.
- Shows contextual sentence or clause evidence instead of classifying isolated words without context.
- Provides an adjustable detection threshold with clear sensitivity guidance.

ToxiScan uses [`unitary/unbiased-toxic-roberta`](https://huggingface.co/unitary/unbiased-toxic-roberta), a multi-label classifier trained with unintended-bias evaluation. Identity-attribute outputs from the model are excluded from moderation results; mentioning an identity is not treated as a toxic category.

## Web extraction boundaries

Web extraction is best effort. ToxiScan can read content that is publicly available in the rendered page, including many visible comment sections, but it does not bypass authentication, paywalls, CAPTCHAs, access controls, platform restrictions, or private APIs. It honours published crawler rules and rejects private, loopback, link-local, and metadata-service destinations.

Default safeguards include:

| Boundary | Default |
| --- | ---: |
| Page download | 5 MiB |
| Render time | 30 seconds |
| Scroll passes | 20 |
| Comment expansion controls | 5 |
| Extracted page text | 75,000 characters |
| Extracted document text | 100,000 characters |
| Uploaded document | 20 MiB |
| Text blocks per source | 250 |

These limits prevent unusually large pages or documents from exhausting a CPU-hosted Streamlit process. When a content limit is reached, the interface reports that the available portion was analysed.

## Analysis pipeline

1. The selected source is validated and converted into readable text in memory.
2. Text is normalised, deduplicated, divided on natural boundaries, and capped safely.
3. Long blocks are tokenised into overlapping windows.
4. Windows are classified in batches and the maximum score for each category is retained per source block.
5. Flagged blocks receive a second contextual pass over bounded sentence and clause candidates.
6. Results preserve source order and display confidence rather than presenting the model as an authority.

## Running locally

Python 3.11 or newer is required. Create an isolated environment before installing dependencies.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --requirement requirements.txt
streamlit run toxiscan.py
```

The first analysis downloads the configured model from Hugging Face and caches it locally. To use another compatible multi-label model, set `TOXISCAN_MODEL_NAME` before starting Streamlit. The replacement must expose every moderation label expected by the classifier.

## Tests and quality checks

```powershell
python -m pip install --requirement requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m pip_audit --requirement requirements.txt
```

The test suite uses model, network, and browser doubles, so routine tests do not download model weights or scrape live sites. GitHub Actions runs the same tests, linting, and dependency audits with read-only repository permissions.

## Responsible use and limitations

Toxicity scores are probabilistic indicators, not factual or legal judgements. Sarcasm, reclaimed language, quotations, dialect, coded language, and cultural context can produce false positives or false negatives. Review surrounding context and apply human oversight before moderating users or taking consequential action.

Fine-tuning is intentionally not performed in this repository. Responsible fine-tuning requires a representative labelled dataset, an untouched evaluation set, category-level metrics, calibration analysis, and demographic bias testing. Changing weights without that evidence could make the application less accurate whilst appearing more confident.

## Author

Built by [Daveokw](https://github.com/Daveokw).
