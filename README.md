# ToxiScan — Toxic Language Detection App (BERT & Streamlit)

A Natural Language Processing application that detects hateful or toxic language in any URL, pasted text, or uploaded document. Built with **PyTorch**, **Hugging Face Transformers**, and **Streamlit**.

## Live Demo
Try the live app here: [ToxiScan on Streamlit](https://toxiscan.streamlit.app/)

## About the Project

ToxiScan uses a fine-tuned BERT model (`unitary/toxic-bert`) to classify text across six toxicity categories:

- **Toxic** — General toxic language
- **Highly Toxic** — Severe toxicity
- **Obscene** — Profane or vulgar content
- **Hate** — Identity-based attacks
- **Insult** — Insulting language
- **Threat** — Threatening language

The app goes beyond simple text classification. It also performs word-level analysis, flagging the specific words within a passage that triggered the detection. Users can adjust sensitivity thresholds for both passage-level and word-level classification.

## Features

- **URL Scraping** — Paste any URL and ToxiScan will render the page using Selenium, extract the visible text, and analyse it for toxicity.
- **Text Input** — Paste raw text directly for instant analysis.
- **File Upload** — Upload `.txt`, `.pdf`, or `.docx` files for batch analysis.
- **Word-Level Flagging** — Identifies and highlights the exact words that triggered each toxicity category.
- **Adjustable Sensitivity** — Tune both text and word sensitivity thresholds via interactive sliders.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers (BERT)
- Streamlit
- Selenium / BeautifulSoup (web scraping)
- PyMuPDF / python-docx (document parsing)

## How It Works

1. Text is extracted from the chosen input source (URL, raw text, or file).
2. Each text block is tokenised and passed through the fine-tuned BERT model.
3. Sigmoid activation is applied to the logits to produce per-category confidence scores.
4. Blocks exceeding the text sensitivity threshold are flagged.
5. Individual words are also classified and mapped back to their source passages.

## Status

This is a working prototype. The model performs well on clearly toxic content but may occasionally produce false positives on ambiguous or sarcastic language. Future improvements could include contextual analysis and additional training data to reduce false positives.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run toxiscan.py
```

## Author

Built by [Daveokw](https://github.com/Daveokw)
