import os
import re
import time
import shutil
import io
import streamlit as st
import docx
import fitz
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# DEVICE SETUP
device = torch.device("cpu")

# LOAD MODEL
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map=None 
)
model.eval()

labels = [
    'toxicity', 'severe_toxicity', 'obscene',
    'identity_attack', 'insult', 'threat'
]
label_map = {
    'toxicity': 'Toxic',
    'severe_toxicity': 'Highly Toxic',
    'obscene': 'Obscene',
    'identity_attack': 'Hate',
    'insult': 'Insult',
    'threat': 'Threat'
}

# SCRAPE FROM URL
def get_webdriver():
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    if os.path.exists("/usr/bin/chromium"):
        opts.binary_location = "/usr/bin/chromium"

    sys_drv = shutil.which("chromedriver")
    if sys_drv:
        service = Service(executable_path=sys_drv)
    else:
        service = Service(ChromeDriverManager().install())

    return webdriver.Chrome(service=service, options=opts)

def scrape_text_from_url(url: str) -> list[str]:
    driver = get_webdriver()
    driver.get(url)

    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    blocks = soup.find_all(["article", "p", "li", "blockquote", "div"])
    texts = [b.get_text(strip=True) for b in blocks if len(b.get_text(strip=True)) > 5]
    return list(dict.fromkeys(texts))[:50]

# TEXT CLASSIFICATION
def batch_classify_texts(
    texts: list[str],
    threshold_text: float = 0.5,
    threshold_word: float = 0.3
) -> list[tuple[str, list[str], dict[str, list[str]]]]:
    results = []
    word_set = set()
    for txt in texts:
        word_set |= set(re.findall(r"\b\w{3,}\b", txt.lower()))
    word_list = list(word_set)

    if word_list:
        tok_w = tokenizer(word_list, return_tensors="pt", padding=True, truncation=True, max_length=16).to(device)
        with torch.no_grad():
            logits_w = model(**tok_w).logits
        scores_w = torch.sigmoid(logits_w).cpu().numpy()
        word_map = {
            word_list[i]: [labels[j] for j, s in enumerate(row) if s >= threshold_word]
            for i, row in enumerate(scores_w)
            if any(s >= threshold_word for s in row)
        }
    else:
        word_map = {}

    for txt in texts:
        tok_t = tokenizer(txt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            logits_t = model(**tok_t).logits
        scores_t = torch.sigmoid(logits_t)[0].cpu().numpy()
        cats = [labels[i] for i, s in enumerate(scores_t) if s >= threshold_text]
        if cats:
            toks = set(re.findall(r"\b\w{3,}\b", txt.lower()))
            flagged = {w: word_map[w] for w in toks if w in word_map}
            results.append((txt, cats, flagged))

    return results

# STREAMLIT APP
st.set_page_config(page_title="🛡️ ToxiScan", layout="wide")
st.title("🛡️ ToxiScan")
st.markdown("Enter a URL, paste text, or upload a (`.txt`, `.docx`, `.pdf`) file to detect toxicity.")

# Sensitivity sliders
threshold_text = st.slider("📊 Text Sensitivity Threshold", 0.0, 1.0, 0.5, 0.05)
threshold_word = st.slider("📉 Word Sensitivity Threshold", 0.0, 1.0, 0.3, 0.05)

mode = st.radio("Input type:", ("URL", "Text", "File"))
user_input: list[str] = []

if mode == "URL":
    url = st.text_input("Enter URL:")
    if url:
        st.info("Scraping text…")
        try:
            user_input = scrape_text_from_url(url)
            st.success(f"Scraped {len(user_input)} text blocks.")
        except Exception as e:
            st.error(f"Error scraping URL: {e}")

elif mode == "Text":
    txt = st.text_area("Paste text here:", height=200)
    if txt:
        user_input = [p for p in txt.split("\n\n") if p.strip()]
        st.success(f"Loaded {len(user_input)} text blocks from pasted text.")

elif mode == "File":
    f = st.file_uploader("Upload a `.txt`, `.pdf`, or `.docx` file:", type=["txt", "pdf", "docx"])
    if f:
        file_type = f.type
        try:
            if file_type == "text/plain":
                text = f.read().decode("utf-8")
                user_input = [p for p in text.split("\n\n") if p.strip()]
                st.success(f"Loaded {len(user_input)} blocks from .txt file.")

            elif file_type == "application/pdf":
                pdf = fitz.open(stream=f.read(), filetype="pdf")
                text = "\n".join(page.get_text() for page in pdf)
                user_input = [p for p in text.split("\n\n") if p.strip()]
                st.success(f"Loaded {len(user_input)} blocks from PDF.")

            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                file_bytes = f.read()
                doc = docx.Document(io.BytesIO(file_bytes))
                text = "\n".join(para.text for para in doc.paragraphs)
                st.text_area("📝 Extracted Text Preview", text, height=200)
                user_input = [p for p in text.split("\n\n") if len(p.strip()) > 10]
                st.success(f"Loaded {len(user_input)} text blocks from DOCX.")
        except Exception as e:
            st.error(f"Failed to process file: {e}")

# ANALYZE
if st.button("Analyze"):
    st.write(f"DEBUG: {len(user_input)} text blocks ready for analysis.")
    if not user_input:
        st.warning("No content provided.")
    else:
        try:
            results = batch_classify_texts(user_input, threshold_text, threshold_word)
            if not results:
                st.info("No toxic content detected.")
            else:
                st.success("Toxic content detected!")
                for i, (blk, cats, flagged) in enumerate(results, start=1):
                    title = ", ".join(label_map[c] for c in cats)
                    with st.expander(f"{i}. {title}"):
                        st.write(blk)
                        if flagged:
                            st.markdown("**Flagged Words:**")
                            for w, tags in flagged.items():
                                st.markdown(f"- `{w}`: {', '.join(label_map[t] for t in tags)}")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
