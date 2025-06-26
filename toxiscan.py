import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import re, time, shutil

# ─── Load tokenizer & model ─────────────────────────────────────────────────────
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to("cpu")
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

# ─── Scraper ────────────────────────────────────────────────────────────────────
def scrape_text_from_url(url: str) -> list[str]:
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--disable-gpu")
    opts.binary_location = "/usr/bin/chromium"               # APT-installed Chromium

    # find the APT-installed chromedriver on $PATH
    driver_path = shutil.which("chromedriver")
    service = Service(executable_path=driver_path)

    driver = webdriver.Chrome(service=service, options=opts)
    driver.get(url)

    # scroll to lazy-load
    time.sleep(1)
    last_h = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        new_h = driver.execute_script("return document.body.scrollHeight")
        if new_h == last_h:
            break
        last_h = new_h

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    # remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()

    blocks = soup.find_all(["article", "p", "li", "blockquote", "div"])
    texts = [
        b.get_text(strip=True)
        for b in blocks
        if len(b.get_text(strip=True)) > 30
    ]
    return list(dict.fromkeys(texts))[:50]

# ─── Classifier ────────────────────────────────────────────────────────────────
def batch_classify_texts(
    texts: list[str],
    threshold_text: float = 0.5,
    threshold_word: float = 0.3
) -> list[tuple[str, list[str], dict[str, list[str]]]]:
    results, word_set = [], set()
    for txt in texts:
        word_set |= set(re.findall(r"\b\w{3,}\b", txt.lower()))
    word_list = list(word_set)

    # classify words
    tok_word_inputs = tokenizer(
        word_list,
        return_tensors="pt",
        padding=True, truncation=True, max_length=16
    )
    with torch.no_grad():
        tok_outputs = model(**tok_word_inputs)
    word_scores = torch.sigmoid(tok_outputs.logits).cpu().numpy()
    word_map = {
        word_list[i]: [
            labels[j] for j, s in enumerate(row) if s >= threshold_word
        ]
        for i, row in enumerate(word_scores)
        if any(s >= threshold_word for s in row)
    }

    # classify full texts
    for txt in texts:
        inputs = tokenizer(
            txt, return_tensors="pt",
            truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits)[0].cpu().numpy()
        cats = [labels[i] for i, s in enumerate(scores) if s >= threshold_text]
        if cats:
            toks = set(re.findall(r"\b\w{3,}\b", txt.lower()))
            flagged = {w: word_map[w] for w in toks if w in word_map}
            results.append((txt, cats, flagged))
    return results

# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🛡️ ToxiScan", layout="wide")
st.title("🛡️ ToxiScan")
st.markdown("Enter a URL, paste text, or upload a `.txt` to detect toxicity.")

opt = st.radio("Input type:", ("URL", "Text", "File"))
user_input: list[str] = []

if opt == "URL":
    url = st.text_input("Enter URL:")
    if url:
        st.info("Scraping text…")
        try:
            user_input = scrape_text_from_url(url)
        except Exception as e:
            st.error(f"Error scraping URL: {e}")

elif opt == "Text":
    txt = st.text_area("Paste text:", height=200)
    user_input = [txt] if txt else []

else:  # File
    f = st.file_uploader("Upload `.txt`:", type=["txt"])
    if f:
        content = f.read().decode("utf-8")
        user_input = [line for line in content.splitlines() if len(line) > 30]

if st.button("Analyze"):
    if not user_input:
        st.warning("No content provided.")
    else:
        out = batch_classify_texts(user_input)
        if not out:
            st.info("No toxic content detected.")
        else:
            for i, (blk, cats, flagged) in enumerate(out, 1):
                with st.expander(f"{i}. {', '.join(label_map[c] for c in cats)}"):
                    st.write(blk)
                    if flagged:
                        st.markdown("**Flagged Words:**")
                        for w, tags in flagged.items():
                            st.markdown(f"- `{w}`: {', '.join(label_map[t] for t in tags)}")
