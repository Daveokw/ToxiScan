import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import re
import time

model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    low_cpu_mem_usage=False,
    device_map=None
)
model.to("cpu")
model.eval()

labels = ['toxicity','severe_toxicity','obscene','identity_attack','insult','threat']
label_map = {
    'toxicity': 'Toxic',
    'severe_toxicity': 'Highly Toxic',
    'obscene': 'Obscene',
    'identity_attack': 'Hate',
    'insult': 'Insult',
    'threat': 'Threat'
}

def scrape_text_from_url(url: str, wait_time: float = 5.0, scroll_pause: float = 1.0):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60_000)
        time.sleep(wait_time)

    
        prev_height = page.evaluate("() => document.body.scrollHeight")
        while True:
            page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(scroll_pause)
            new_height = page.evaluate("() => document.body.scrollHeight")
            if new_height == prev_height:
                break
            prev_height = new_height

        html = page.content()
        page.close()
        browser.close()

    # parse and clean
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
        tag.decompose()

    blocks = soup.find_all(['article','p','li','blockquote','div'])
    texts = [b.get_text(strip=True) for b in blocks if len(b.get_text(strip=True)) > 30]
    # dedupe and limit
    return list(dict.fromkeys(texts))[:50]

def batch_classify_texts(texts, threshold_text=0.5, threshold_word=0.3):
    results = []
    # extract unique words ≥3 letters
    word_set = set()
    for txt in texts:
        word_set |= set(re.findall(r'\b\w{3,}\b', txt.lower()))
    word_list = list(word_set)

    # classify individual words for fine‐grained highlights
    tok_word_inputs = tokenizer(
        word_list, return_tensors='pt',
        padding=True, truncation=True, max_length=16
    )
    with torch.no_grad():
        tok_outputs = model(**tok_word_inputs)
    word_scores = torch.sigmoid(tok_outputs.logits).cpu().numpy()

    word_map = {
        word_list[i]: [labels[j] for j,s in enumerate(row) if s >= threshold_word]
        for i,row in enumerate(word_scores)
        if any(s >= threshold_word for s in row)
    }

    # classify full text blocks
    for txt in texts:
        inputs = tokenizer(
            txt, return_tensors='pt',
            truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits)[0].cpu().numpy()
        categories = [labels[i] for i,s in enumerate(scores) if s >= threshold_text]
        if categories:
            toks = set(re.findall(r'\b\w{3,}\b', txt.lower()))
            flagged_words = {w: word_map[w] for w in toks if w in word_map}
            results.append((txt, categories, flagged_words))
    return results

# ——— Streamlit UI ———
st.set_page_config(page_title="🛡️ ToxiScan", layout="wide")
st.title("🛡️ ToxiScan")
st.markdown("Enter a URL, paste some text, or upload a `.txt` file to detect offensive content.")

input_option = st.radio("Select input type:", ("URL","Text","File"))
user_input = []

if input_option == "URL":
    url = st.text_input("Enter URL:")
    if url:
        st.info("Scraping text from URL…")
        try:
            user_input = scrape_text_from_url(url)
        except Exception as e:
            st.error(f"Error scraping URL: {e}")

elif input_option == "Text":
    txt = st.text_area("Enter text:", height=200)
    if txt:
        user_input = [txt]

else:
    uploaded_file = st.file_uploader("Upload .txt file:", type=["txt"])
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        user_input = [line for line in content.splitlines() if len(line.strip()) > 30]

if st.button("Analyze"):
    if not user_input:
        st.warning("No content provided for analysis.")
    else:
        results = batch_classify_texts(user_input)
        if not results:
            st.info("No toxic content detected.")
        else:
            for idx,(text,cats,flagged_words) in enumerate(results,1):
                header = ", ".join(label_map[c] for c in cats)
                with st.expander(f"{idx}. {header}"):
                    st.write(text)
                    if flagged_words:
                        st.markdown("**Flagged Words:**")
                        for w,tags in flagged_words.items():
                            st.markdown(f"- `{w}`: {', '.join(label_map[t] for t in tags)}")
