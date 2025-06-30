import os
import re
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import fitz 
import docx 

model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

labels = ['toxicity', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
label_map = {
    'toxicity': 'Toxic',
    'severe_toxicity': 'Highly Toxic',
    'obscene': 'Obscene',
    'identity_attack': 'Hate',
    'insult': 'Insult',
    'threat': 'Threat'
}

def scrape_text_from_url(url):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    time.sleep(5)

    while True:
        last_h = driver.execute_script("return document.body.scrollHeight")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_h = driver.execute_script("return document.body.scrollHeight")
        if new_h == last_h:
            break

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
        tag.decompose()

    blocks = soup.find_all(['article', 'p', 'li', 'blockquote', 'div'])
    texts = [b.get_text(strip=True) for b in blocks if len(b.get_text(strip=True)) > 5]
    return list(dict.fromkeys(texts))[:50]

def read_file_text(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    try:
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        elif ext == ".pdf":
            pdf = fitz.open(filepath)
            text = "\n".join([page.get_text() for page in pdf])
            return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        elif ext == ".docx":
            doc = docx.Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
            return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    except Exception as e:
        messagebox.showerror("File Error", f"Failed to read {filepath}:\n{e}")
    return []

def batch_classify_texts(texts, threshold_text=0.5, threshold_word=0.3):
    results = []
    word_set = set()
    for txt in texts:
        word_set |= set(re.findall(r'\b\w{3,}\b', txt.lower()))

    word_list = list(word_set)
    tok_word_inputs = tokenizer(word_list, return_tensors='pt', padding=True, truncation=True, max_length=16)
    with torch.no_grad():
        tok_outputs = model(**tok_word_inputs)
    word_scores = torch.sigmoid(tok_outputs.logits).cpu().numpy()
    word_map = {
        word_list[i]: [labels[j] for j, s in enumerate(row) if s >= threshold_word]
        for i, row in enumerate(word_scores)
        if any(s >= threshold_word for s in row)
    }

    for txt in texts:
        inputs = tokenizer(txt, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits)[0].cpu().numpy()
        categories = [labels[i] for i, s in enumerate(scores) if s >= threshold_text]
        if categories:
            toks = set(re.findall(r'\b\w{3,}\b', txt.lower()))
            flagged_words = {w: word_map[w] for w in toks if w in word_map}
            results.append((txt, categories, flagged_words))
    return results

def analyze_input_text(inp):
    if re.match(r'https?://', inp):
        return scrape_text_from_url(inp)
    else:
        return [inp] if inp.strip() else []

def analyze_file():
    path = filedialog.askopenfilename(filetypes=[
        ("Text files", "*.txt"),
        ("PDF files", "*.pdf"),
        ("Word Documents", "*.docx")
    ])
    if not path:
        return
    results_display.delete('1.0', tk.END)
    chunks = read_file_text(path)
    if not chunks:
        results_display.insert(tk.END, "No readable text found in file.\n")
        return
    results = batch_classify_texts(chunks)
    show_results(results)

def analyze_text_or_url():
    inp = input_entry.get('1.0', tk.END).strip()
    results_display.delete('1.0', tk.END)
    if not inp:
        messagebox.showwarning('Input Error', 'Please enter a URL or some text.')
        return

    chunks = analyze_input_text(inp)
    if not chunks:
        results_display.insert(tk.END, "No readable content found.\n")
        return

    results = batch_classify_texts(chunks)
    show_results(results)

def show_results(results):
    if not results:
        results_display.insert(tk.END, "No toxic content detected.\n")
        return

    for idx, (txt, cats, flagged) in enumerate(results, 1):
        label_names = ', '.join(label_map[c] for c in cats)
        results_display.insert(tk.END, f"{idx}. {label_names}\n{txt}\n\n")
        if flagged:
            results_display.insert(tk.END, "Flagged Words:\n")
            for w, tags in flagged.items():
                readable_tags = ', '.join(label_map[t] for t in tags)
                results_display.insert(tk.END, f" - {w}: {readable_tags}\n")
            results_display.insert(tk.END, "\n")

def build_gui():
    global root, input_entry, results_display
    root = tk.Tk()
    root.title("\U0001F6E1\ufe0f Toxiscan")
    root.geometry("900x750")
    root.config(bg="#f0f0f0")

    tk.Label(root, text="Enter a URL or paste text:", font=("Arial", 12), bg="#f0f0f0").pack(pady=5)
    input_entry = scrolledtext.ScrolledText(root, width=100, height=6, font=("Arial", 10))
    input_entry.pack(pady=5)

    tk.Button(root, text="Analyze Text/URL", command=analyze_text_or_url,
              font=("Arial", 12, "bold"), bg="#0078D7", fg="white", width=18).pack(pady=10)

    tk.Button(root, text="Analyze File (.txt, .pdf, .docx)", command=analyze_file,
              font=("Arial", 11), bg="#28a745", fg="white", width=26).pack(pady=5)

    tk.Label(root, text="Results:", font=("Arial", 12), bg="#f0f0f0").pack(pady=5)
    results_display = scrolledtext.ScrolledText(root, width=100, height=25, font=("Arial", 10))
    results_display.pack(pady=5)

    root.mainloop()

if __name__ == '__main__':
    build_gui()
