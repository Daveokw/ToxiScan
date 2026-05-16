Just shipped something I'm proud of: built and deployed an NLP toxicity detector from scratch.

No third party moderation APIs. No basic keyword filters. Raw Natural Language Processing.
Here's what the setup looks like:

🧠 Fine-tuned BERT model (toxic-bert) classifying text across 6 toxicity categories
🔍 Word-level analysis that pinpoints the exact words triggering each detection
🌐 Full URL scraping via Selenium, rendering JavaScript-heavy pages before analysis
📄 Supports .txt, .pdf, and .docx file uploads for batch scanning
🎚️ Adjustable sensitivity thresholds for both passage and word-level classification
🖥️ Deployed as an interactive Streamlit web app

That's it.

The difference between this and just calling a moderation API? I own the model pipeline. I built the scraping layer, the tokenisation logic, the multi-label classification, and the word-level mapping. Every component is mine.

It's a working prototype, so there's room to improve, particularly around sarcasm and context-dependent language. But it's a solid proof of concept for automated content moderation using deep learning.

If you want to try it out or look at the code:

Live Demo: https://toxiscan.streamlit.app/
GitHub: https://github.com/Daveokw/ToxiScan

#NaturalLanguageProcessing #NLP #PyTorch #BERT #MachineLearning #Streamlit #Python #DataScience #DeepLearning

---

LinkedIn Project Section:

Project Name: ToxiScan — Toxic Language Detection App (BERT & Streamlit)

Description:
- Developed an NLP-powered web application that detects toxic, hateful, and abusive language across six categories using a fine-tuned BERT model.
- Built a full text analysis pipeline including URL scraping via Selenium, document parsing (PDF, DOCX, TXT), and word-level toxicity mapping.
- Deployed the application as an interactive Streamlit web app with adjustable sensitivity controls.

Skills: Natural Language Processing, Deep Learning, PyTorch, Python, Streamlit, BERT

Image 1 Title: ToxiScan Interface
Image 1 Description: The ToxiScan web application interface showing input options for URL, text, and file upload with adjustable sensitivity sliders.

Image 2 Title: Toxicity Analysis Results
Image 2 Description: ToxiScan displaying flagged toxic content with word-level analysis, categorising detected passages as Toxic, Obscene, Insult, or Hate.
