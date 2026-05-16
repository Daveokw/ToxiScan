I've recently been exploring automated content moderation and just deployed a new prototype: ToxiScan, an NLP toxicity detector.

Instead of relying on third-party APIs, I wanted to build the pipeline myself using Natural Language Processing to better understand the mechanics under the hood. Here's a look at the current setup:

🧠 Fine-tuned BERT model (toxic-bert) classifying text across 6 toxicity categories
🔍 Word-level analysis that highlights the specific words triggering each detection
🌐 URL scraping via Selenium to process live web pages
📄 Support for .txt, .pdf, and .docx file uploads for batch scanning
🎚️ Adjustable sensitivity thresholds for both passage and word-level classification
🖥️ Deployed as an interactive Streamlit web app

This is very much a working prototype, and there's definitely room for improvement—especially when it comes to handling sarcasm and complex context. It's been a great learning experience piecing together the scraping, tokenisation, and multi-label classification.

I'm really excited to share this and would absolutely welcome any feedback, suggestions, or ideas for improvement from the community!

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
