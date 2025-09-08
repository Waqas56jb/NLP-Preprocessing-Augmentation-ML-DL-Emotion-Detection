# Emotion NLP Prototype (Tkinter)

Requirements
- Python 3.10+
- See `requirements.txt`

Setup
1) Create a venv and install dependencies:
```
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

2) Ensure `test.csv` exists at project root with columns `text` and `Emotion`.

Run
```
python app.py
```

Features
- Load CSV and auto-suggest preprocessing based on dataset stats
- Select preprocessing: lowercase, remove punctuation, numbers, extra whitespace, stopwords
- Preview processed samples
- Save processed CSV
- Train models (LogReg, Naive Bayes, Linear SVM) with TF-IDF; show accuracy and report
- Augmentation preview and export: synonym replace, insert, swap, delete, truncate, noise, sentence shuffle, back-translation, paraphrase

Notes
- Some augmentation methods are implemented in `src/augment.py` (synonym replacement, insertion, swap, deletion, truncation, noise). You may wire them in the GUI if required.
- Back-translation and paraphrasing use `transformers` and will download models on first run. If these are heavy for your environment, prefer light augmentations.
