from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None  # type: ignore

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore


ModelName = Literal[
    "logreg", "nb", "svm", "rf", "knn", "xgb", "lgbm", "lstm", "gru", "cnn", "distilbert"
]


def build_pipeline(model: ModelName = "logreg") -> Pipeline:
    if model == "logreg":
        clf = LogisticRegression(max_iter=200)
    elif model == "nb":
        clf = MultinomialNB()
    elif model == "svm":
        clf = LinearSVC()
    elif model == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    elif model == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif model == "xgb":
        if XGBClassifier is None:
            raise ValueError("XGBoost not installed")
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, objective="multi:softmax")
    elif model == "lgbm":
        if LGBMClassifier is None:
            raise ValueError("LightGBM not installed")
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.1)
    else:
        raise ValueError("Unsupported classical/ensemble model")

    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", clf),
    ])


@dataclass
class TrainResult:
    pipeline: object
    accuracy: float
    report: str


def _split(df: pd.DataFrame, text_col: str, label_col: str, test_size: float = 0.8) -> Tuple[List[str], List[str], List[str], List[str]]:
    df = df.dropna(subset=[text_col, label_col])
    X = df[text_col].astype(str).tolist()
    y = df[label_col].astype(str).tolist()
    unique_labels = sorted(set(y))
    if len(unique_labels) != 13:
        raise ValueError(f"Expected 13 labels, found {len(unique_labels)}: {unique_labels}")
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def _train_deeplearning(model_name: str, X_train: List[str], y_train: List[str], X_test: List[str], y_test: List[str]) -> TrainResult:
    if torch is None:
        raise ValueError("PyTorch not installed")

    # Build vocabulary
    def tokenize(s: str) -> List[str]:
        return s.split()

    vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
    for s in X_train:
        for w in tokenize(s):
            if w not in vocab:
                vocab[w] = len(vocab)
    max_len = 100

    def encode(s: str) -> List[int]:
        ids = [vocab.get(w, 1) for w in tokenize(s)][:max_len]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

    classes = sorted(set(y_train + y_test))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}

    Xtr = torch.tensor([encode(s) for s in X_train], dtype=torch.long)
    ytr = torch.tensor([label2id[c] for c in y_train], dtype=torch.long)
    Xte = torch.tensor([encode(s) for s in X_test], dtype=torch.long)
    yte = torch.tensor([label2id[c] for c in y_test], dtype=torch.long)

    class TextDataset(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y
        def __len__(self):
            return self.x.size(0)
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    train_loader = DataLoader(TextDataset(Xtr, ytr), batch_size=32, shuffle=True)

    vocab_size = len(vocab)
    embed_dim = 64
    num_classes = len(classes)

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, 64, batch_first=True)
            self.fc = nn.Linear(64, num_classes)
        def forward(self, x):
            e = self.emb(x)
            out, _ = self.lstm(e)
            h = out[:, -1, :]
            return self.fc(h)

    class GRUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.gru = nn.GRU(embed_dim, 64, batch_first=True)
            self.fc = nn.Linear(64, num_classes)
        def forward(self, x):
            e = self.emb(x)
            out, _ = self.gru(e)
            h = out[:, -1, :]
            return self.fc(h)

    class CNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.conv = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(64, num_classes)
        def forward(self, x):
            e = self.emb(x).transpose(1, 2)  # (B, E, T)
            c = torch.relu(self.conv(e))
            p = self.pool(c).squeeze(-1)
            return self.fc(p)

    if model_name == "lstm":
        net = LSTMModel()
    elif model_name == "gru":
        net = GRUModel()
    elif model_name == "cnn":
        net = CNNModel()
    else:
        raise ValueError("Unsupported DL model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    net.train()
    for _ in range(2):  # few epochs for prototype
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()

    net.eval()
    with torch.no_grad():
        logits = net(Xte.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(yte.numpy(), preds)
    rep = classification_report(yte.numpy(), preds, target_names=classes)
    return TrainResult(pipeline=net, accuracy=acc, report=rep)


def _train_distilbert_feature_extractor(X_train: List[str], y_train: List[str], X_test: List[str], y_test: List[str]) -> TrainResult:
    if AutoTokenizer is None or AutoModel is None or torch is None:
        raise ValueError("Transformers/PyTorch not installed")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def embed(texts: List[str]) -> np.ndarray:
        feats: List[np.ndarray] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc).last_hidden_state[:, 0, :]  # CLS token
            feats.append(out.cpu().numpy())
        return np.vstack(feats)

    Xtr = embed(X_train)
    Xte = embed(X_test)
    classes = sorted(set(y_train + y_test))
    clf = LogisticRegression(max_iter=500)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    acc = accuracy_score(y_test, preds)
    rep = classification_report(y_test, preds)
    return TrainResult(pipeline=(model, clf, tokenizer), accuracy=acc, report=rep)


def train_and_eval(df: pd.DataFrame, text_col: str = "text", label_col: str = "Emotion", model: ModelName = "logreg") -> TrainResult:
    # Per user requirement: 80% test, 20% train, stratified across 13 labels
    X_train, X_test, y_train, y_test = _split(df, text_col, label_col, test_size=0.8)

    if model in {"logreg", "nb", "svm", "rf", "xgb", "lgbm"}:
        pipeline = build_pipeline(model)

        # XGBoost/LightGBM require numeric class labels
        if model in {"xgb", "lgbm"}:
            enc = LabelEncoder()
            y_train_enc = enc.fit_transform(y_train)
            y_test_enc = enc.transform(y_test)
            pipeline.fit(X_train, y_train_enc)
            preds_enc = pipeline.predict(X_test)
            preds = enc.inverse_transform(preds_enc)
        else:
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        rep = classification_report(y_test, preds)
        return TrainResult(pipeline=pipeline, accuracy=acc, report=rep)
    elif model in {"lstm", "gru", "cnn"}:
        return _train_deeplearning(model, X_train, y_train, X_test, y_test)
    elif model == "distilbert":
        return _train_distilbert_feature_extractor(X_train, y_train, X_test, y_test)
    else:
        raise ValueError("Unsupported model")

