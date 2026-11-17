"""Few-shot classifiers. All are trained from only k labelled examples per class."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .incongruity import feature_matrix

class TfidfLR:
    name = "TF-IDF char + LR (baseline)"
    def fit(self, texts, y):
        self.v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.c = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.c.fit(self.v.fit_transform(texts), y); return self
    def predict_proba(self, texts):
        return self.c.predict_proba(self.v.transform(texts))[:, 1]

class EmbProto:
    name = "Embedding prototype (cosine)"
    def __init__(self, embed_fn): self.e = embed_fn
    def fit(self, texts, y):
        X, y = self.e(texts), np.array(y)
        self.p = [X[y == c].mean(0) for c in (0, 1)]
        self.p = [v / np.linalg.norm(v) for v in self.p]; return self
    def predict_proba(self, texts):
        X = self.e(texts)
        s = np.stack([X @ self.p[0], X @ self.p[1]], 1) * 10
        s = np.exp(s - s.max(1, keepdims=True)); return (s / s.sum(1, keepdims=True))[:, 1]

class EmbLR:
    def __init__(self, embed_fn, use_incon=False):
        self.e, self.use_incon = embed_fn, use_incon
        self.name = "Embedding + incongruity + LR (proposed)" if use_incon else "Embedding + LR"
    def _x(self, texts):
        X = self.e(texts)
        if self.use_incon:
            F = feature_matrix(texts)
            X = np.hstack([X, self.sc.transform(F) if hasattr(self, "sc") else F])
        return X
    def fit(self, texts, y):
        if self.use_incon:
            self.sc = StandardScaler().fit(feature_matrix(texts))
        self.c = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.c.fit(self._x(texts), y); return self
    def predict_proba(self, texts):
        return self.c.predict_proba(self._x(texts))[:, 1]
