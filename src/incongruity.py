"""Lightweight, language-agnostic incongruity features.

Idea: sarcasm in grievances = POSITIVE surface wording + NEGATIVE situation.
We count positive-sentiment cues and negative-situation cues (English, Hindi,
Hinglish) and flag when both co-occur. This is a transparent lexicon proxy for
'incongruity' and is meant to be ablated against the plain embedding model.
"""
import re
import numpy as np

POS_CUES = [
    "wah", "wow", "great", "wonderful", "brilliant", "amazing", "lovely", "perfect",
    "fantastic", "excellent", "thanks", "thankful", "so kind", "how kind", "love how",
    "nothing makes me happier", "kamaal", "badhiya", "mast hai", "meherbani", "kitni jaldi",
    "kya timing", "kya kismat", "favourite", "bahut acche", "itne acche", "kitna imaandar",
    "kitni badhiya", "asardaar", "accurate", "friendly", "so fast",
    "वाह", "बढ़िया", "कमाल", "शुक्रिया", "मेहरबानी", "कितनी जल्दी", "क्या किस्मत",
    "क्या समय", "पसंदीदा", "बहुत असरदार", "बड़ी सटीक", "कितना ईमानदार", "कितनी अच्छी",
    "इतने अच्छे", "क्या रफ़्तार", "काफी है", "ख्याल रखते",
]
NEG_CUES = [
    "fake", "half sand", "flood", "died", "dead", "dry", "expired", "hail", "crash",
    "closed", "after", "twenty times", "six hours", "2 am", "doubled", "not even one",
    "killed", "sookh", "baadh", "ole ", "band", "kat chuki", "marne ke baad", "mitti nikli",
    "ugaa nahi", "gir gaya", "6 ghante", "2 baje", "double", "lagat", "talaab", "expired",
    "बाढ़", "सूख", "नहीं निकला", "मरने", "ओले", "बंद", "एक्सपायर", "आधी तो मिट्टी",
    "कट चुकी", "घंटे", "दो बजे", "दोगुनी", "लागत", "तालाब", "के बाद", "गिर गया", "मारा",
]
FEATURE_NAMES = ["pos_cues", "neg_cues", "pos_and_neg", "exclaim"]

def _count(text, cues):
    t = text.lower()
    return sum(1 for c in cues if c in t)

def features(text: str) -> list:
    p, n = _count(text, POS_CUES), _count(text, NEG_CUES)
    return [min(p, 3), min(n, 3), float(p > 0 and n > 0), float("!" in text)]

def feature_matrix(texts) -> np.ndarray:
    return np.array([features(t) for t in texts], dtype=float)
