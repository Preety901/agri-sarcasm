import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from src.data import load
from src.models import TfidfLR, EmbProto, EmbLR

_hv = HashingVectorizer(n_features=64, analyzer="char_wb", ngram_range=(2, 3), alternate_sign=False)
def dummy_embed(texts):            # offline stand-in for the real multilingual encoder
    return _hv.transform(list(texts)).toarray()

def _support(k=6):
    train, ev = load("dev")
    sup = np.concatenate([train.index[train.label == c][:k] for c in (0, 1)])
    return train.loc[sup], ev

def test_all_models_fit_and_output_probabilities():
    s, ev = _support()
    for m in (TfidfLR(), EmbProto(dummy_embed), EmbLR(dummy_embed), EmbLR(dummy_embed, True)):
        p = m.fit(s.text.tolist(), s.label.tolist()).predict_proba(ev.text.tolist())
        assert p.shape == (len(ev),) and np.all((p >= 0) & (p <= 1)), m.name

def test_tfidf_beats_chance_with_enough_shots():
    train, ev = load("dev")
    sup = np.concatenate([train.index[train.label == c][:20] for c in (0, 1)])
    m = TfidfLR().fit(train.loc[sup].text.tolist(), train.loc[sup].label.tolist())
    acc = ((m.predict_proba(ev.text.tolist()) >= .5) == ev.label.values).mean()
    assert acc > 0.5
