import pandas as pd
from src.lang import detect

def test_basic_languages():
    assert detect("Which fertilizer is best for wheat?") == "en"
    assert detect("गेहूं के लिए सबसे अच्छी खाद कौन सी है?") == "hi"
    assert detect("Gehun ke liye sabse achi khaad kaun si hai?") == "hn"

def test_accuracy_on_seed_data():
    from src.data import SEED_CSV
    df = pd.read_csv(SEED_CSV)
    gold = df.lang.map({"en": "en", "hi": "hi", "hinglish": "hn"})
    assert (df.text.map(detect) == gold).mean() >= 0.9
