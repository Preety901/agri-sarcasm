from src.data import ensure_splits

def test_split_covers_every_row_once():
    df = ensure_splits()
    assert df.id.is_unique
    assert set(df.split) == {"train", "dev", "test"}

def test_each_split_has_both_classes_and_all_languages():
    df = ensure_splits()
    for s in ("train", "dev", "test"):
        part = df[df.split == s]
        assert set(part.label) == {0, 1}
        assert set(part.lang) == {"en", "hi", "hinglish"}

def test_test_split_is_roughly_a_quarter():
    df = ensure_splits()
    assert 0.2 <= (df.split == "test").mean() <= 0.3
