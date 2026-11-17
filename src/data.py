"""Data loading with a FIXED, stratified train / dev / test split.

* train : the pool few-shot examples are sampled from
* dev   : use while developing (tuning lexicon, prompts, k ...)
* test  : look at it ONCE, for the final numbers in the report
Split is stored in data/split.csv, so it never changes between runs. New rows added to the
seed CSV get a split the next time ensure_splits() runs; existing assignments stay untouched.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEED_CSV = os.path.join(ROOT, "data", "agri_sarcasm_seed.csv")
SPLIT_CSV = os.path.join(ROOT, "data", "split.csv")
FRAC_TEST, FRAC_DEV = 0.25, 0.15

def assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Stratified by (lang, label). Returns a Series of 'train'/'dev'/'test' indexed like df."""
    out = pd.Series("train", index=df.index, dtype=object)
    for i, (_, g) in enumerate(df.groupby(["lang", "label"])):
        idx = np.random.RandomState(seed + i).permutation(g.index.values)
        n_test, n_dev = int(round(FRAC_TEST * len(idx))), int(round(FRAC_DEV * len(idx)))
        out.loc[idx[:n_test]] = "test"
        out.loc[idx[n_test:n_test + n_dev]] = "dev"
    return out

def ensure_splits() -> pd.DataFrame:
    df = pd.read_csv(SEED_CSV)
    sp = pd.read_csv(SPLIT_CSV) if os.path.exists(SPLIT_CSV) else pd.DataFrame(columns=["id", "split"])
    new = df[~df.id.isin(sp.id)]
    if len(new):
        add = pd.DataFrame({"id": new.id.values, "split": assign_splits(new).values})
        sp = pd.concat([sp, add], ignore_index=True)
        sp.to_csv(SPLIT_CSV, index=False)
    return df.merge(sp, on="id")

def load(eval_split: str = "dev", test_file: str = None):
    """Returns (train_df, eval_df). test_file = optional external CSV (id,text,lang,label)."""
    full = ensure_splits()
    train = full[full.split == "train"].reset_index(drop=True)
    if test_file:
        ev = pd.read_csv(test_file)
    else:
        ev = full[full.split == eval_split].reset_index(drop=True)
    return train, ev

if __name__ == "__main__":
    full = ensure_splits()
    print(full.groupby(["split", "lang", "label"]).size().unstack("split").fillna(0).astype(int))
