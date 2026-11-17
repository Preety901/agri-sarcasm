"""Which queries does the model get wrong?  Confusion matrix + list of errors.

  python -m src.error_analysis --no-embed          # TF-IDF baseline, offline
  python -m src.error_analysis --k 8               # embedding + incongruity (proposed)
  python -m src.error_analysis --split test
Predictions are averaged over 5 random few-shot support sets.
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from .data import load, ROOT
from .models import TfidfLR, EmbLR

def main(no_embed, k, split, test_file):
    train, ev = load(split, test_file)
    if no_embed:
        mk = lambda: TfidfLR()
    else:
        from .embed import embed
        mk = lambda: EmbLR(embed, True)
    probs = []
    for s in range(5):
        rng = np.random.RandomState(s)
        sup = np.concatenate([rng.choice(train.index[train.label == c], k, replace=False) for c in (0, 1)])
        m = mk().fit(train.loc[sup, "text"].tolist(), train.loc[sup, "label"].tolist())
        probs.append(m.predict_proba(ev.text.tolist()))
    ev = ev.copy(); ev["prob"] = np.mean(probs, 0).round(3); ev["pred"] = (ev.prob >= .5).astype(int)
    ev["error"] = np.where(ev.pred == ev.label, "", np.where(ev.pred == 1, "false_alarm (literal -> sarcastic)",
                                                              "missed_sarcasm (sarcastic -> literal)"))
    name = m.name
    print(f"Model: {name} | k={k} | eval n={len(ev)}\n")
    print(classification_report(ev.label, ev.pred, target_names=["literal", "sarcastic"], digits=3, zero_division=0))
    print("Errors per language:\n", ev[ev.error != ""].groupby("lang").size().to_string() or "none")
    out = os.path.join(ROOT, "results"); os.makedirs(out, exist_ok=True)
    errs = ev[ev.error != ""].sort_values("prob")
    errs.to_csv(os.path.join(out, "errors.csv"), index=False)
    print(f"\n{len(errs)} errors saved to results/errors.csv")
    print(errs[["lang", "label", "prob", "text"]].head(15).to_string(index=False))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(ev.label, ev.pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.8, 3.4)); ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1], ["literal", "sarcastic"]); ax.set_yticks([0, 1], ["literal", "sarcastic"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix (k={k})", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(out, "confusion_matrix.png"), dpi=200)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true"); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--split", choices=["dev", "test"], default="dev"); ap.add_argument("--test-file", default=None)
    a = ap.parse_args(); main(a.no_embed, a.k, a.split, a.test_file)
