"""k-shot experiment. k labelled examples per class are sampled from the TRAIN pool only;
models are scored on a fixed held-out split (dev by default, test for final numbers).

  python -m src.evaluate --no-embed              # offline baseline only
  python -m src.evaluate                         # dev split, all models
  python -m src.evaluate --split test            # FINAL numbers for the report (run once)
  python -m src.evaluate --test-file data/fresh_test.csv   # your own fresh test set
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from .data import load, ROOT
from .models import TfidfLR, EmbProto, EmbLR

def builders(no_embed=False):
    b = [lambda: TfidfLR()]
    if not no_embed:
        from .embed import embed
        b += [lambda: EmbProto(embed), lambda: EmbLR(embed), lambda: EmbLR(embed, True)]
    return b

def run(no_embed=False, split="dev", test_file=None, ks=(1, 2, 4, 8, 16), seeds=10):
    train, ev = load(split, test_file)
    tag = "external" if test_file else split
    rows = []
    for k in ks:
        for s in range(seeds):
            rng = np.random.RandomState(s)
            sup = np.concatenate([rng.choice(train.index[train.label == c], k, replace=False) for c in (0, 1)])
            for b in builders(no_embed):
                m = b().fit(train.loc[sup, "text"].tolist(), train.loc[sup, "label"].tolist())
                pred = (m.predict_proba(ev.text.tolist()) >= 0.5).astype(int)
                rec = {"k": k, "seed": s, "model": m.name,
                       "overall": f1_score(ev.label, pred, average="macro")}
                for lg in ("en", "hi", "hinglish"):
                    mk = (ev.lang == lg).values
                    rec[lg] = f1_score(ev.label[mk], pred[mk], average="macro") if mk.sum() else np.nan
                rows.append(rec)
    res = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results"); os.makedirs(out, exist_ok=True)
    res.to_csv(os.path.join(out, f"raw_results_{tag}.csv"), index=False)
    summ = res.groupby(["model", "k"])[["overall", "en", "hi", "hinglish"]].agg(["mean", "std"]).round(3)
    summ.to_csv(os.path.join(out, f"summary_{tag}.csv"))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5, 4.2))
    for name, g in res.groupby("model"):
        a = g.groupby("k")["overall"].agg(["mean", "std"])
        plt.errorbar(a.index, a["mean"], yerr=a["std"], marker="o", capsize=3, label=name)
    plt.xscale("log", base=2); plt.xlabel("k (labelled examples per class)")
    plt.ylabel("Macro-F1 (mean ± std over seeds)"); plt.grid(alpha=.3); plt.legend(fontsize=8)
    plt.title(f"Few-shot sarcasm detection ({tag} set, n={len(ev)})"); plt.tight_layout()
    plt.savefig(os.path.join(out, f"kshot_curve_{tag}.png"), dpi=200)
    print(f"Eval set: {tag}, n={len(ev)}  |  train pool: {len(train)}")
    print(summ["overall"].unstack("k")["mean"].to_string())
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--test-file", default=None)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    run(a.no_embed, a.split, a.test_file, seeds=a.seeds)
