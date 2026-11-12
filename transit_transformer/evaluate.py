"""Test-set evaluation of fine-tuned runs, with bootstrap confidence intervals.

    python3 evaluate.py runs/ft_main runs/ft_random runs/ft_none runs/ft_frozen

Per run (writes <run>/eval.json and <run>/confusion_<set>.png), for each set:
    tess_test, tess_test_future, kepler_test
  - macro-F1 and balanced accuracy over 4 classes (95% bootstrap CI, 1000 resamples by signal)
  - one-vs-rest PR-AUC per class
  - FP vs (planet + candidate): FP recall at >= 90% precision
  - "no planet" PR-AUC two ways: real labels only (FA / NTP) and incl. random-ephemeris negatives
Prints a comparison table across runs and writes results/finetune_comparison.md.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_curve)
from torch.utils.data import DataLoader

from build_examples import NS
from finetune import CLASS_NAMES, SignalSet, collate, predict
from finetune_model import SignalClassifier
from model import ModelConfig
from pretrain import pick_device

SETS = {"tess_test": ("test", ("tess",)), "tess_test_future": ("test_future", ("tess",)),
        "kepler_test": ("test", ("kepler",))}


def macro_f1(y, p):
    labels = [c for c in range(4) if (y == c).any()]
    return f1_score(y, p.argmax(1), labels=labels, average="macro", zero_division=0)


def bootstrap(fn, y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        try:
            vals.append(fn(y[i], p[i]))
        except ValueError:
            pass
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def recall_at_precision(y_bin, score, target=0.9):
    if y_bin.sum() == 0:
        return float("nan")
    pr, rc, _ = precision_recall_curve(y_bin, score)
    ok = pr >= target
    return float(rc[ok].max()) if ok.any() else 0.0


def score_set(y, p, raw):
    out = dict(n=int(len(y)), counts={CLASS_NAMES[c]: int((y == c).sum()) for c in range(4)})
    out["macro_f1"] = float(macro_f1(y, p)); out["macro_f1_ci"] = bootstrap(macro_f1, y, p)
    ba = lambda a, b: balanced_accuracy_score(a, b.argmax(1))
    out["bal_acc"] = float(ba(y, p)); out["bal_acc_ci"] = bootstrap(ba, y, p)
    out["pr_auc"] = {CLASS_NAMES[c]: (float(average_precision_score(y == c, p[:, c])) if (y == c).any() else None)
                     for c in range(4)}
    sel = y <= 2
    if (y[sel] == 2).any():
        s = p[sel, 2] / p[sel, :3].sum(1)
        out["fp_recall_at_90_precision"] = recall_at_precision(y[sel] == 2, s)
        ap = lambda a, b: average_precision_score(a == 2, b[:, 2] / b[:, :3].sum(1))
        out["fp_vs_planet_pr_auc"] = float(ap(y[sel], p[sel])); out["fp_vs_planet_pr_auc_ci"] = bootstrap(ap, y[sel], p[sel])
    real = raw != "RAND"
    if ((y == 3) & real).any():
        out["no_planet_pr_auc_real_only"] = float(average_precision_score(y[real] == 3, p[real, 3]))
    out["confusion"] = confusion_matrix(y, p.argmax(1), labels=range(4)).tolist()
    return out


def plot_conf(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm); norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(4), [c.replace("_", "\n") for c in CLASS_NAMES], fontsize=8)
    ax.set_yticks(range(4), [c.replace("_", "\n") for c in CLASS_NAMES], fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def evaluate_run(run, device, workers):
    ck = torch.load(os.path.join(run, "best.pt"), map_location=device, weights_only=False)
    a = ck["args"]
    pre = torch.load(a["pretrained"], map_location="cpu", weights_only=False)
    model = SignalClassifier(ModelConfig(**pre["model_cfg"]), NS, a["encoder"], a["dropout"]).to(device)
    model.load_state_dict(ck["model"])
    res = dict(run=run, encoder=a["encoder"], best_epoch=ck["epoch"], tess_val=ck["tess_val"])
    for name, (split, missions) in SETS.items():
        ds = SignalSet(split, missions=missions)
        dl = DataLoader(ds, batch_size=64, num_workers=workers, collate_fn=collate)
        p, y, _, idx = predict(model, dl, device, a["amp"])
        raw = ds.meta.raw_label.to_numpy()[idx]
        res[name] = score_set(y, p, raw)
        plot_conf(res[name]["confusion"], f"{os.path.basename(run)} - {name}",
                  os.path.join(run, f"confusion_{name}.png"))
        np.savez(os.path.join(run, f"pred_{name}.npz"), prob=p, y=y, idx=idx)
    with open(os.path.join(run, "eval.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


def table(results):
    def ci(r, k):
        return f"{r[k]:.3f} [{r[k + '_ci'][0]:.3f}, {r[k + '_ci'][1]:.3f}]"
    lines = ["| run | encoder | TESS test macro-F1 | TESS test bal-acc | FP-vs-planet PR-AUC | FP recall @90% prec "
             "| TESS future macro-F1 | Kepler test macro-F1 |", "|" + "---|" * 8]
    for r in results:
        t, f, k = r["tess_test"], r["tess_test_future"], r["kepler_test"]
        lines.append(f"| {os.path.basename(r['run'])} | {r['encoder']} | {ci(t, 'macro_f1')} | {ci(t, 'bal_acc')} | "
                     f"{ci(t, 'fp_vs_planet_pr_auc')} | {t['fp_recall_at_90_precision']:.3f} | "
                     f"{f['macro_f1']:.3f} (n={f['n']}) | {k['macro_f1']:.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    device = pick_device(args.device)
    results = [evaluate_run(r, device, args.workers) for r in args.runs]
    md = table(results)
    print(md)
    for r in results:
        t = r["tess_test"]
        print(f"\n{r['run']}: TESS test PR-AUC per class {t['pr_auc']}; "
              f"no-planet PR-AUC real-only {t.get('no_planet_pr_auc_real_only')}")
    os.makedirs("results", exist_ok=True)
    with open("results/finetune_comparison.md", "w") as fh:
        fh.write(md + "\n")


if __name__ == "__main__":
    main()
