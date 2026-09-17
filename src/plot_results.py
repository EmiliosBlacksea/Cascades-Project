from __future__ import annotations
import argparse, json, os, re
import pandas as pd
import matplotlib.pyplot as plt

def parse_fname(fname: str):
    # expects features_k{K}_{model}.json
    m = re.search(r"features_k(\d+)_(logreg|mlp)\.json$", fname)
    if not m:
        return None
    return int(m.group(1)), m.group(2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="allldata")
    ap.add_argument("--out_dir", default="reports/plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for fn in os.listdir(args.metrics_dir):
        parsed = parse_fname(fn)
        if parsed is None:
            continue
        k, model = parsed
        path = os.path.join(args.metrics_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        tm = d.get("test_metrics", {})
        rows.append({
            "k": k,
            "model": model,
            "accuracy": tm.get("accuracy"),
            "f1": tm.get("f1"),
            "auc": tm.get("auc"),
            "best_epoch": d.get("best_epoch"),
            "n_train": d.get("n_train"),
            "n_val": d.get("n_val"),
            "n_test": d.get("n_test"),
        })

    if not rows:
        raise SystemExit(f"No metrics found in {args.metrics_dir}")

    df = pd.DataFrame(rows).sort_values(["model", "k"])
    print(df)

    # Save summary table
    out_csv = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved {out_csv}")

    # Plot Accuracy vs k
    plt.figure()
    for model, g in df.groupby("model"):
        g = g.sort_values("k")
        plt.plot(g["k"], g["accuracy"], marker="o", label=model)
    plt.xlabel("k")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(args.out_dir, "accuracy_vs_k.png")
    plt.savefig(acc_path, dpi=200, bbox_inches="tight")
    print(f"[OK] saved {acc_path}")

    # Plot AUC vs k (skip NaNs)
    plt.figure()
    for model, g in df.groupby("model"):
        g = g.sort_values("k")
        gg = g.dropna(subset=["auc"])
        if len(gg) == 0:
            continue
        plt.plot(gg["k"], gg["auc"], marker="o", label=model)
    plt.xlabel("k")
    plt.ylabel("Test ROC-AUC")
    plt.legend()
    plt.grid(True)
    auc_path = os.path.join(args.out_dir, "auc_vs_k.png")
    plt.savefig(auc_path, dpi=200, bbox_inches="tight")
    print(f"[OK] saved {auc_path}")

if __name__ == "__main__":
    main()
