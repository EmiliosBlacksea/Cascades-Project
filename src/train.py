from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import re

from .datasets import TabularCascadeDataset, fit_standardizer
from .models import LogisticRegression, MLP
from .eval import evaluate
from typing import List



def _safe_tag(s: str) -> str:
    # για filename σε Windows
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)

def get_feature_cols(df: pd.DataFrame, drop_features=None) -> List[str]:
    ignore = {"message_id", "k", "total_retweets", "fk_threshold", "label"}
    cols = [c for c in df.columns if c not in ignore]
    if drop_features:
        drop = set(drop_features)
        cols = [c for c in cols if c not in drop]
    return cols


# def get_feature_cols(df: pd.DataFrame):
#     ignore = {"message_id", "k", "total_retweets", "label"}
#     return [c for c in df.columns if c not in ignore]

def train_one(
    data_path: str,
    model_type: str,
    out_dir: str = "reports",
    seed: int = 42,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    patience: int = 5,
    device: str = "cpu",
    drop_feature: List[str] | None = None,
):

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    df = pd.read_parquet(data_path)
    drop_feature = drop_feature or []
    feature_cols = get_feature_cols(df, drop_features=drop_feature)
    print(f"[INFO] Using {len(feature_cols)} features. Dropped={drop_feature}")

    if len(feature_cols) < 1:
        raise ValueError("No features left after dropping.")


    train_df, test_df = train_test_split(df, test_size=0.15, random_state=seed, stratify=df["label"])
    train_df, val_df = train_test_split(train_df, test_size=0.1765, random_state=seed, stratify=train_df["label"])  # ~15%

    stdzr = fit_standardizer(train_df[feature_cols].to_numpy(dtype=np.float32))

    train_ds = TabularCascadeDataset(train_df, feature_cols, standardizer=stdzr)
    val_ds = TabularCascadeDataset(val_df, feature_cols, standardizer=stdzr)
    test_ds = TabularCascadeDataset(test_df, feature_cols, standardizer=stdzr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    d_in = len(feature_cols)
    if model_type == "logreg":
        model = LogisticRegression(d_in)
    elif model_type == "mlp":
        model = MLP(d_in)
    else:
        raise ValueError("model must be logreg or mlp")

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_score = -1.0
    best_epoch = -1
    bad = 0

    ckpt_path = os.path.join(
        out_dir, "checkpoints",
        f"{os.path.basename(data_path).replace('.parquet','')}_{model_type}.pt"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * x.size(0)
        train_loss = total_loss / len(train_ds)

        val_metrics = evaluate(model, val_loader, device=device)
        score = val_metrics["auc"]
        if np.isnan(score):
            score = val_metrics["accuracy"]

        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "standardizer": {"mean": stdzr.mean.tolist(), "std": stdzr.std.tolist()},
            }, ckpt_path)
        else:
            bad += 1

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_metrics['accuracy']:.3f} val_f1={val_metrics['f1']:.3f} val_auc={val_metrics['auc']:.3f} | best={best_score:.3f} (ep {best_epoch})")
        if bad >= patience:
            print("[EARLY STOP]")
            break

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device=device)

    meta = {
        "data": data_path,
        "model": model_type,
        "feature_dim": d_in,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    }
    def _safe_tag(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    drop_tag = ""
    drop_tag = ""
    if drop_feature:
        drop_tag = "_drop_" + "_".join(_safe_tag(x) for x in drop_feature)


    out_json = os.path.join(
        out_dir, "metrics",
        f"{os.path.basename(data_path).replace('.parquet','')}_{model_type}{drop_tag}.json"
    )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] metrics -> {out_json}")
    print(f"[TEST] acc={test_metrics['accuracy']:.3f} f1={test_metrics['f1']:.3f} auc={test_metrics['auc']:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
    "--drop_feature",
    action="append",
    default=[],
    help="Drop a feature column (can be given multiple times)."
    )

    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    train_one(
        data_path=args.data,
        model_type=args.model,
        out_dir=args.out_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        drop_feature=args.drop_feature,
    )

if __name__ == "__main__":
    main()
