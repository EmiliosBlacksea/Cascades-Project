from __future__ import annotations
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: str = "cpu") -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).detach().cpu().numpy().reshape(-1)
        prob = 1.0 / (1.0 + np.exp(-logits))
        ys.append(y.numpy().reshape(-1))
        ps.append(prob)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= 0.5).astype(np.int32)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc"] = float("nan")
    return out
