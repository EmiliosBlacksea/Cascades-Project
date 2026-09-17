from __future__ import annotations
import argparse
import os
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

from .parse import parse_weibo_line, build_incremental_events
from .features import extract_features_for_k


def compute_fk_thresholds(input_path: str, ks: List[int], max_lines: int | None = None) -> Dict[int, float]:
    """
    1st pass: compute f(k) = median(final_size) over cascades with final_size >= k.
    """
    totals_by_k: Dict[int, List[int]] = {k: [] for k in ks}

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(tqdm(f, desc="Pass 1/2: computing f(k)")):
            if max_lines is not None and idx >= max_lines:
                break
            parsed = parse_weibo_line(line)
            if parsed is None:
                continue
            events_raw = parsed["events_raw"]
            if not events_raw:
                continue

            root = parsed["root_user_id"]
            events = build_incremental_events(root, events_raw)
            total = len(events)

            for k in ks:
                if total >= k:
                    totals_by_k[k].append(total)

    fk: Dict[int, float] = {}
    for k in ks:
        arr = np.array(totals_by_k[k], dtype=np.int32)
        if arr.size == 0:
            fk[k] = float("nan")
        else:
            # "upper median" (robust). You can also use np.median(arr).
            arr.sort()
            fk[k] = float(arr[arr.size // 2])

    return fk


def preprocess(
    input_path: str,
    out_dir: str,
    ks: List[int],
    max_lines: int | None = None,
    include_temporal: bool = True,
    include_structural: bool = True,
    include_user_proxy: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # --- compute thresholds f(k) ---
    fk = compute_fk_thresholds(input_path, ks, max_lines=max_lines)
    print("[INFO] f(k) thresholds:", {k: fk[k] for k in ks})

    rows_by_k: Dict[int, List[dict]] = {k: [] for k in ks}
    feature_names_by_k: Dict[int, List[str]] = {}

    # --- 2nd pass: build features + labels using f(k) ---
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(tqdm(f, desc="Pass 2/2: features+labels")):
            if max_lines is not None and idx >= max_lines:
                break
            parsed = parse_weibo_line(line)
            if parsed is None:
                continue

            mid = parsed["message_id"]
            root = parsed["root_user_id"]
            t0 = parsed["publish_time"]
            events_raw = parsed["events_raw"]
            if not events_raw:
                continue

            events = build_incremental_events(root, events_raw)
            total = len(events)

            for k in ks:
                if total < k:
                    continue
                if np.isnan(fk[k]):
                    continue

                # label by median threshold f(k) (balanced by design-ish)
                label = 1 if total >= fk[k] else 0

                snap = extract_features_for_k(
                    root_user_id=root,
                    publish_time=t0,
                    events=events,
                    k=k,
                    include_temporal=include_temporal,
                    include_structural=include_structural,
                    include_user_proxy=include_user_proxy,
                )
                feature_names_by_k.setdefault(k, snap.feature_names)

                row = {
                    "message_id": mid,
                    "k": k,
                    "total_retweets": total,
                    "fk_threshold": float(fk[k]),
                    "label": int(label),
                }
                for name, val in zip(snap.feature_names, snap.x.tolist()):
                    row[name] = float(val)
                rows_by_k[k].append(row)

    # --- save ---
    for k in ks:
        rows = rows_by_k[k]
        if not rows:
            print(f"[WARN] No rows for k={k}")
            continue
        df = pd.DataFrame(rows)
        out_path = os.path.join(out_dir, f"features_k{k}.parquet")
        df.to_parquet(out_path, index=False)

        vc = df["label"].value_counts(dropna=False).to_dict()
        print(f"[OK] Saved k={k}: {len(df):,} rows -> {out_path} | label_counts={vc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ks", nargs="+", type=int, required=True)
    ap.add_argument("--max_lines", type=int, default=None)
    ap.add_argument("--no_temporal", action="store_true")
    ap.add_argument("--no_structural", action="store_true")
    ap.add_argument("--no_user_proxy", action="store_true")
    args = ap.parse_args()

    preprocess(
        input_path=args.input,
        out_dir=args.out_dir,
        ks=args.ks,
        max_lines=args.max_lines,
        include_temporal=not args.no_temporal,
        include_structural=not args.no_structural,
        include_user_proxy=not args.no_user_proxy,
    )


if __name__ == "__main__":
    main()
