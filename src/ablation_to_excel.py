import os
import json
import math
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd


def safe_get_metrics(json_path: str) -> Tuple[float, float]:
    """
    Returns (accuracy, auc) from a result json.
    If file missing or malformed, returns (nan, nan).
    """
    if not os.path.isfile(json_path):
        return (math.nan, math.nan)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            obj: Dict[str, Any] = json.load(f)

        tm = obj.get("test_metrics", {}) or {}
        acc = tm.get("accuracy", math.nan)
        auc = tm.get("auc", math.nan)

        # ensure numeric
        acc = float(acc) if acc is not None else math.nan
        auc = float(auc) if auc is not None else math.nan
        return (acc, auc)

    except Exception:
        return (math.nan, math.nan)


def main():
    # ====== EDIT THESE ======
    results_dir = r"allldata"     # folder that contains your json files
    out_xlsx = r"ablation_summary.xlsx" # output excel
    k_min, k_max = 1, 200
    # =======================

    # List the variants you want as columns.
    # "stem" is what comes AFTER "features_k{K}_logreg" and BEFORE ".json"
    variants: List[Dict[str, str]] = [
        {"name": "baseline", "stem": ""},

        # From your list:
        {"name": "drop_burstiness_cv", "stem": "_drop_burstiness_cv"},
        {"name": "drop_duplicate_leaf_users_k", "stem": "_drop_duplicate_leaf_users_k"},
        {"name": "drop_interarrival_trend", "stem": "_drop_interarrival_trend"},
        {"name": "drop_max_depth_k", "stem": "_drop_max_depth_k"},
        {"name": "drop_max_outdeg_k", "stem": "_drop_max_outdeg_k"},
        {"name": "drop_mean_inter_second_half", "stem": "_drop_mean_inter_second_half"},
        {"name": "drop_mean_inter_first_half", "stem": "_drop_mean_inter_first_half"},
        {"name": "drop_mean_interarrival", "stem": "_drop_mean_interarrival"},
        {"name": "drop_num_leaves_k", "stem": "_drop_num_leaves_k"},
        {"name": "drop_p90_depth_k", "stem": "_drop_p90_depth_k"},
        {"name": "drop_root_outdeg_k", "stem": "_drop_root_outdeg_k"},
        {"name": "drop_std_interarrival", "stem": "_drop_std_interarrival"},
        {"name": "drop_time_k", "stem": "_drop_time_k"},
        {
            "name": "drop_time_k_plus_many",
            "stem": "_drop_time_k_mean_inter_first_half_mean_inter_second_half_interarrival_trend_burstiness_cv_mean_interarrival_std_interarrival",
        },
        {"name": "drop_unique_leaf_users_k", "stem": "_drop_unique_leaf_users_k"},
    ]

    # Build table
    rows = []
    for k in range(k_min, k_max + 1):
        row: Dict[str, Any] = {"k": k}

        for v in variants:
            filename = f"features_k{k}_logreg{v['stem']}.json"
            path = os.path.join(results_dir, filename)

            acc, auc = safe_get_metrics(path)
            row[f"{v['name']}_acc"] = acc
            row[f"{v['name']}_auc"] = auc

        rows.append(row)

    df = pd.DataFrame(rows)

    # Optional: keep columns grouped (acc, auc) per variant, matching your desired Excel layout
    ordered_cols = ["k"]
    for v in variants:
        ordered_cols.append(f"{v['name']}_acc")
        ordered_cols.append(f"{v['name']}_auc")
    df = df[ordered_cols]

    # Write Excel
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="metrics")

    print(f"Saved: {out_xlsx}")
    print(f"Rows: {len(df)} (k={k_min}..{k_max})")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
