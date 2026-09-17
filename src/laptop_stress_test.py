from __future__ import annotations
import argparse, json, os, sys, subprocess
import pandas as pd

IGNORE = {"message_id", "k", "total_retweets", "fk_threshold", "label"}

def feature_cols_from_parquet(path: str):
    df = pd.read_parquet(path)
    return [c for c in df.columns if c not in IGNORE]

def safe_tag(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)

def metrics_path(data_path: str, model: str, out_dir: str, dropped: list[str] | None):
    base = os.path.basename(data_path).replace(".parquet", "")
    drop_tag = ""
    if dropped:
        drop_tag = "_drop_" + "_".join(safe_tag(x) for x in dropped)
    return os.path.join(out_dir, "metrics", f"{base}_{model}{drop_tag}.json")

def run_train(data_path: str, model: str, out_dir: str, device: str, dropped: list[str] | None):
    cmd = [sys.executable, "-m", "src.train",
           "--data", data_path, "--model", model, "--device", device, "--out_dir", out_dir]
    if dropped:
        for f in dropped:
            cmd += ["--drop_feature", f]
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--k_min", type=int, default=1)
    ap.add_argument("--k_max", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=["logreg", "mlp"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--log", default="reports/overnight_loo.log")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--max_drop_features", type=int, default=None,
                    help="Optional: limit number of features to drop (for debugging).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    TEMPORAL = [
    "time_k",
    "mean_inter_first_half",
    "mean_inter_second_half",
    "interarrival_trend",
    "burstiness_cv",
    "mean_interarrival",
    "std_interarrival",
]


    def log(msg: str):
        print(msg)
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"START {pd.Timestamp.now()}")

    for k in range(args.k_min, args.k_max + 1):
        data_path = os.path.join(args.processed_dir, f"features_k{k}.parquet")
        if not os.path.exists(data_path):
            log(f"[k={k}] missing {data_path} -> skip")
            continue

        feats = feature_cols_from_parquet(data_path)
        if args.max_drop_features is not None:
            feats = feats[:args.max_drop_features]

        # Baseline first (dropped = None)
        for model in args.models:
            mpath = metrics_path(data_path, model, args.out_dir, dropped=None)
            if args.skip_existing and os.path.exists(mpath):
                log(f"[k={k}] {model} baseline exists -> skip")
            else:
                log(f">> k={k} model={model} BASELINE")
                try:
                    run_train(data_path, model, args.out_dir, args.device, dropped=None)
                except Exception as e:
                    log(f"[ERROR] k={k} model={model} baseline failed: {e}")

        # Leave-one-out
        for drop_f in feats:
            for model in args.models:
                mpath = metrics_path(data_path, model, args.out_dir, dropped=[drop_f])
                if args.skip_existing and os.path.exists(mpath):
                    log(f"[k={k}] {model} drop={drop_f} exists -> skip")
                    continue
                log(f">> k={k} model={model} DROP {drop_f}")
                try:
                    run_train(data_path, model, args.out_dir, args.device, dropped=[drop_f])
                except Exception as e:
                    log(f"[ERROR] k={k} model={model} drop={drop_f} failed: {e}")
                    # NO TEMPORAL (drop all tempered/temporal features once)
        for model in args.models:
            mpath = metrics_path(data_path, model, args.out_dir, dropped=TEMPORAL)
            if args.skip_existing and os.path.exists(mpath):
                log(f"[k={k}] {model} no-temporal exists -> skip")
                continue
            log(f">> k={k} model={model} DROP ALL TEMPORAL (no-temporal)")
            try:
                run_train(data_path, model, args.out_dir, args.device, dropped=TEMPORAL)
            except Exception as e:
                log(f"[ERROR] k={k} model={model} no-temporal failed: {e}")


        

    log(f"END {pd.Timestamp.now()}")

if __name__ == "__main__":
    main()
