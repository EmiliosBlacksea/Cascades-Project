# from __future__ import annotations
# import argparse, os, sys, subprocess

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--processed_dir", required=True)
#     ap.add_argument("--ks", nargs="+", type=int, required=True)
#     ap.add_argument("--device", default="cpu")
#     args = ap.parse_args()

#     for k in args.ks:
#         data_path = os.path.join(args.processed_dir, f"features_k{k}.parquet")
#         if not os.path.exists(data_path):
#             print(f"[WARN] Missing {data_path}, skipping.")
#             continue
#         for model in ["logreg", "mlp"]:
#             cmd = [sys.executable, "-m", "src.train", "--data", data_path, "--model", model, "--device", args.device]
#             print("\n>>", " ".join(cmd))
#             subprocess.run(cmd, check=True)

#     print("\n[DONE]")

# if __name__ == "__main__":
#     main()
from __future__ import annotations
import argparse, os, sys, subprocess
import pandas as pd

IGNORE = {"message_id", "k", "total_retweets", "fk_threshold", "label"}

def feature_cols_from_parquet(path: str):
    df = pd.read_parquet(path, columns=None)
    return [c for c in df.columns if c not in IGNORE]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--ks", nargs="*", type=int, default=None, help="Explicit ks (optional).")
    ap.add_argument("--k_min", type=int, default=None, help="Range start (optional).")
    ap.add_argument("--k_max", type=int, default=None, help="Range end (optional).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--loo", action="store_true", help="Run leave-one-feature-out ablation.")
    ap.add_argument("--models", nargs="+", default=["logreg", "mlp"], choices=["logreg", "mlp"])
    args = ap.parse_args()

    # build ks list
    if args.ks and len(args.ks) > 0:
        ks = args.ks
    else:
        if args.k_min is None or args.k_max is None:
            raise ValueError("Provide either --ks ... or both --k_min and --k_max")
        ks = list(range(args.k_min, args.k_max + 1))

    for k in ks:
        data_path = os.path.join(args.processed_dir, f"features_k{k}.parquet")
        if not os.path.exists(data_path):
            print(f"[WARN] Missing {data_path}, skipping.")
            continue

        # 1) baseline: all features
        for model in args.models:
            cmd = [sys.executable, "-m", "src.train",
                   "--data", data_path, "--model", model, "--device", args.device]
            print("\n>>", " ".join(cmd))
            subprocess.run(cmd, check=True)

        # 2) leave-one-out: drop each feature
        if args.loo:
            feats = feature_cols_from_parquet(data_path)
            print(f"[INFO] k={k} | features={len(feats)}")
            for f in feats:
                for model in args.models:
                    cmd = [sys.executable, "-m", "src.train",
                           "--data", data_path, "--model", model, "--device", args.device,
                           "--drop_feature", f]
                    print("\n>>", " ".join(cmd))
                    subprocess.run(cmd, check=True)

    print("\n[DONE]")

if __name__ == "__main__":
    main()
