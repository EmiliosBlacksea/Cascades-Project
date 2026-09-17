## Ablations (temporal-only / structural-only)

Temporal-only:
```bash
python -m src.preprocess --input data/weibo.txt --out_dir data/processed_temporal --ks 5 10 25 50 100 --no_structural
```

Structural-only:
```bash
python -m src.preprocess --input data/weibo.txt --out_dir data/processed_structural --ks 5 10 25 50 100 --no_temporal
```

Then train:
```bash
python -m src.train --data data/processed_temporal/features_k5.parquet --model logreg
```
