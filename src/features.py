from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from .parse import RetweetEvent

@dataclass
class SnapshotFeatures:
    x: np.ndarray
    feature_names: List[str]

def _safe_mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else 0.0

def _safe_std(x: np.ndarray) -> float:
    return float(x.std()) if x.size else 0.0

def _p90(x: np.ndarray) -> float:
    return float(np.percentile(x, 90)) if x.size else 0.0

def extract_features_for_k(
    root_user_id: int,
    publish_time: int,
    events: List[RetweetEvent],
    k: int,
    include_temporal: bool = True,
    include_structural: bool = True,
    include_user_proxy: bool = True,
) -> SnapshotFeatures:
    """Features from first k retweet events; incremental tree reconstruction."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(events) < k:
        raise ValueError(f"Need {k} events, got {len(events)}")

    events_k = events[:k]

    # --- temporal ---
    # times = np.array([e.t - publish_time for e in events_k], dtype=np.float64)
    times = np.array([e.t for e in events_k], dtype=np.float64)  # e.t is offset seconds
  
    times.sort()
    time_k = float(times[-1]) if times.size else 0.0
    inter = np.diff(times) if times.size >= 2 else np.array([], dtype=np.float64)

    half = (inter.size // 2) if inter.size else 0
    inter_first = inter[:half] if inter.size else np.array([], dtype=np.float64)
    inter_second = inter[half:] if inter.size else np.array([], dtype=np.float64)

    mean_first = _safe_mean(inter_first)
    mean_second = _safe_mean(inter_second)
    trend = mean_second - mean_first

    mean_inter = _safe_mean(inter)
    std_inter = _safe_std(inter)
    burstiness = (std_inter / (mean_inter + 1e-9)) if inter.size else 0.0

    # --- structural (incremental tree) ---
    parent: Dict[int, int] = {root_user_id: -1}
    depth: Dict[int, int] = {root_user_id: 0}
    outdeg: Dict[int, int] = {root_user_id: 0}
    seen = {root_user_id}

    for e in events_k:
        leaf = e.leaf
        if leaf in seen:
            continue
        p = root_user_id
        for u in reversed(e.path[:-1]):
            if u in seen:
                p = u
                break
        parent[leaf] = p
        depth[leaf] = depth.get(p, 0) + 1
        outdeg[p] = outdeg.get(p, 0) + 1
        outdeg.setdefault(leaf, 0)
        seen.add(leaf)

    leaf_nodes = [n for n in seen if n != root_user_id]
    leaf_depths = np.array([depth.get(n, 0) for n in leaf_nodes], dtype=np.float64)

    max_depth = float(leaf_depths.max()) if leaf_depths.size else 0.0
    avg_depth = _safe_mean(leaf_depths)
    p90_depth = _p90(leaf_depths)
    root_children = float(outdeg.get(root_user_id, 0))
    max_outdeg = float(max(outdeg.values())) if outdeg else 0.0
    num_leaves = float(sum(1 for n, d in outdeg.items() if d == 0 and n != root_user_id))

    # --- user proxies ---
    unique_users = float(len(set([e.leaf for e in events_k])))
    duplicate_users = float(k - unique_users)

    feats = []
    names = []

    if include_temporal:
        feats += [time_k, mean_first, mean_second, trend, burstiness, mean_inter, std_inter]
        names += ["time_k", "mean_inter_first_half", "mean_inter_second_half",
                  "interarrival_trend", "burstiness_cv", "mean_interarrival", "std_interarrival"]

    if include_structural:
        feats += [max_depth, avg_depth, p90_depth, root_children, num_leaves, max_outdeg]
        names += ["max_depth_k", "avg_depth_k", "p90_depth_k",
                  "root_outdeg_k", "num_leaves_k", "max_outdeg_k"]

    if include_user_proxy:
        feats += [unique_users, duplicate_users]
        names += ["unique_leaf_users_k", "duplicate_leaf_users_k"]

    x = np.asarray(feats, dtype=np.float32)
    return SnapshotFeatures(x=x, feature_names=names)
