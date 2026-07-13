"""
DreamZero SO-101 Dataset Validator
python validate_dataset.py --dataset-path /home/ubuntu/harsha/dreamzero/so101-table-cleanup --action-horizon 24 --action-dim 6
Checks everything that can cause training to crash mid-run.
Usage: python validate_dataset.py --dataset-path /path/to/dataset --action-horizon 24 --action-dim 6
"""

import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path


def load_meta(dataset_path):
    meta = {}
    for fname in ["episodes.jsonl", "tasks.jsonl", "info.json", "modality.json", "embodiment.json"]:
        fpath = Path(dataset_path) / "meta" / fname
        if not fpath.exists():
            meta[fname] = None
            continue
        if fname.endswith(".jsonl"):
            with open(fpath) as f:
                meta[fname] = [json.loads(l) for l in f if l.strip()]
        else:
            with open(fpath) as f:
                meta[fname] = json.load(f)
    return meta


def check_meta_files(meta, errors, warnings):
    print("\n[1] Checking meta files...")
    required = ["episodes.jsonl", "tasks.jsonl", "info.json", "modality.json"]
    for f in required:
        if meta[f] is None:
            errors.append(f"Missing required meta file: {f}")
        else:
            print(f"  OK: {f}")

    # Check embodiment tag
    if meta.get("embodiment.json"):
        tag = meta["embodiment.json"].get("embodiment_tag")
        print(f"  OK: embodiment_tag = {tag}")
    else:
        warnings.append("embodiment.json missing — embodiment tag won't be auto-detected")

    # Check modality annotation key
    if meta["modality.json"]:
        ann = meta["modality.json"].get("annotation", {})
        if not ann:
            errors.append("modality.json has no annotation section — language conditioning will get zero embeddings")
        elif "task" not in ann:
            errors.append(f"modality.json annotation missing 'task' key, found: {list(ann.keys())}")
        else:
            orig = ann["task"].get("original_key")
            print(f"  OK: annotation.task original_key = {orig}")

        # Check video keys
        video = meta["modality.json"].get("video", {})
        print(f"  OK: video keys = {list(video.keys())}")

        # Check state/action keys
        state = meta["modality.json"].get("state", {})
        action = meta["modality.json"].get("action", {})
        print(f"  OK: state keys = {list(state.keys())}")
        print(f"  OK: action keys = {list(action.keys())}")


def check_episodes(meta, action_horizon, errors, warnings):
    print(f"\n[2] Checking episodes (action_horizon={action_horizon})...")
    if not meta["episodes.jsonl"]:
        errors.append("episodes.jsonl is empty or missing")
        return

    episodes = meta["episodes.jsonl"]
    lengths = [e["length"] for e in episodes]

    short = [e for e in episodes if e["length"] < action_horizon]
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Min length: {min(lengths)}")
    print(f"  Max length: {max(lengths)}")
    print(f"  Median length: {np.median(lengths):.0f}")
    print(f"  Mean length: {np.mean(lengths):.1f}")
    print(f"  Episodes shorter than action_horizon ({action_horizon}): {len(short)}")

    if short:
        for e in short:
            errors.append(
                f"Episode {e['episode_index']} has length {e['length']} < action_horizon {action_horizon} "
                f"— will cause shape mismatch during batching"
            )

    # Check for episodes with no tasks
    no_task = [e for e in episodes if not e.get("tasks") or e["tasks"] == [""]]
    if no_task:
        warnings.append(f"{len(no_task)} episodes have empty task annotations — language conditioning will be blank")

    # Check task indices resolve against tasks.jsonl
    if meta["tasks.jsonl"]:
        valid_task_texts = {t["task"] for t in meta["tasks.jsonl"]}
        for e in episodes:
            for t in e.get("tasks", []):
                if t and t not in valid_task_texts:
                    errors.append(f"Episode {e['episode_index']} has task '{t}' not found in tasks.jsonl")
    else:
        errors.append("tasks.jsonl missing — language conditioning will fail")


def check_parquet_files(dataset_path, meta, action_dim, action_horizon, errors, warnings):
    print(f"\n[3] Checking parquet files (action_dim={action_dim})...")
    parquet_files = sorted(glob.glob(str(Path(dataset_path) / "data" / "**" / "*.parquet"), recursive=True))

    if not parquet_files:
        errors.append("No parquet files found under data/")
        return

    print(f"  Found {len(parquet_files)} parquet files")

    # Get expected columns from modality.json
    expected_state_keys = []
    expected_action_keys = []
    if meta["modality.json"]:
        for k, v in meta["modality.json"].get("state", {}).items():
            expected_state_keys.append(v.get("original_key", k))
        for k, v in meta["modality.json"].get("action", {}).items():
            expected_action_keys.append(v.get("original_key", k))

    action_value_violations = 0
    state_dim_mismatches = 0
    action_dim_mismatches = 0
    total_rows = 0
    short_chunk_episodes = set()

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        total_rows += len(df)

        # Check required columns exist
        for col in expected_state_keys + expected_action_keys:
            if col not in df.columns and col not in ["action", "observation.state"]:
                errors.append(f"Missing column '{col}' in {os.path.basename(pf)}")

        # Check action column
        if "action" in df.columns:
            action_vals = np.stack(df["action"].values)

            # Action dim check
            if action_vals.shape[1] != action_dim:
                action_dim_mismatches += 1
                if action_dim_mismatches <= 3:
                    errors.append(
                        f"Action dim mismatch in {os.path.basename(pf)}: "
                        f"expected {action_dim}, got {action_vals.shape[1]}"
                    )

        # Check state column
        if "observation.state" in df.columns:
            state_vals = np.stack(df["observation.state"].values)
            if state_vals.shape[1] != action_dim:
                state_dim_mismatches += 1

        # Check task_index resolves
        if "task_index" in df.columns and meta["tasks.jsonl"]:
            valid_indices = {t["task_index"] for t in meta["tasks.jsonl"]}
            invalid = set(df["task_index"].unique()) - valid_indices
            if invalid:
                errors.append(f"task_index values {invalid} in {os.path.basename(pf)} not in tasks.jsonl")

        # Check for NaN/Inf in action and state
        if "action" in df.columns:
            action_arr = np.stack(df["action"].values)
            if np.any(np.isnan(action_arr)):
                errors.append(f"NaN values in action column of {os.path.basename(pf)}")
            if np.any(np.isinf(action_arr)):
                errors.append(f"Inf values in action column of {os.path.basename(pf)}")

        if "observation.state" in df.columns:
            state_arr = np.stack(df["observation.state"].values)
            if np.any(np.isnan(state_arr)):
                errors.append(f"NaN values in observation.state of {os.path.basename(pf)}")

    print(f"  Total rows across all parquets: {total_rows}")
    if action_dim_mismatches:
        print(f"  Action dim mismatches: {action_dim_mismatches} files")
    if state_dim_mismatches:
        warnings.append(f"State dim mismatches in {state_dim_mismatches} files — check state_keys config")
    if action_value_violations:
        warnings.append(
            f"{action_value_violations} action values outside [-1,1] — normalization may not be applied yet, "
            f"this will cause assertion failure in forward pass if not normalized by transform"
        )
    if not action_dim_mismatches and not state_dim_mismatches:
        print(f"  OK: action_dim={action_dim} consistent across files")


def check_relative_stats(dataset_path, action_horizon, errors, warnings):
    print("\n[4] Checking relative action stats...")
    stats_path = Path(dataset_path) / "meta" / "relative_stats_dreamzero.json"
    if not stats_path.exists():
        warnings.append(
            "relative_stats_dreamzero.json missing — relative action normalization will be skipped "
            "or recomputed at training start (slow)"
        )
        return

    with open(stats_path) as f:
        stats = json.load(f)

    for key, vals in stats.items():
        required_fields = ["max", "min", "mean", "std", "q01", "q99"]
        missing = [f for f in required_fields if f not in vals]
        if missing:
            errors.append(f"relative_stats_dreamzero.json key '{key}' missing fields: {missing}")
        else:
            print(f"  OK: {key} stats present ({len(vals['mean'])} dims)")

        # Check for degenerate stats
        std = np.array(vals.get("std", [1]))
        if np.any(std == 0):
            warnings.append(f"Zero std in relative stats for '{key}' — will cause division by zero during normalization")


def check_video_files(dataset_path, meta, errors, warnings):
    print("\n[5] Checking video files...")
    if not meta["modality.json"]:
        return

    video_keys = list(meta["modality.json"].get("video", {}).keys())
    if not video_keys:
        errors.append("No video keys in modality.json")
        return

    print(f"  Expected video keys: {video_keys}")
    print(f"  Camera count: {len(video_keys)} (ensure num_views={len(video_keys)} in training script)")

    # Check video directories exist
    video_dir = Path(dataset_path) / "videos"
    if not video_dir.exists():
        errors.append(f"videos/ directory missing at {video_dir}")
        return

    for vk in video_keys:
        # Try short key first (e.g. 'front'), then original key variants
        original_key = meta["modality.json"]["video"][vk].get("original_key", vk)
        candidates = [vk, original_key, original_key.replace(".", "/")]

        found = False
        for candidate in candidates:
            chunks = list(video_dir.glob(f"chunk-*/{candidate}"))
            if chunks:
                total_vids = sum(len(list(c.glob("*.mp4")) + list(c.glob("*.mkv"))) for c in chunks)
                print(f"  OK: {vk} (path: {candidate}) — {total_vids} video files across {len(chunks)} chunks")
                found = True
                break
            # Also try flat mp4 directly under chunk
            flat = list(video_dir.glob(f"chunk-*/{candidate}*.mp4"))
            if flat:
                print(f"  OK: {vk} — {len(flat)} video files found")
                found = True
                break

        if not found:
            errors.append(f"No video files found for key '{vk}' (tried: {candidates}) under videos/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--action-dim", type=int, default=6)
    args = parser.parse_args()

    print(f"Validating dataset: {args.dataset_path}")
    print(f"action_horizon={args.action_horizon}, action_dim={args.action_dim}")

    errors = []
    warnings = []

    meta = load_meta(args.dataset_path)
    check_meta_files(meta, errors, warnings)
    check_episodes(meta, args.action_horizon, errors, warnings)
    check_parquet_files(args.dataset_path, meta, args.action_dim, args.action_horizon, errors, warnings)
    check_relative_stats(args.dataset_path, args.action_horizon, errors, warnings)
    check_video_files(args.dataset_path, meta, errors, warnings)

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN] {w}")
    if errors:
        print(f"\nERRORS ({len(errors)}) — these will crash training:")
        for e in errors:
            print(f"  [ERROR] {e}")
    if not errors and not warnings:
        print("\nAll checks passed — dataset looks clean.")
    elif not errors:
        print("\nNo blocking errors — warnings are non-fatal but worth reviewing.")
    else:
        print(f"\nFix {len(errors)} error(s) before launching training.")


if __name__ == "__main__":
    main()
