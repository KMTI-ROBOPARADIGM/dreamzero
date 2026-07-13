 
#!/usr/bin/env python3
"""
Check each episode's step count against a chunk horizon (default 24).

Reports:
  - per-episode step count and whether it's divisible by the horizon
  - total count / list of episodes that are NOT a multiple of horizon
  - remainder (steps needed to pad, or steps over) for each bad episode

Usage:
    python check_episode_horizon.py --dataset ~/harsha/dreamzero/so101-table-cleanup --horizon 24
    python check_episode_horizon.py --dataset ~/harsha/dreamzero/so101-table-cleanup --horizon 24 --key action
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def find_episode_files(dataset_root: Path):
    """
    Locate per-episode parquet files under a LeRobot v2-style dataset.
    Typical layout: data/chunk-000/episode_000000.parquet ...
    Falls back to scanning for any *.parquet if that structure isn't found.
    """
    data_dir = dataset_root / "data"
    if data_dir.exists():
        files = sorted(data_dir.rglob("episode_*.parquet"))
        if files:
            return files
    # fallback: scan whole dataset root
    return sorted(dataset_root.rglob("episode_*.parquet"))


def load_episode_index(dataset_root: Path):
    """
    Load episodes.jsonl if present, to cross-reference episode indices
    and catch orphan parquet files (like the indices 23/24 issue in your notes).
    Returns dict: episode_index -> metadata dict (or {} if not found).
    """
    candidates = [
        dataset_root / "meta" / "episodes.jsonl",
        dataset_root / "episodes.jsonl",
    ]
    for path in candidates:
        if path.exists():
            index = {}
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    ep_idx = rec.get("episode_index")
                    if ep_idx is not None:
                        index[ep_idx] = rec
            return index
    return {}


def episode_index_from_filename(path: Path):
    # episode_000000.parquet -> 0
    stem = path.stem  # episode_000000
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="Path to dataset root")
    ap.add_argument("--horizon", type=int, default=24, help="Chunk horizon to check divisibility against (default 24)")
    ap.add_argument("--key", default=None,
                     help="Optional: name of a column (e.g. 'action' or 'observation.state') to count rows from. "
                          "If omitted, uses len(dataframe) i.e. total rows in the parquet as the step count.")
    ap.add_argument("--csv-out", default=None, help="Optional path to write full per-episode results as CSV")
    args = ap.parse_args()

    dataset_root = Path(args.dataset).expanduser().resolve()
    if not dataset_root.exists():
        print(f"ERROR: dataset path does not exist: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    episode_files = find_episode_files(dataset_root)
    if not episode_files:
        print(f"ERROR: no episode_*.parquet files found under {dataset_root}", file=sys.stderr)
        sys.exit(1)

    episode_meta = load_episode_index(dataset_root)

    rows = []
    for f in episode_files:
        ep_idx = episode_index_from_filename(f)
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            rows.append({
                "episode_index": ep_idx,
                "file": f.name,
                "n_steps": None,
                "remainder": None,
                "divisible": None,
                "error": str(e),
            })
            continue

        if args.key is not None:
            if args.key not in df.columns:
                print(f"WARNING: key '{args.key}' not found in {f.name}, columns: {list(df.columns)}", file=sys.stderr)
                n_steps = len(df)
            else:
                n_steps = len(df[args.key])
        else:
            n_steps = len(df)

        remainder = n_steps % args.horizon
        rows.append({
            "episode_index": ep_idx,
            "file": f.name,
            "n_steps": n_steps,
            "remainder": remainder,
            "pad_needed": (args.horizon - remainder) if remainder != 0 else 0,
            "divisible": remainder == 0,
            "in_episodes_jsonl": ep_idx in episode_meta if episode_meta else None,
            "error": None,
        })

    result_df = pd.DataFrame(rows).sort_values("episode_index", na_position="last")

    n_total = len(result_df)
    n_bad = int((~result_df["divisible"].fillna(False)).sum())
    n_ok = n_total - n_bad
    n_errors = int(result_df["error"].notna().sum())

    print(f"Dataset: {dataset_root}")
    print(f"Horizon: {args.horizon}")
    if args.key:
        print(f"Counting steps from column: {args.key}")
    print(f"Total episode files found: {n_total}")
    print(f"  OK (divisible by {args.horizon}):     {n_ok}")
    print(f"  NOT divisible by {args.horizon}:      {n_bad}")
    if n_errors:
        print(f"  Failed to read (errors):        {n_errors}")
    print()

    if n_bad > 0:
        print(f"Episodes NOT divisible by {args.horizon}:")
        print("-" * 70)
        bad = result_df[result_df["divisible"] == False]  # noqa: E712
        for _, r in bad.iterrows():
            orphan_flag = ""
            if episode_meta and r["in_episodes_jsonl"] is False:
                orphan_flag = "  [ORPHAN: not in episodes.jsonl]"
            print(f"  ep {r['episode_index']:>4}  {r['file']:<30}  n_steps={r['n_steps']:<6}  "
                  f"remainder={r['remainder']:<3}  pad_needed={r['pad_needed']}{orphan_flag}")
        print("-" * 70)

    if n_errors > 0:
        print("\nEpisodes that failed to read:")
        for _, r in result_df[result_df["error"].notna()].iterrows():
            print(f"  {r['file']}: {r['error']}")

    if args.csv_out:
        result_df.to_csv(args.csv_out, index=False)
        print(f"\nFull results written to: {args.csv_out}")


if __name__ == "__main__":
    main()
