#!/bin/bash
# Fresh download, convert, and filter for SO-101 table-cleanup dataset

DATASET_PATH="/home/ubuntu/dreamzero/so101-table-cleanup"
HF_REPO="youliangtan/so101-table-cleanup"

# Step 1: Remove existing converted dataset (keep a backup of parquets just in case)
echo "Removing stale converted dataset..."
rm -rf $DATASET_PATH

# Step 2: Download fresh from HuggingFace
echo "Downloading fresh dataset..."
huggingface-cli download $HF_REPO \
    --repo-type dataset \
    --local-dir $DATASET_PATH

# Step 3: Convert to GEAR format
echo "Converting to GEAR format..."
cd /home/ubuntu/dreamzero
python scripts/data/convert_lerobot_to_gear.py \
    --dataset-path $DATASET_PATH \
    --embodiment-tag so101 \
    --state-keys '{"joint_pos": [0, 5], "gripper_pos": [5, 6]}' \
    --action-keys '{"joint_pos": [0, 5], "gripper_pos": [5, 6]}' \
    --relative-action-keys joint_pos gripper_pos \
    --task-key task_index

# Step 4: Filter episodes shorter than action_horizon=24 using ground truth from parquets
echo "Filtering short episodes..."
python3 -c "
import json, glob
import pandas as pd

dataset_path = '$DATASET_PATH'

# Ground truth lengths from parquets
parquet_files = sorted(glob.glob(f'{dataset_path}/data/**/*.parquet', recursive=True))
episode_lengths = {}
for pf in parquet_files:
    df = pd.read_parquet(pf, columns=['episode_index'])
    for ep_idx, count in df.groupby('episode_index').size().items():
        episode_lengths[int(ep_idx)] = episode_lengths.get(int(ep_idx), 0) + count

# Fix episodes.jsonl lengths and filter
episodes_path = f'{dataset_path}/meta/episodes.jsonl'
episodes = [json.loads(l) for l in open(episodes_path)]

result = []
removed = []
for e in episodes:
    real_len = episode_lengths.get(e['episode_index'], 0)
    e['length'] = real_len  # overwrite with ground truth
    if real_len >= 24:
        result.append(e)
    else:
        removed.append(e)

print(f'Total: {len(episodes)}')
print(f'Removed {len(removed)} short episodes:')
for e in removed:
    print(f'  episode_index={e[\"episode_index\"]}, length={e[\"length\"]}')
print(f'Remaining: {len(result)}')

with open(episodes_path, 'w') as f:
    for e in result:
        f.write(json.dumps(e) + '\n')
print('Done')
"

echo "Dataset ready at $DATASET_PATH"
