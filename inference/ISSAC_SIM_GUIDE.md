# DreamZero × SO-101 Isaac Sim — Setup Guide

Testing a DreamZero LoRA fine-tune (cross-embodiment, SO-101 arm) in Isaac Sim
using LeIsaac's PickOrange environment as the sim scene.

**Two-machine setup:**
- **4090 box** — runs Isaac Sim + LeIsaac (sim scene + client script)
- **H200 box** — runs DreamZero inference server (or mock server for testing)

---

## Requirements

| Machine | GPU | OS |
|---|---|---|
| 4090 box | NVIDIA RTX 4090 (24 GB) | Ubuntu 22.04 / 24.04 with Selkies-EGL |
| H200 box | NVIDIA H200 | Ubuntu 22.04 / 24.04 |

---

## Phase 0 — Verify the container (4090 box)

Before installing anything, confirm the container is GPU-ready.

```bash
nvidia-smi
vulkaninfo --summary
eglinfo | head -20
lsb_release -a
python3 --version
```

Expected: `nvidia-smi` shows RTX 4090, `vulkaninfo` shows a GPU device (not
ERROR_INCOMPATIBLE_DRIVER), `eglinfo` shows your GPU name, Ubuntu 22.04/24.04.

**Quick Isaac Sim sanity check** (if already installed):

```bash
python -c "
from isaacsim.simulation_app import SimulationApp
app = SimulationApp({'headless': False})
import time; time.sleep(30)
app.close()
"
```

The Isaac Sim viewport should appear in your Selkies browser stream for 30s
then close. If it opens and closes — Isaac Sim is working.

---

## Phase 1 — Conda environment

```bash
conda create -n leisaac python=3.11 -y
conda activate leisaac

conda install -c "nvidia/label/cuda-12.8.1" cuda-toolkit -y

pip install -U torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

pip install "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com
```

> **Note:** torch 2.5.1 + cu124 is the exact combination Isaac Sim 5.1 was
> built against. Using a different version causes a
> `libtorch_cuda_linalg.so: cannot open shared object file` crash.

### Fix the torch library path (must be set every session)

```bash
export LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/leisaac/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
```

Make it permanent:

```bash
# In ~/.bashrc
echo 'export LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/leisaac/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Also set it via conda activate hook so it's automatic
mkdir -p /home/ubuntu/miniconda3/envs/leisaac/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/leisaac/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH' \
    > /home/ubuntu/miniconda3/envs/leisaac/etc/conda/activate.d/torch_libs.sh
```

---

## Phase 2 — Isaac Lab + LeIsaac

```bash
sudo apt install cmake build-essential -y

# Clone LeIsaac (includes IsaacLab as a submodule)
git clone https://github.com/LightwheelAI/leisaac.git
cd leisaac
git submodule update --init --recursive

# Install Isaac Lab
cd dependencies/IsaacLab
./isaaclab.sh --install
cd ../..

# Fix evdev (container kernel headers are too old to build it from source;
# use system gcc which points at the updated headers)
sudo apt install -y gcc linux-libc-dev
CC=gcc CFLAGS="-I/usr/include" pip install evdev

# Install LeIsaac
pip install -e "source/leisaac[lerobot]"
pip install numpy==1.26.0
```

---

## Phase 3 — Download SO-101 assets

Run these from inside the `leisaac/` directory:

```bash
mkdir -p assets/robots assets/scenes

# SO-101 robot USD
curl -L -o assets/robots/so101_follower.usd \
    https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd

# Kitchen scene with oranges
curl -L -o kitchen.zip \
    https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/kitchen_with_orange.zip
unzip kitchen.zip -d assets/scenes/
rm kitchen.zip
```

Expected directory layout:

```
leisaac/
└── assets/
    ├── robots/
    │   └── so101_follower.usd
    └── scenes/
        └── kitchen_with_orange/
            ├── scene.usd
            └── assets/
```

---

## Phase 4 — Verify SO-101 loads in Isaac Sim

```bash
cd ~/leisaac

export LD_LIBRARY_PATH=~/miniconda3/envs/leisaac/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH

python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --teleop_device=keyboard \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras
```

You should see the SO-101 arm in a kitchen scene in your Selkies stream.
You can move it with keyboard keys (key map prints in terminal).

**This is the green light checkpoint** — if this works, the full pipeline
(Isaac Sim + LeIsaac + SO-101 assets) is confirmed working.

---

## Phase 5 — DreamZero inference server (H200 box)

### With your LoRA checkpoint (real inference)

```bash
# On the H200 box, inside the dreamzero0/dreamzero repo
conda activate dreamzero  # or whatever env you trained in

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port 5000 \
    --enable-dit-cache \
    --model-path <path/to/your/lora_checkpoint>
```

> **LoRA checkpoint note:** if `socket_test_optimized_AR.py` doesn't have a
> `--lora-path` flag, merge your LoRA weights into the base checkpoint first:
>
> ```python
> from peft import PeftModel
> base = load_base_model(...)
> model = PeftModel.from_pretrained(base, "path/to/lora")
> merged = model.merge_and_unload()
> merged.save_pretrained("path/to/merged_checkpoint")
> ```
> Then point `--model-path` at the merged checkpoint.

### With mock server (for testing without real weights)

```bash
# On any machine (no GPU needed)
pip install websockets msgpack

python mock_dreamzero_server.py --port 5000 --mode sine
```

Available modes:

| Mode | What happens |
|---|---|
| `sine` | `shoulder_pan` sweeps left/right — easy to see visually |
| `still` | All joints at zero — arm holds rest pose |
| `scripted` | Rough reach → close gripper → lift sequence |

---

## Phase 6 — Run the sim client (4090 box)

Copy the client files to your leisaac directory:

```bash
cp dreamzero_so101_client.py ~/leisaac/
cp mock_dreamzero_server.py ~/leisaac/    # only needed for mock testing
```

Install the openpi client and pin the WebSocket library version:

```bash
pip install openpi-client msgpack

# Pin websockets to the same version on BOTH machines.
# The openpi-client uses the websockets library internally.
# A version mismatch between client and server causes silent
# connection failures or garbled frames with no clear error message.
pip install "websockets==12.0"
```

> **Both machines must use the same `websockets` version.**
> Check what the DreamZero server installs (look at its `requirements.txt`
> or run `pip show websockets` on the H200 after setting up the server env)
> and pin the same version on the 4090 box. If you're using the mock server,
> both terminals are on the same machine so this is automatic.

```bash
# Quick version check — run on both machines and confirm they match
pip show websockets | grep Version
```

Run from inside `~/leisaac/` (AppLauncher needs this as the working directory):

```bash
cd ~/leisaac

export LD_LIBRARY_PATH=~/miniconda3/envs/leisaac/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH

# Against mock server on same machine
python dreamzero_so101_client.py \
    --enable_cameras \
    --host 127.0.0.1 \
    --port 5000 \
    --task "pick up the orange and place it on the plate" \
    --episodes 3

# Against real DreamZero on H200
python dreamzero_so101_client.py \
    --enable_cameras \
    --host <H200_IP> \
    --port 5000 \
    --task "pick up the orange and place it on the plate" \
    --episodes 5
```

### Client flags

| Flag | Default | Description |
|---|---|---|
| `--enable_cameras` | off | Required — enables TiledCamera rendering |
| `--host` | 127.0.0.1 | IP of the DreamZero server |
| `--port` | 5000 | Port of the DreamZero server |
| `--task` | "pick up..." | Language instruction sent to the model |
| `--episodes` | 3 | Number of rollout episodes |
| `--max-steps` | 300 | Max steps per episode before timeout |
| `--chunk-reuse` | 8 | Steps to execute per chunk before re-querying |
| `--control-hz` | 7.0 | Target control frequency |
| `--headless` | off | Run without display window |

---

## What you should see

**Mock server terminal:**
```
[server] Mock DreamZero server starting
[server] mode=sine | chunk_size=24 | port=5000
[server] Waiting for sim client...
[server] client connected: ('127.0.0.1', 54321)
[server] step=  0 | state=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | prompt='pick up...' | cams=['wrist', 'front']
[server]   → sent chunk (24, 6) in 2.1ms
```

**Client terminal:**
```
[SO101Env] ready  action_dim=6
[DreamZeroClient] connected  ws://127.0.0.1:5000
===================================================
  Episode 1/3  task: pick up the orange and place it on the plate
===================================================
  step   0  query 12ms  chunk (24, 6)
  step   8  query 11ms  chunk (24, 6)
  ...
  Episode 1: timeout in 300 steps
```

**Selkies stream:** SO-101 shoulder pan joint sweeping left and right (sine mode).

---

## Observation / Action spec

This is what the client sends to the DreamZero server and what it receives back.

> ⚠️ **These keys must match on both sides — client AND server.**
> The #1 cause of a model producing bad/random actions without any error is
> a key name mismatch between what the sim client sends and what the server
> expects. There is no runtime error — it just silently feeds wrong data to
> the model.

**Observations sent from client → server:**

| Key | Shape | Type | Description |
|---|---|---|---|
| `observation.images.wrist` | (480, 640, 3) | uint8 | Wrist camera RGB |
| `observation.images.front` | (480, 640, 3) | uint8 | Front camera RGB |
| `observation.state` | (6,) | float32 | Joint positions [rad] |
| `prompt` | string | — | Task language instruction |

State vector order: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`

**Actions received from server → client:**

| Key | Shape | Type | Description |
|---|---|---|---|
| `actions` | (24, 6) | float32 | Chunk of 24 joint position targets [rad] |

### How to verify your keys match

The ground truth is whatever keys were used when converting your SO-101 dataset
to GEAR format for LoRA training. Check two places:

**1. On the DreamZero server side** — open `test_client_AR.py` in the
`dreamzero0/dreamzero` repo and look for the dict it builds to send to the
server. The keys in that dict are exactly what the server expects:

```bash
# On the H200 box
grep "observation\." test_client_AR.py
```

**2. On the training side** — check your dataset conversion config
(`modality.json` or equivalent) for the `--state-keys` and `--image-keys`
that were passed during GEAR dataset conversion. These are the keys your
LoRA was trained on and must be used at inference time.

**If your keys differ from the defaults above**, edit `DreamZeroClient.get_action_chunk()`
in `dreamzero_so101_client.py`:

```python
wire = {
    # Change these to match your training keys exactly
    "observation.images.wrist": obs["images"]["wrist"],
    "observation.images.front": obs["images"]["front"],
    "observation.state": np.concatenate([obs["joint_pos"], obs["gripper_pos"]]),
    "prompt": self._task,
}
```

### WebSocket compatibility

The `openpi-client` library uses `websockets` internally. The client (4090 box)
and the server (H200 box) must use the **same major version** of `websockets`
or connections will fail or produce garbled frames.

```bash
# Check version on both machines — they must match
pip show websockets | grep Version

# To pin to a specific version on the client side
pip install "websockets==12.0"
```

Check the DreamZero server's `requirements.txt` for the version it expects and
match it on the client side. If you're testing with the mock server locally,
both sides share the same environment so this is handled automatically.

---

## Troubleshooting

### `libtorch_cuda_linalg.so: cannot open shared object file`
Wrong torch version or missing `LD_LIBRARY_PATH`. Fix:
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
export LD_LIBRARY_PATH=~/miniconda3/envs/leisaac/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
```

### `evdev` build fails with undeclared identifiers (`BUS_AMD_SFH` etc.)
Container kernel headers are too old. Fix:
```bash
sudo apt install -y linux-libc-dev gcc
CC=gcc CFLAGS="-I/usr/include" pip install evdev
```

### `RuntimeError: A camera was spawned without the --enable_cameras flag`
Always pass `--enable_cameras` when running the client script. The flag is
handled by `AppLauncher` and enables TiledCamera rendering.

### `TypeError: Missing values ... actions.arm_action / actions.gripper_action`
The `PickOrangeEnvCfg` requires action cfgs to be set manually. These are
already set in `SO101Env.__init__()` in the client script — make sure you're
running the latest version of the script.

### Isaac Sim crashes with `omni.syntheticdata` / `omni.graph.core` backtrace
This happens when `SimulationApp` is launched with a conflicting experience kit
file. The fix is to use `AppLauncher` (from `isaaclab.app`) instead of
`SimulationApp` directly — the current client script already does this.

### Connection fails or hangs when switching from mock to real DreamZero server
Most likely a `websockets` version mismatch between the 4090 box and the H200
box. Check and pin both to the same version:
```bash
# On both machines
pip show websockets | grep Version
# Then pin the client to match the server
pip install "websockets==<version_from_server>"
```

### Model connects but produces random / nonsensical arm motion
Observation key mismatch — the model is receiving data in the wrong fields.
Run this on the H200 to check what keys the server actually expects:
```bash
grep "observation\." test_client_AR.py
```
Then update `DreamZeroClient.get_action_chunk()` in `dreamzero_so101_client.py`
to use those exact key names. Also verify the state vector order matches your
training data.

### Mock server receives observations but arm doesn't move
Check that `LD_LIBRARY_PATH` is set and that you're running from `~/leisaac/`.
Also confirm the mock server is returning `{"actions": [[...], ...]}` — check
its terminal output.

---

## File overview

```
leisaac/
├── dreamzero_so101_client.py     Main sim client — runs on 4090 box
├── mock_dreamzero_server.py      Fake DreamZero server — for testing
├── assets/
│   ├── robots/so101_follower.usd
│   └── scenes/kitchen_with_orange/
└── dependencies/
    └── IsaacLab/
```
