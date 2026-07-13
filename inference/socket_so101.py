"""
DreamZero inference server for SO-101.

Identical in structure to socket_test_optimized_AR.py but with:
  - embodiment_tag = "so101"  (matches your EmbodimentTag + modality config)
  - state dim = 6  (5 arm joints + 1 gripper)
  - action dim = 6 (same)
  - image_resolution = (480, 640)  ← SO-101 LeIsaac TiledCamera default;
    change to match YOUR actual camera resolution if different
  - n_external_cameras = 1 (SO-101 has 1 front cam + 1 wrist cam)

Launch (from the dreamzero repo root):
    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    python -m torch.distributed.run --standalone --nproc_per_node=4 \\
        socket_so101.py \\
        --model-path ./checkpoints/your_so101_checkpoint \\
        --port 5000 \\
        --enable-dit-cache

    # Single GPU (slow, for debugging):
    CUDA_VISIBLE_DEVICES=0 \\
    python -m torch.distributed.run --standalone --nproc_per_node=1 \\
        socket_so101.py \\
        --model-path ./checkpoints/your_so101_checkpoint \\
        --port 5000

Notes:
  - --enable-dit-cache speeds up inference significantly; disable only for
    debugging (it gets enabled after the first forward pass).
  - The embodiment_tag string must match exactly what you registered in
    groot/vla/data/schema.py (EmbodimentTag enum) during dataset prep.
  - The server advertises image_resolution, n_external_cameras, etc. to
    clients via PolicyServerConfig so the client can validate its setup.
"""

import asyncio
import dataclasses
import datetime
import http
import logging
import os
import socket
import sys
import time
import traceback

# This script lives in inference/, while project packages such as eval_utils/
# live at the repository root. Make those imports independent of the caller's
# current working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import imageio
import numpy as np
import torch
import torch.distributed as dist
import tyro
from einops import rearrange
from tianshou.data import Batch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from eval_utils.policy_server import PolicyServerConfig
from eval_utils.policy_server import WebsocketPolicyServer as RoboarenaServer

logger = logging.getLogger(__name__)

# ── SO-101 configuration constants ────────────────────────────────────────────
# Change these to match your actual setup.

# The string you registered in groot/vla/data/schema.py → EmbodimentTag
SO101_EMBODIMENT_TAG = "so101"

# Joint dimensions: SO-101 has 5 arm joints + 1 gripper = 6 total.
SO101_ARM_JOINTS   = 5
SO101_GRIPPER_DIM  = 1
SO101_STATE_DIM    = SO101_ARM_JOINTS + SO101_GRIPPER_DIM   # = 6
SO101_ACTION_DIM   = SO101_STATE_DIM                         # = 6

# Camera resolution that LeIsaac's TiledCamera produces.
# Must match what your training dataset used; (H, W).
SO101_IMAGE_H = 480
SO101_IMAGE_W = 640


# ── CLI args ───────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    port: int = 5000
    model_path: str = "./checkpoints/dreamzero_so101"
    enable_dit_cache: bool = False
    timeout_seconds: int = 36000   # 10 hours
    # Index suffix for the output video directory (useful when running
    # multiple eval sessions on the same day).
    index: int = 0
    # Override the action chunk size returned per infer() call.
    # None = use whatever the model config says (usually 24 steps).
    max_chunk_size: int | None = None


# ── SO-101 roboarena policy wrapper ───────────────────────────────────────────

class SO101RoboarenaPolicy:
    """
    Bridges the roboarena WebSocket interface to GrootSimPolicy for SO-101.

    The roboarena layer (RoboarenaServer / eval_utils) sends observations in a
    fixed format and expects a numpy action array back.  This wrapper:

      1. Converts incoming roboarena obs → AR_droid batch keys (video.*, state.*)
         that GrootSimPolicy / DreamTransform actually reads.
      2. Accumulates frames across calls (roboarena sends one frame per call;
         the model needs 1 frame on the first call, then 4-frame temporal windows
         on subsequent calls — identical to the DROID server logic).
      3. Runs the distributed forward pass.
      4. Converts the model's action dict back to a (N, SO101_ACTION_DIM) array.

    Roboarena obs keys expected (as advertised via PolicyServerConfig):
        observation/exterior_image_0_left   (H, W, 3) uint8 — front camera
        observation/wrist_image_left        (H, W, 3) uint8 — wrist camera
        observation/joint_position          (5,)  float32    — arm joints
        observation/gripper_position        (1,)  float32    — gripper
        prompt                              str
        session_id                          str

    AR_droid batch keys produced (consumed by GrootSimPolicy):
        video.front                         (T, H, W, 3)
        video.wrist              (T, H, W, 3)
        video.top                           (T, H, W, 3) — front-view alias
        state.joint_pos                     (1, 5)
        state.gripper_pos                   (1, 1)
        annotation.task                     str

    LeIsaac provides one front camera and one wrist camera. The checkpoint
    also expects ``video.top``, so the front frames are reused for that
    modality during observation conversion.
    """

    FRAMES_PER_CHUNK = 4   # temporal window after the first call

    # Roboarena key → AR_droid video key
    IMAGE_KEY_MAP = {
        "observation/exterior_image_0_left": "video.front",
        "observation/wrist_image_left":       "video.wrist",
    }

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
    ) -> None:
        self._policy      = groot_policy
        self._signal_group = signal_group
        self._output_dir  = output_dir

        # Per-camera frame ring buffers.
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            v: [] for v in self.IMAGE_KEY_MAP.values()
        }
        self._is_first_call = True
        self._call_count    = 0
        self._current_session_id: str | None = None
        self.video_across_time: list = []

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    # ── observation conversion ─────────────────────────────────────────────

    def _convert_observation(self, obs: dict) -> dict:
        """Roboarena obs dict → AR_droid obs dict."""
        converted: dict = {}

        # ── cameras ───────────────────────────────────────────────────────
        for ra_key, dz_key in self.IMAGE_KEY_MAP.items():
            if ra_key not in obs:
                continue
            data = obs[ra_key]
            if not isinstance(data, np.ndarray):
                continue
            if data.ndim == 4:
                # Already (T, H, W, 3) — extend buffer with all T frames.
                self._frame_buffers[dz_key].extend(list(data))
            else:
                # Single (H, W, 3) frame.
                self._frame_buffers[dz_key].append(data)

        num_frames = 1 if self._is_first_call else self.FRAMES_PER_CHUNK

        for dz_key, buf in self._frame_buffers.items():
            if not buf:
                # Camera missing — fill with zeros so the model still runs.
                converted[dz_key] = np.zeros(
                    (num_frames, SO101_IMAGE_H, SO101_IMAGE_W, 3), dtype=np.uint8
                )
                continue
            if len(buf) >= num_frames:
                frames = buf[-num_frames:]
            else:
                # Pad by repeating the earliest available frame.
                pad = [buf[0]] * (num_frames - len(buf))
                frames = pad + list(buf)
            converted[dz_key] = np.stack(frames, axis=0)   # (T, H, W, 3)

        # This LeIsaac environment has only one external camera, while the
        # checkpoint was trained with both front and top external modalities.
        # Reuse the front view so the checkpoint receives every required key.
        converted["video.top"] = converted["video.front"].copy()

        # ── proprioception ─────────────────────────────────────────────────
        # joint_position: expect (5,) → reshape to (1, 5)
        if "observation/joint_position" in obs:
            jp = np.asarray(obs["observation/joint_position"], dtype=np.float64)
            converted["state.joint_pos"] = jp.reshape(1, -1)
        else:
            converted["state.joint_pos"] = np.zeros(
                (1, SO101_ARM_JOINTS), dtype=np.float64
            )

        # gripper_position: expect (1,) → reshape to (1, 1)
        if "observation/gripper_position" in obs:
            gp = np.asarray(obs["observation/gripper_position"], dtype=np.float64)
            converted["state.gripper_pos"] = gp.reshape(1, -1)
        else:
            converted["state.gripper_pos"] = np.zeros(
                (1, SO101_GRIPPER_DIM), dtype=np.float64
            )

        # ── language ───────────────────────────────────────────────────────
        # This is the language key used by the trained SO-101 checkpoint.
        converted["annotation.task"] = obs.get("prompt", "")

        return converted

    # ── action conversion ──────────────────────────────────────────────────

    def _convert_action(self, action_batch: Batch) -> np.ndarray:
        """
        Extract action arrays from the model output Batch and concatenate
        into a (N, SO101_ACTION_DIM) float32 array.

        The model outputs keys like:
            action.joint_pos     (N, 5)
            action.gripper_pos   (N, 1)
        (exact names come from your modality config — adjust if yours differ).
        """
        joint_action   = None
        gripper_action = None

        # Dotted modality names are mapping keys in a Tianshou Batch; they are
        # not reliably exposed by dir()/getattr().
        for k, v in action_batch.items():
            if not str(k).startswith("action."):
                continue
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if "gripper" in k:
                gripper_action = v
            elif "joint" in k or "pos" in k:
                joint_action = v

        # Missing actions are a protocol error; silently commanding zeros
        # hides model or modality configuration problems.
        if joint_action is None:
            raise RuntimeError(
                f"No joint action found in model output. Available keys: {list(action_batch.keys())}"
            )

        joint_action = np.asarray(joint_action)
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        if joint_action.shape[-1] != SO101_ARM_JOINTS:
            raise ValueError(
                f"Expected {SO101_ARM_JOINTS} arm actions, got {joint_action.shape}"
            )
        N = joint_action.shape[0]

        if gripper_action is None:
            raise RuntimeError(
                f"No gripper action found in model output. Available keys: {list(action_batch.keys())}"
            )
        gripper_action = np.asarray(gripper_action)
        if gripper_action.ndim == 1:
            gripper_action = gripper_action.reshape(-1, 1)
        if gripper_action.shape != (N, SO101_GRIPPER_DIM):
            raise ValueError(
                f"Expected gripper actions with shape ({N}, {SO101_GRIPPER_DIM}), "
                f"got {gripper_action.shape}"
            )

        action = np.concatenate(
            [joint_action, gripper_action], axis=-1
        ).astype(np.float32)   # (N, 6)

        return action

    # ── distributed helpers ────────────────────────────────────────────────

    def _broadcast_to_workers(self, obs: dict) -> None:
        import pickle
        blob  = pickle.dumps(obs)
        size  = torch.tensor([len(blob)], dtype=torch.int64, device="cuda")
        dist.broadcast(size, src=0)
        data  = torch.frombuffer(blob, dtype=torch.uint8).cuda()
        dist.broadcast(data, src=0)

    # ── main infer entry point ─────────────────────────────────────────────

    def infer(self, obs: dict) -> np.ndarray:
        # ── session tracking — reset buffers when a new session starts ────
        session_id = obs.get("session_id")
        if session_id and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info(
                    "Session changed %s → %s, resetting frame buffers",
                    self._current_session_id, session_id,
                )
                self._reset_state(save_video=True)
            else:
                logger.info("New session: %s", session_id)
            self._current_session_id = session_id

        self._call_count += 1

        # Convert to AR_droid format.
        converted_obs = self._convert_observation(obs)

        # Signal worker ranks to participate (0 = run forward pass).
        signal = torch.zeros(1, dtype=torch.int32, device="cpu")
        dist.broadcast(signal, src=0, group=self._signal_group)

        # Broadcast converted obs to all ranks.
        self._broadcast_to_workers(converted_obs)

        batch = Batch(obs=converted_obs)

        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()

        self.video_across_time.append(video_pred)

        action = self._convert_action(result_batch.act)

        if self._is_first_call:
            self._is_first_call = False

        return action   # (N, 6)

    # ── reset ──────────────────────────────────────────────────────────────

    def _save_video(self) -> None:
        if not self.video_across_time or not self._output_dir:
            return
        try:
            cat    = torch.cat(self.video_across_time, dim=2)
            frames = self._policy.trained_model.action_head.vae.decode(
                cat,
                tiled=self._policy.trained_model.action_head.tiled,
                tile_size=(
                    self._policy.trained_model.action_head.tile_size_height,
                    self._policy.trained_model.action_head.tile_size_width,
                ),
                tile_stride=(
                    self._policy.trained_model.action_head.tile_stride_height,
                    self._policy.trained_model.action_head.tile_stride_width,
                ),
            )
            frames = rearrange(frames, "B C T H W -> B T H W C")[0]
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
            ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path   = os.path.join(self._output_dir, f"rollout_{ts}.mp4")
            imageio.mimsave(path, list(frames), fps=5, codec="libx264")
            logger.info("Saved rollout video: %s", path)
        except Exception as exc:
            logger.warning("Video save failed: %s", exc)

    def _reset_state(self, save_video: bool = True) -> None:
        if save_video:
            self._save_video()
        for buf in self._frame_buffers.values():
            buf.clear()
        self._call_count    = 0
        self._is_first_call = True
        self.video_across_time = []

    def reset(self, reset_info: dict) -> None:
        """Called by the roboarena server at the end of each episode."""
        self._reset_state(save_video=True)
        self._current_session_id = None


# ── distributed helpers (worker side) ─────────────────────────────────────────

def _worker_receive_and_forward(policy: GrootSimPolicy) -> None:
    """
    Non-rank-0 worker loop.
    Waits for a broadcast from rank 0, participates in the forward pass,
    then loops back to wait for the next request.
    """
    import pickle
    signal = torch.zeros(1, dtype=torch.int32, device="cpu")

    # We need the signal_group reference; it's created before this function
    # is called in main(), so we look it up from the global below.
    signal_group = _WORKER_SIGNAL_GROUP

    while True:
        dist.broadcast(signal, src=0, group=signal_group)
        sig = signal.item()

        if sig == 1:
            logger.info("Rank %d: shutdown signal received.", dist.get_rank())
            break
        if sig == 2:
            # Idle: no client connected, keep waiting.
            continue

        # Receive obs broadcast.
        size_t = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(size_t, src=0)
        data_t = torch.zeros(size_t.item(), dtype=torch.uint8, device="cuda")
        dist.broadcast(data_t, src=0)
        obs = pickle.loads(data_t.cpu().numpy().tobytes())

        batch = Batch(obs=obs)
        dist.barrier()
        with torch.no_grad():
            policy.lazy_joint_forward_causal(batch)
        dist.barrier()


# Module-level holder so the worker loop can access it without threading.
_WORKER_SIGNAL_GROUP = None


# ── distributed init ───────────────────────────────────────────────────────────

def init_mesh() -> DeviceMesh:
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    logger.info("Rank %d/%d on cuda:%d", rank, world_size, rank)
    return init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("ip",),
    )


def _health_check(
    conn: _server.ServerConnection, req: _server.Request
) -> _server.Response | None:
    if req.path == "/healthz":
        return conn.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ── main ───────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    global _WORKER_SIGNAL_GROUP

    os.environ["ENABLE_DIT_CACHE"]    = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"]   = "TE"
    torch._dynamo.config.recompile_limit = 800

    device_mesh = init_mesh()
    rank        = dist.get_rank()

    timeout     = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout)
    _WORKER_SIGNAL_GROUP = signal_group
    logger.info("Rank %d: signal group (gloo) ready.", rank)

    # Load the fine-tuned SO-101 model.
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(SO101_EMBODIMENT_TAG),
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )
    logger.info("Rank %d: GrootSimPolicy loaded (embodiment=%s).", rank, SO101_EMBODIMENT_TAG)

    if args.max_chunk_size is not None:
        # Override max chunk size if requested (e.g. for latency experiments).
        try:
            policy.trained_model.action_head.max_chunk_size = args.max_chunk_size
            logger.info("Overriding max_chunk_size → %d", args.max_chunk_size)
        except AttributeError:
            logger.warning("Could not override max_chunk_size — attribute not found.")

    if rank == 0:
        hostname   = socket.gethostname()
        local_ip   = socket.gethostbyname(hostname)
        date_str   = datetime.datetime.now().strftime("%Y%m%d")
        ckpt_name  = os.path.basename(args.model_path)
        output_dir = os.path.join(
            os.path.dirname(args.model_path),
            f"so101_eval_{date_str}_{args.index}",
            ckpt_name,
        )
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Server: %s (%s)  port=%d", hostname, local_ip, args.port)
        logger.info("Rollout videos → %s", output_dir)

        wrapper = SO101RoboarenaPolicy(
            groot_policy=policy,
            signal_group=signal_group,
            output_dir=output_dir,
        )

        # Advertise SO-101 capabilities to connecting clients.
        server_config = PolicyServerConfig(
            image_resolution=(SO101_IMAGE_H, SO101_IMAGE_W),
            needs_wrist_camera=True,
            n_external_cameras=1,        # 1 front/exterior cam
            needs_stereo_camera=False,
            needs_session_id=True,
            action_space="joint_position",
        )

        logger.info("PolicyServerConfig: %s", server_config)

        roboarena_server = RoboarenaServer(
            policy=wrapper,
            server_config=server_config,
            host="0.0.0.0",
            port=args.port,
        )
        roboarena_server.serve_forever()

    else:
        logger.info("Rank %d: entering worker loop.", rank)
        _worker_receive_and_forward(policy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)
