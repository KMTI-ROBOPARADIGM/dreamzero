#!/usr/bin/env python3
"""
Standalone test client for socket_so101.py — no Isaac Sim required.

Sends synthetic (zero or random) observations in the correct roboarena
format and validates that the server returns properly shaped actions.

Useful for:
  - Confirming the server loaded your checkpoint without errors.
  - Measuring inference latency before setting up LeIsaac.
  - CI smoke tests.

Usage
-----
    # Start the server first (on the GPU machine):
    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    python -m torch.distributed.run --standalone --nproc_per_node=4 \\
        socket_so101.py --port 5000 --model-path ./checkpoints/so101

    # Then run this test (can be on the same or a different machine):
    python test_client_so101.py --host <SERVER_IP> --port 5000 --steps 10

    # Use random images instead of zeros (better for detecting channel issues):
    python test_client_so101.py --host <SERVER_IP> --port 5000 --random-images
"""

import argparse
import logging
import time
import uuid

import numpy as np

# Uses the same client library as dreamzero_so101_client.py.
from eval_utils.policy_client import WebsocketClientPolicy

# ── SO-101 constants — must match socket_so101.py ─────────────────────────────
IMAGE_H         = 480
IMAGE_W         = 640
ARM_JOINT_DIM   = 5
GRIPPER_DIM     = 1
STATE_DIM       = ARM_JOINT_DIM + GRIPPER_DIM   # 6
EXPECTED_ACTION_DIM = STATE_DIM                  # 6


def make_observation(
    session_id: str,
    task: str,
    random_images: bool = False,
) -> dict:
    """
    Build a single synthetic observation in the roboarena format that
    socket_so101.py's _convert_observation() expects.
    """
    if random_images:
        front = np.random.randint(0, 256, (IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        wrist = np.random.randint(0, 256, (IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
    else:
        front = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        wrist = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)

    return {
        "observation/exterior_image_0_left": front,
        "observation/wrist_image_left":       wrist,
        "observation/joint_position":        np.zeros(ARM_JOINT_DIM, dtype=np.float32),
        "observation/gripper_position":      np.zeros(GRIPPER_DIM,   dtype=np.float32),
        "observation/cartesian_position":    np.zeros(6,              dtype=np.float32),
        "prompt":      task,
        "session_id":  session_id,
    }


def validate_action(actions: np.ndarray, step: int) -> None:
    """Assert that the returned action array has the expected shape and dtype."""
    assert isinstance(actions, np.ndarray), (
        f"Step {step}: expected np.ndarray, got {type(actions)}"
    )
    assert actions.ndim == 2, (
        f"Step {step}: expected 2-D action array, got shape {actions.shape}"
    )
    assert actions.shape[-1] == EXPECTED_ACTION_DIM, (
        f"Step {step}: expected last dim = {EXPECTED_ACTION_DIM} "
        f"(5 arm + 1 gripper), got {actions.shape[-1]}. "
        "Check that SO101_ARM_JOINTS / SO101_GRIPPER_DIM match in socket_so101.py "
        "and that your modality config uses the same action dimensions."
    )


def run_test(
    host: str,
    port: int,
    num_steps: int,
    task: str,
    random_images: bool,
) -> None:
    logging.info("Connecting to %s:%d ...", host, port)
    client = WebsocketClientPolicy(host=host, port=port)

    # ── validate server metadata ───────────────────────────────────────────
    meta = client.get_server_metadata()
    logging.info("Server metadata: %s", meta)

    embodiment = meta.get("embodiment", "")
    if embodiment and "so101" not in embodiment.lower():
        logging.warning(
            "Server reports embodiment='%s'; expected 'so101'. "
            "Are you pointing at the right server?", embodiment
        )

    n_cams = meta.get("n_external_cameras", -1)
    if n_cams != 1:
        logging.warning(
            "Server reports n_external_cameras=%d; SO-101 should report 1.", n_cams
        )

    res = meta.get("image_resolution")
    if res and tuple(res) != (IMAGE_H, IMAGE_W):
        logging.warning(
            "Server image_resolution=%s but this client sends (%d, %d). "
            "Update IMAGE_H / IMAGE_W in this script to match.", res, IMAGE_H, IMAGE_W
        )

    # ── run inference steps ────────────────────────────────────────────────
    session_id = str(uuid.uuid4())
    logging.info("Session ID: %s", session_id)
    logging.info(
        "Sending %d step(s) with %s images (%dx%d) ...",
        num_steps,
        "random" if random_images else "zero",
        IMAGE_H, IMAGE_W,
    )

    latencies: list[float] = []

    for step in range(num_steps):
        obs = make_observation(session_id, task, random_images)

        t0      = time.perf_counter()
        actions = client.infer(obs)
        dt      = time.perf_counter() - t0

        actions = np.asarray(actions, dtype=np.float32)
        validate_action(actions, step)

        latencies.append(dt)
        logging.info(
            "Step %2d/%d  shape=%s  range=[%.4f, %.4f]  latency=%.2fs",
            step + 1, num_steps,
            actions.shape,
            actions.min(), actions.max(),
            dt,
        )

    # ── send reset (flushes frame buffer + saves rollout video on server) ──
    logging.info("Sending reset ...")
    client.reset({})

    # ── summary ───────────────────────────────────────────────────────────
    if latencies:
        logging.info(
            "Latency — mean: %.2fs  min: %.2fs  max: %.2fs",
            np.mean(latencies), np.min(latencies), np.max(latencies),
        )

    logging.info("Test passed ✓")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for socket_so101.py — no Isaac Sim needed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host",          default="localhost",
                        help="Server hostname or IP")
    parser.add_argument("--port",          type=int, default=5000,
                        help="Server port")
    parser.add_argument("--steps",         type=int, default=10,
                        help="Number of infer() calls to send")
    parser.add_argument("--task",
                        default="pick up the orange and place it on the plate",
                        help="Language prompt to send with each observation")
    parser.add_argument("--random-images", action="store_true",
                        help="Send random uint8 images instead of zeros "
                             "(better for detecting silent channel mismatches)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run_test(
        host=args.host,
        port=args.port,
        num_steps=args.steps,
        task=args.task,
        random_images=args.random_images,
    )


if __name__ == "__main__":
    main()
