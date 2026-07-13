"""
DreamZero ↔ SO-101 Isaac Sim (LeIsaac) bridge — compatible with socket_so101.py.

Confirmed from single_arm_env_cfg.py:
  - raw_obs["policy"]["wrist"]     → (1, 480, 640, 3) or (1, 480, 640, 4) uint8 GPU tensor
  - raw_obs["policy"]["front"]     → (1, 480, 640, 3) or (1, 480, 640, 4) uint8 GPU tensor
  - raw_obs["policy"]["joint_pos"] → (1, 6) float32 GPU tensor
                                      [shoulder_pan, shoulder_lift, elbow_flex,
                                       wrist_flex, wrist_roll, gripper]

Server (socket_so101.py) expects roboarena keys:
  - observation/exterior_image_0_left  → video.front   (your modality config key)
  - observation/wrist_image_left       → video.wrist   (your modality config key)
  - observation/joint_position         → state.joint_pos
  - observation/gripper_position       → state.gripper_pos
  - prompt, session_id

Usage:
    cd ~/harsha/leisaac

    python /home/ubuntu/harsha/dreamzero/dreamzero_so101_client.py \\
        --enable_cameras \\
        --host <SERVER_IP> --port 5000 \\
        --episodes 5 \\
        --task "pick up the orange and place it on the plate"
"""

import argparse
import os
import sys
import uuid

# ── AppLauncher must come before any Isaac/OmniVerse imports ──────────────────
from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DreamZero SO-101 LeIsaac eval client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",        default="127.0.0.1",
                   help="DreamZero inference server hostname or IP")
    p.add_argument("--port",        type=int, default=5000,
                   help="DreamZero inference server port")
    p.add_argument("--task",
                   default="Pick three oranges and put them into the plate, then reset the arm to rest state.",
                   help="Natural-language task instruction for the policy")
    p.add_argument("--episodes",    type=int, default=3,
                   help="Number of evaluation episodes to run")
    p.add_argument("--max-steps",   type=int, default=750,
                   help="Max environment steps per episode before timeout")
    p.add_argument("--chunk-reuse", type=int, default=8,
                   help="How many actions from each chunk to execute before re-querying")
    p.add_argument("--control-hz",  type=float, default=30.0,
                   help="Target closed-loop control frequency in Hz")

    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


args = parse_args()

if not getattr(args, "enable_cameras", False):
    print(
        "\n[ERROR] This script requires --enable_cameras for camera observations.\n"
        "        Re-run with:\n\n"
        f"        python {sys.argv[0]} --enable_cameras "
        f"--host {args.host} --port {args.port}\n"
    )
    sys.exit(1)

app_launcher   = AppLauncher(args)
simulation_app = app_launcher.app

# ── All Isaac / LeIsaac imports go AFTER app is created ───────────────────────
import time
import numpy as np
import torch


# =============================================================================
# LeIsaac SO-101 environment wrapper
# =============================================================================

class SO101Env:
    """
    Wraps LeIsaac's PickOrange environment.

    Confirmed camera/joint layout from single_arm_env_cfg.py:
        scene.wrist : TiledCameraCfg  640×480  attached to Robot/gripper/wrist_camera
        scene.front : TiledCameraCfg  640×480  attached to Robot/base/front_camera
        joint_pos   : (1, 6)  [shoulder_pan, shoulder_lift, elbow_flex,
                                wrist_flex, wrist_roll, gripper]

    get_observation(raw_obs) returns:
        {
            "joint_pos":   np.ndarray (5,)  float32  — arm joints only
            "gripper_pos": np.ndarray (1,)  float32  — gripper
            "images": {
                "front": np.ndarray (480, 640, 3) uint8 RGB
                "wrist": np.ndarray (480, 640, 3) uint8 RGB
            }
        }

    apply_action(action):
        action: np.ndarray (6,) joint position targets
                [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    """

    def __init__(self, device: str = "cuda:0", control_hz: float = 30.0) -> None:
        import gymnasium as gym
        import leisaac  # noqa: F401

        from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
        from leisaac.tasks.pick_orange.pick_orange_env_cfg import PickOrangeEnvCfg
        from leisaac.assets.robots.lerobot import SO101_FOLLOWER_MOTOR_LIMITS
        from leisaac.utils.robot_utils import (
            convert_leisaac_action_to_lerobot,
            convert_lerobot_action_to_leisaac,
        )

        self._device = device
        self._env    = None

        cfg = PickOrangeEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device     = device
        # Advance enough physics substeps for one policy step. Without this,
        # a 30 Hz action stream advances only one tiny physics tick per action.
        cfg.decimation = max(1, round(1.0 / (control_hz * cfg.sim.dt)))

        cfg.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift",
                         "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
            use_default_offset=False,
        )
        cfg.actions.gripper_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
            use_default_offset=False,
        )

        self._to_lerobot = convert_leisaac_action_to_lerobot
        self._to_leisaac = convert_lerobot_action_to_leisaac
        motor_limits = np.asarray(
            list(SO101_FOLLOWER_MOTOR_LIMITS.values()), dtype=np.float32
        )
        self._motor_low = motor_limits[:, 0]
        self._motor_high = motor_limits[:, 1]

        self._env = gym.make(
            "LeIsaac-SO101-PickOrange-v0",
            cfg=cfg,
        ).unwrapped

        print(
            f"[SO101Env] ready  "
            f"action_dim={self._env.action_space.shape[0]}  "
            f"device={device}"
        )

    def reset(self) -> dict:
        raw_obs, _ = self._env.reset()
        return raw_obs

    def get_observation(self, raw_obs: dict) -> dict:
        """
        Extract structured observations from a raw LeIsaac obs dict.

        raw_obs["policy"] contains:
            joint_pos  : (1, 6) GPU tensor — all joints incl. gripper
            wrist      : (1, 480, 640, C) GPU tensor — C=3 RGB or C=4 RGBA
            front      : (1, 480, 640, C) GPU tensor
        """
        obs = raw_obs["policy"]

        # The checkpoint was trained from a LeRobot dataset in SO-101 motor
        # coordinates (roughly -100..100), while Isaac stores radians.
        joints_rad = obs["joint_pos"].detach().cpu().numpy()  # (1, 6)
        joints = self._to_lerobot(joints_rad)[0]               # (6,)

        def _to_rgb(t: torch.Tensor) -> np.ndarray:
            """(1, H, W, C) GPU tensor → (H, W, 3) uint8 numpy RGB."""
            arr = t.cpu().numpy().squeeze(0)   # (H, W, C)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]             # RGBA → RGB
            return arr.astype(np.uint8)

        return {
            "joint_pos":   joints[:5].astype(np.float32),   # arm: (5,)
            "gripper_pos": joints[5:6].astype(np.float32),  # gripper: (1,)
            "images": {
                "front": _to_rgb(obs["front"]),   # (480, 640, 3)
                "wrist": _to_rgb(obs["wrist"]),   # (480, 640, 3)
            },
        }

    def apply_action(self, action: np.ndarray) -> tuple[dict, bool, bool]:
        """Step with one absolute target in LeRobot motor coordinates."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,):
            raise ValueError(f"Expected action shape (6,), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError(f"DreamZero returned a non-finite action: {action}")

        # Keep predictions inside the physical motor limits, then convert the
        # LeRobot training representation back to Isaac joint radians.
        action = np.clip(action, self._motor_low, self._motor_high)
        action_rad = self._to_leisaac(action[None, :])[0].astype(np.float32)
        action_t = torch.from_numpy(action_rad).unsqueeze(0).to(self._device)

        raw_obs, _reward, terminated, truncated, _info = self._env.step(action_t)
        success = bool(terminated[0])
        done = success or bool(truncated[0])
        return raw_obs, done, success

    def close(self) -> None:
        """
        Detach replicator annotators before closing to avoid the
        syntheticdata bad_variant_access crash on Carbonite shutdown.
        """
        try:
            import omni.replicator.core as rep
            rep.orchestrator.stop()
        except Exception:
            pass
        if self._env is not None:
            self._env.close()
            self._env = None


# =============================================================================
# DreamZero WebSocket policy client
# =============================================================================

class DreamZeroClient:
    """
    Connects to socket_so101.py via WebSocket and returns SO-101 action chunks.

    Wire format (roboarena keys) sent to the server:
        observation/exterior_image_0_left  (480, 640, 3) uint8  ← front cam
        observation/wrist_image_left       (480, 640, 3) uint8  ← wrist cam
        observation/joint_position         (5,)  float32        ← arm joints
        observation/gripper_position       (1,)  float32        ← gripper
        observation/cartesian_position     (6,)  float32        ← zeros (unused)
        prompt                             str
        session_id                         str

    Server converts these to:
        video.front          via IMAGE_KEY_MAP in socket_so101.py
        video.wrist
        state.joint_pos
        state.gripper_pos
        annotation.task

    Returns: np.ndarray (N, 6) — N action steps, 6 DoF each.
    """

    def __init__(self, host: str, port: int, task_instruction: str) -> None:
        from eval_utils.policy_client import WebsocketClientPolicy

        self._client     = WebsocketClientPolicy(host=host, port=port)
        self._task       = task_instruction
        self._session_id = ""

        meta = self._client.get_server_metadata()
        print(f"[DreamZeroClient] connected  ws://{host}:{port}")
        print(f"[DreamZeroClient] server metadata: {meta}")
        self._warn_if_misconfigured(meta)

    @staticmethod
    def _warn_if_misconfigured(meta: dict) -> None:
        embodiment = meta.get("embodiment", "")
        if embodiment and "so101" not in embodiment.lower():
            print(
                f"[DreamZeroClient] WARNING: server embodiment='{embodiment}', "
                "expected 'so101'. Make sure you're running socket_so101.py."
            )
        res = meta.get("image_resolution")
        if res and list(res) != [480, 640]:
            print(
                f"[DreamZeroClient] WARNING: server image_resolution={res}, "
                "but this client sends (480, 640). "
                "Update SO101_IMAGE_H/W in socket_so101.py if needed."
            )

    def new_session(self) -> None:
        """Call at the start of each episode — resets the server's frame buffer."""
        self._session_id = str(uuid.uuid4())
        print(f"[DreamZeroClient] new session: {self._session_id}")

    def get_action_chunk(self, obs: dict) -> np.ndarray:
        """
        Query the server for an action chunk.

        Parameters
        ----------
        obs : dict from SO101Env.get_observation()

        Returns
        -------
        np.ndarray (N, 6)  — joint position targets
            [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        """
        wire = {
            # roboarena image keys → server maps to video.front / video.wrist
            "observation/exterior_image_0_left": obs["images"]["front"],  # (480,640,3)
            "observation/wrist_image_left":       obs["images"]["wrist"],  # (480,640,3)

            # proprioception
            "observation/joint_position":     obs["joint_pos"],                    # (5,)
            "observation/gripper_position":   obs["gripper_pos"],                  # (1,)
            "observation/cartesian_position": np.zeros(6, dtype=np.float32),       # unused

            # language + session
            "prompt":     self._task,
            "session_id": self._session_id,
        }

        result  = self._client.infer(wire)
        actions = np.asarray(result, dtype=np.float32)

        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        assert actions.shape[-1] == 6, (
            f"Expected 6-DoF actions (5 arm + 1 gripper), got shape {actions.shape}. "
            "Check SO101_ARM_JOINTS / SO101_GRIPPER_DIM in socket_so101.py."
        )
        return actions   # (N, 6)

    def reset_episode(self) -> None:
        """Signal end of episode — triggers video save + frame buffer flush on server."""
        self._client.reset({})


# =============================================================================
# Episode rollout
# =============================================================================

def run_episode(
    sim_env: SO101Env,
    policy: DreamZeroClient,
    max_steps: int,
    chunk_reuse: int,
    control_hz: float,
) -> dict:
    period = 1.0 / control_hz

    policy.new_session()       # fresh session ID → server resets frame buffer
    raw_obs = sim_env.reset()

    step    = 0
    done    = False
    success = False

    while step < max_steps and not done:
        obs = sim_env.get_observation(raw_obs)

        t0    = time.perf_counter()
        chunk = policy.get_action_chunk(obs)   # (N, 6)
        qms   = (time.perf_counter() - t0) * 1000

        print(
            f"  step {step:3d}  query {qms:5.0f}ms  "
            f"chunk {chunk.shape}  "
            f"range [{chunk.min():.3f}, {chunk.max():.3f}]"
        )

        n_exec = min(chunk_reuse, chunk.shape[0])
        for i in range(n_exec):
            t0 = time.perf_counter()
            raw_obs, done, action_success = sim_env.apply_action(chunk[i])
            step += 1
            if done:
                success = action_success
                break
            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

        if done or step >= max_steps:
            break

    policy.reset_episode()
    return {"steps": step, "success": success}


# =============================================================================
# Isaac Sim clean shutdown
# =============================================================================

def _shutdown(sim_env) -> None:
    if sim_env is not None:
        sim_env.close()
    simulation_app.close()
    os._exit(0)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    sim_env = None
    results = []

    try:
        sim_env = SO101Env(device="cuda:0", control_hz=args.control_hz)
    except Exception:
        import traceback
        traceback.print_exc()
        _shutdown(sim_env)

    try:
        policy = DreamZeroClient(
            host=args.host,
            port=args.port,
            task_instruction=args.task,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        _shutdown(sim_env)

    try:
        for ep in range(args.episodes):
            sep = "=" * 60
            print(f"\n{sep}")
            print(f"  Episode {ep + 1}/{args.episodes}")
            print(f"  Task   : {args.task}")
            print(f"{sep}")

            result = run_episode(
                sim_env=sim_env,
                policy=policy,
                max_steps=args.max_steps,
                chunk_reuse=args.chunk_reuse,
                control_hz=args.control_hz,
            )
            results.append(result)
            status = "SUCCESS ✓" if result["success"] else "timeout ✗"
            print(f"  Episode {ep + 1}: {status} in {result['steps']} steps")

    except Exception:
        import traceback
        traceback.print_exc()

    finally:
        if results:
            n_success = sum(r["success"] for r in results)
            avg_steps = np.mean([r["steps"] for r in results])
            sep = "=" * 60
            print(f"\n{sep}")
            print(f"  Results  : {n_success}/{len(results)} success")
            print(f"  Avg steps: {avg_steps:.1f}")
            print(f"{sep}\n")

    _shutdown(sim_env)


if __name__ == "__main__":
    main()
