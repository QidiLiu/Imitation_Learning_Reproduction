from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .cameras import backproject, camera_fovy, camera_pose

ASSET_DIR = Path(__file__).parent / "assets"
DEFAULT_XML = ASSET_DIR / "push_arm.xml"

STATE_DIM = 18
ACTION_DIM = 2
ARM_JOINTS = ("joint1", "joint2")
BASE_XY = np.array([-0.38, 0.0], dtype=np.float64)
LINK_LENGTHS = (0.28, 0.38)

STATE_SLICES = {
    "q": slice(0, 2),
    "dq": slice(2, 4),
    "ee": slice(4, 6),
    "puck": slice(6, 8),
    "puck_vel": slice(8, 10),
    "goal": slice(10, 12),
    "last_action": slice(12, 14),
    "puck_minus_goal": slice(14, 16),
    "ee_minus_puck": slice(16, 18),
}


@dataclass
class PushArmConfig:
    xml_path: str = str(DEFAULT_XML)
    obs_mode: str = "full"            # state | image | point_cloud | full
    image_size: int = 96
    n_points: int = 512
    frame_skip: int = 25              # 25 * 0.002 s = 0.05 s = 20 Hz
    episode_steps: int = 60
    action_scale: float = 0.06        # rad per unit action
    success_threshold: float = 0.05   # metres
    terminate_on_success: bool = True
    camera: str = "top"
    visual_variant: str = "default"   # default | color | light | shift | distractors
    seed: int | None = None
    max_episode_steps: int | None = None  # alias for episode_steps


class PushArmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: PushArmConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg or PushArmConfig()
        if self.cfg.max_episode_steps is not None:
            self.cfg.episode_steps = self.cfg.max_episode_steps
        if self.cfg.obs_mode not in {"state", "image", "point_cloud", "full"}:
            raise ValueError(f"unknown obs_mode {self.cfg.obs_mode!r}")
        if self.cfg.visual_variant not in {"default", "color", "light", "shift", "distractors"}:
            raise ValueError(f"unknown visual_variant {self.cfg.visual_variant!r}")
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002

        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in ARM_JOINTS]
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.dadr = self.model.jnt_dofadr[self.joint_ids]
        self.puck_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "puck_free")
        self.puck_qadr = self.model.jnt_qposadr[self.puck_joint]
        self.puck_dadr = self.model.jnt_dofadr[self.puck_joint]
        self.ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self.goal_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self.distractor_bodies = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"distractor{i}")
            for i in range(1, 4)
        ]
        self.distractor_geoms = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"distractor{i}_geom")
            for i in range(1, 4)
        ]

        self.q_min = self.model.jnt_range[self.joint_ids, 0].copy()
        self.q_max = self.model.jnt_range[self.joint_ids, 1].copy()

        # Default visual state, restored on every reset before applying a variant.
        self._default_mat_rgba = self.model.mat_rgba.copy()
        self._default_light_diffuse = self.model.light_diffuse.copy()
        self._default_light_dir = self.model.light_dir.copy()
        self._default_distractor_pos = self.model.body_pos[self.distractor_bodies].copy()
        self._default_distractor_rgba = self.model.geom_rgba[self.distractor_geoms].copy()
        self._default_camera = self.cfg.camera
        self.camera_name = self.cfg.camera

        self.renderer: mujoco.Renderer | None = None
        self._last_rgb: np.ndarray | None = None
        self._last_depth: np.ndarray | None = None
        self._depth = None

        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)

        obs_spaces = {
            "state": spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
        }
        if self.cfg.obs_mode in {"image", "full"}:
            obs_spaces["image"] = spaces.Box(
                0, 255, (self.cfg.image_size, self.cfg.image_size, 3), np.uint8
            )
        if self.cfg.obs_mode in {"point_cloud", "full"}:
            obs_spaces["point_cloud"] = spaces.Box(
                -np.inf, np.inf, (self.cfg.n_points, 3), np.float32
            )
        self.observation_space = spaces.Dict(obs_spaces)

        self.t = 0
        self.success = False
        self.min_dist = np.inf
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._goal = np.zeros(2, dtype=np.float64)

        if self.cfg.seed is not None:
            self.reset(seed=self.cfg.seed)

    def _sample_task(self) -> tuple[np.ndarray, np.ndarray]:
        rng = self.np_random
        for _ in range(200):
            r = rng.uniform(0.26, 0.46)
            theta = rng.uniform(-0.8, 0.8)
            puck = BASE_XY + np.array([r * np.cos(theta), r * np.sin(theta)])

            gr = r + rng.uniform(0.13, 0.25)
            gtheta = theta + rng.uniform(-0.25, 0.25)
            goal = BASE_XY + np.array([gr * np.cos(gtheta), gr * np.sin(gtheta)])

            in_table = (
                -0.55 < puck[0] < 0.55 and -0.42 < puck[1] < 0.42
                and -0.55 < goal[0] < 0.55 and -0.42 < goal[1] < 0.42
            )
            reachable = np.linalg.norm(goal - BASE_XY) < 0.62
            if in_table and reachable and np.linalg.norm(goal - puck) > 0.12:
                return puck, goal

        # Deterministic fallback; never silently return an invalid task.
        return BASE_XY + np.array([0.35, 0.0]), BASE_XY + np.array([0.55, 0.0])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.success = False
        self.last_action[:] = 0.0

        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        puck, goal = self._sample_task()
        self._goal = goal
        self._set_puck(puck)
        self._set_goal(goal)
        self._reset_arm_behind(puck, goal)
        self._apply_visual_variant()

        mujoco.mj_forward(self.model, self.data)
        self.min_dist = float(np.linalg.norm(self._puck_xy() - goal))

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _set_puck(self, xy: np.ndarray) -> None:
        self.data.qpos[self.puck_qadr : self.puck_qadr + 3] = [xy[0], xy[1], 0.025]
        self.data.qpos[self.puck_qadr + 3 : self.puck_qadr + 7] = [1.0, 0.0, 0.0, 0.0]

    def _set_goal(self, xy: np.ndarray) -> None:
        self.model.body_pos[self.goal_body, :2] = xy
        self._goal = np.asarray(xy, dtype=np.float64).copy()

    def _reset_arm_behind(self, puck: np.ndarray, goal: np.ndarray) -> None:
        u = goal - puck
        u = u / (np.linalg.norm(u) + 1e-9)
        behind = puck - u * 0.07
        q = ik2(behind, BASE_XY, *LINK_LENGTHS, elbow=-1)
        q = np.clip(q, self.q_min, self.q_max)
        self.data.qpos[self.qadr] = q
        self.data.ctrl[:] = q

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError(f"non-finite action: {action}")
        action = np.clip(action, -1.0, 1.0)

        prev_dist = float(np.linalg.norm(self._puck_xy() - self._goal))
        q = self.data.qpos[self.qadr].copy()
        q_target = np.clip(q + self.cfg.action_scale * action, self.q_min, self.q_max)
        self.data.ctrl[:] = q_target

        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.t += 1
        self.last_action[:] = action
        dist = float(np.linalg.norm(self._puck_xy() - self._goal))
        self.min_dist = min(self.min_dist, dist)
        newly_success = dist < self.cfg.success_threshold and not self.success
        self.success = self.success or (dist < self.cfg.success_threshold)

        reward = (
            -dist
            + 5.0 * float(newly_success)
            - 0.01 * float(np.sum(action ** 2))
        )
        terminated = bool(self.success and self.cfg.terminate_on_success)
        truncated = bool(self.t >= self.cfg.episode_steps and not terminated)

        obs = self._get_obs()
        info = self._get_info()
        info["reward_progress"] = prev_dist - dist
        return obs, reward, terminated, truncated, info

    def _puck_xy(self) -> np.ndarray:
        return self.data.qpos[self.puck_qadr : self.puck_qadr + 2].copy()

    def _puck_vel(self) -> np.ndarray:
        return self.data.qvel[self.puck_dadr : self.puck_dadr + 2].copy()

    def _ee_xy(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site, :2].copy()

    def _state(self) -> np.ndarray:
        q = self.data.qpos[self.qadr].copy()
        dq = self.data.qvel[self.dadr].copy()
        ee = self._ee_xy()
        puck = self._puck_xy()
        pv = self._puck_vel()
        goal = self._goal.copy()
        return np.concatenate(
            [q, dq, ee, puck, pv, goal, self.last_action, puck - goal, ee - puck]
        ).astype(np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = {"state": self._state()}
        if self.cfg.obs_mode in {"image", "full", "point_cloud"}:
            rgb, depth = self._render_rgb_depth()
            if self.cfg.obs_mode in {"image", "full"}:
                obs["image"] = rgb
            if self.cfg.obs_mode in {"point_cloud", "full"}:
                obs["point_cloud"] = self._cloud_from_depth(depth)
        return obs

    def _get_info(self) -> dict:
        puck = self._puck_xy()
        goal = self._goal.copy()
        return {
            "success": bool(self.success),
            "dist": float(np.linalg.norm(puck - goal)),
            "min_dist": float(self.min_dist),
            "puck": puck,
            "goal": goal,
            "ee": self._ee_xy(),
            "qpos": self.data.qpos[self.qadr].copy(),
            "step": self.t,
        }
    def _ensure_renderer(self) -> mujoco.Renderer:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.model, self.cfg.image_size, self.cfg.image_size
            )
        return self.renderer

    def _render_rgb_depth(self) -> tuple[np.ndarray, np.ndarray]:
        renderer = self._ensure_renderer()
        camera = self._camera_name()

        renderer.disable_depth_rendering()
        renderer.update_scene(self.data, camera=camera)
        rgb = renderer.render()

        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=camera)
        depth = renderer.render()
        renderer.disable_depth_rendering()

        self._last_rgb = rgb
        self._last_depth = depth
        return rgb, depth

    def _cloud_from_depth(self, depth: np.ndarray) -> np.ndarray:
        camera = self._camera_name()
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        fovy = camera_fovy(self.model, cam_id)
        pos, mat = camera_pose(self.data, cam_id)
        points = backproject(depth, pos, mat, fovy, self.cfg.image_size, self.cfg.image_size)
        points = points.reshape(-1, 3)

        # Keep the workspace and separate object points from the table.
        keep = (
            (points[:, 0] > -0.70) & (points[:, 0] < 0.70)
            & (points[:, 1] > -0.50) & (points[:, 1] < 0.50)
            & (points[:, 2] > -0.01) & (points[:, 2] < 0.25)
        )
        points = points[keep]
        if len(points) == 0:
            return np.zeros((self.cfg.n_points, 3), dtype=np.float32)

        object_mask = points[:, 2] > 0.004
        objects = points[object_mask]
        table = points[~object_mask]
        rng = self.np_random
        n_obj = min(int(0.8 * self.cfg.n_points), len(objects))
        n_table = self.cfg.n_points - n_obj
        if n_table > len(table):
            n_obj = min(self.cfg.n_points, len(objects))
            n_table = self.cfg.n_points - n_obj

        parts = []
        if n_obj > 0:
            idx = rng.choice(len(objects), size=n_obj, replace=False)
            parts.append(objects[idx])
        if n_table > 0 and len(table) > 0:
            idx = rng.choice(len(table), size=min(n_table, len(table)), replace=False)
            parts.append(table[idx])
        cloud = np.concatenate(parts, axis=0)

        if len(cloud) < self.cfg.n_points:
            extra = rng.choice(len(cloud), size=self.cfg.n_points - len(cloud), replace=True)
            cloud = np.concatenate([cloud, cloud[extra]], axis=0)
        elif len(cloud) > self.cfg.n_points:
            cloud = cloud[: self.cfg.n_points]

        return cloud.astype(np.float32)

    def render(self):
        rgb, _ = self._render_rgb_depth()
        return rgb

    def _camera_name(self) -> str:
        return self.camera_name

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _apply_visual_variant(self) -> None:
        self.model.mat_rgba[:] = self._default_mat_rgba
        self.model.light_diffuse[:] = self._default_light_diffuse
        self.model.light_dir[:] = self._default_light_dir
        self.model.body_pos[self.distractor_bodies] = self._default_distractor_pos
        self.model.geom_rgba[self.distractor_geoms] = self._default_distractor_rgba
        self.camera_name = self.cfg.camera

        variant = self.cfg.visual_variant
        rng = self.np_random
        mat = {
            "table": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "table"),
            "arm": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "arm"),
            "tool": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "tool"),
            "puck": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "puck"),
        }

        if variant == "color":
            self.model.mat_rgba[mat["table"]] = rng.uniform(0.1, 0.9, 4)
            self.model.mat_rgba[mat["arm"]] = rng.uniform(0.1, 0.9, 4)
            self.model.mat_rgba[mat["puck"]] = rng.uniform(0.1, 0.9, 4)
            self.model.mat_rgba[mat["table"], 3] = 1.0
            self.model.mat_rgba[mat["arm"], 3] = 1.0
            self.model.mat_rgba[mat["puck"], 3] = 1.0
        elif variant == "light":
            self.model.light_diffuse[:] = rng.uniform(0.1, 0.9, self.model.light_diffuse.shape)
            self.model.light_dir[:] = rng.normal(0.0, 0.3, self.model.light_dir.shape)
        elif variant == "shift":
            self.camera_name = "top_shift"
        elif variant == "distractors":
            for body, geom in zip(self.distractor_bodies, self.distractor_geoms):
                xy = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.30, 0.30)])
                self.model.body_pos[body, :2] = xy
                self.model.body_pos[body, 2] = 0.03
                self.model.geom_rgba[geom] = np.concatenate(
                    [rng.uniform(0.1, 0.9, 3), [1.0]]
                )



def ik2(target_xy, base_xy, l1, l2, elbow=-1):
    """Analytic IK for a planar 2-link arm. elbow=-1: clockwise, +1: counter-clockwise."""
    d = np.asarray(target_xy, dtype=np.float64) - np.asarray(base_xy, dtype=np.float64)
    dist = float(np.linalg.norm(d))
    dist = float(np.clip(dist, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4))
    cos_q2 = (dist * dist - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    q2 = elbow * float(np.arccos(np.clip(cos_q2, -1.0, 1.0)))
    q1 = float(np.arctan2(d[1], d[0]) - np.arctan2(
        l2 * np.sin(q2), l1 + l2 * np.cos(q2)
    ))
    return np.array([q1, q2], dtype=np.float64)

