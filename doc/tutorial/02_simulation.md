# 02 · The MuJoCo Task: A 2-DoF Planar Pusher

> **Goal.** Build a deterministic, Gymnasium-compatible MuJoCo environment that
> exposes proprioception, a top-down RGB image, and a world-frame point cloud.
> Understand the robot model, the action mapping, the success criterion, and the
> pinhole-camera math that turns depth into 3-D points.
>
> **Prerequisites.** Chapter 01.
>
> **Time.** 1–2 days.
>
> **Artifacts.** `envs/assets/push_arm.xml`, `envs/push_arm.py`,
> `envs/cameras.py`, `envs/wrappers.py`, `tests/test_env.py`,
> `tests/test_cameras.py`.

---

## 1. Task specification

The robot is a planar two-link arm mounted at the left edge of a table. A paddle
at the end of the second link pushes an orange puck toward a green goal marker.
The arm moves in the horizontal plane; gravity holds the puck on the table.

```text
top view (not to scale)

                  goal (green post)
                        ▲
                        │
        puck (orange) ●─┘        push direction u
                     ▲
                    ╱
        paddle ────╱
           link2 ╱
                ╱
   base ──────● link1
```

| Quantity | Value |
|---|---|
| Arm links | 0.28 m, 0.38 m |
| Paddle | 0.09 m wide, 0.06 m tall |
| Puck | radius 0.035 m, height 0.05 m |
| Goal region | radius 0.06 m; success if puck centre within 0.05 m |
| Control rate | 20 Hz (`frame_skip=25`, `timestep=0.002`) |
| Episode | 60 control steps (3 s), terminates early on success |
| Action | 2-D normalized delta joint target `a ∈ [-1,1]²` |
| State | 18-D (see below) |
| Image | 96×96 RGB, top-down camera |
| Point cloud | 512 points in world coordinates |

### State layout

Use a single explicit layout everywhere. Do not let different modules disagree
about the order.

| Slice | Name | Dimension | Units |
|---|---|---|---|
| `0:2` | joint positions `q` | 2 | rad |
| `2:4` | joint velocities `dq` | 2 | rad/s |
| `4:6` | end-effector `xy` | 2 | m |
| `6:8` | puck `xy` | 2 | m |
| `8:10` | puck velocity `xy` | 2 | m/s |
| `10:12` | goal `xy` | 2 | m |
| `12:14` | previous action | 2 | normalized |
| `14:16` | `puck - goal` | 2 | m |
| `16:18` | `ee - puck` | 2 | m |

The last four dimensions are redundant but make the learning problem easier and
make hand-designed features in chapters 7–8 readable. Redundancy is fine in a
small state; it is not fine to silently change the layout later.

### Action mapping

The policy outputs `a ∈ [-1,1]²`. The environment maps it to a joint target:

```text
q_target = clip(q + action_scale * a, q_min, q_max)
```

with `action_scale = 0.06 rad`. Position actuators track `q_target`. This is a
velocity-like action in joint space but with the stability of position control.
The same action is used by BC, PPO, GAIL and the scripted expert; only the way it
is produced changes.

> **Why delta joint targets instead of torques?** Torque control would require
> the learned policies to discover a stabilising controller, which is a separate
> control problem. Position actuators let the imitation-learning comparison focus
> on the policy. The action is still continuous and bounded.

---

## 2. The MuJoCo model

Create `src/imitation_learning_reproduction/envs/assets/push_arm.xml`.

```xml
<mujoco model="push_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <visual>
    <global offwidth="96" offheight="96"/>
    <quality shadowsize="1024"/>
    <map znear="0.01" zfar="5"/>
  </visual>

  <asset>
    <material name="table" rgba="0.28 0.30 0.35 1" specular="0.05" shininess="0.1"/>
    <material name="arm"   rgba="0.75 0.76 0.80 1" specular="0.30" shininess="0.4"/>
    <material name="base"  rgba="0.18 0.19 0.22 1" specular="0.10" shininess="0.1"/>
    <material name="tool"  rgba="0.15 0.35 0.85 1" specular="0.20" shininess="0.2"/>
    <material name="puck"  rgba="0.95 0.55 0.12 1" specular="0.30" shininess="0.3"/>
    <material name="goal"  rgba="0.15 0.85 0.35 1" specular="0.20" shininess="0.2"/>
  </asset>

  <worldbody>
    <light name="key"  pos="0 0 2.2"   dir="0 0 -1"      diffuse="0.55 0.55 0.55" specular="0.10 0.10 0.10"/>
    <light name="fill" pos="1.2 0.8 1.6" dir="-1 -0.7 -1" diffuse="0.35 0.35 0.35" specular="0.10 0.10 0.10"/>

    <!-- Fixed top-down camera and a translated camera used for robustness tests. -->
    <camera name="top"       pos="0.05 0 1.15"  xyaxes="1 0 0 0 1 0" fovy="50"/>
    <camera name="top_shift" pos="0.17 0.08 1.15" xyaxes="1 0 0 0 1 0" fovy="50"/>

    <geom name="table" type="plane" size="1.0 1.0 0.1" material="table"/>

    <body name="base" pos="-0.38 0 0.06">
      <geom name="base_geom" type="cylinder" size="0.07 0.03" pos="0 0 -0.03"
            material="base" contype="0" conaffinity="0"/>

      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.0 3.0"
               damping="0.4" armature="0.01"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0.28 0 0" size="0.018"
              material="arm" contype="0" conaffinity="0"/>

        <body name="link2" pos="0.28 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1" range="-2.9 2.9"
                 damping="0.4" armature="0.01"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0.38 0 0" size="0.016"
                material="arm" contype="0" conaffinity="0"/>

          <!-- Tool: a paddle whose flat face is perpendicular to the link. -->
          <geom name="pusher" type="box" size="0.006 0.045 0.03" pos="0.39 0 -0.02"
                material="tool"/>

          <site name="ee" pos="0.38 0 0" size="0.012" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>

    <body name="puck" pos="0.10 0 0.025">
      <freejoint name="puck_free"/>
      <geom name="puck_geom" type="cylinder" size="0.035 0.025" material="puck"
            density="600" friction="0.8 0.02 0.001"/>
    </body>

    <body name="goal" pos="0.30 0.10 0">
      <geom name="goal_disc" type="cylinder" size="0.06 0.002" pos="0 0 0.002"
            material="goal" contype="0" conaffinity="0"/>
      <geom name="goal_post" type="cylinder" size="0.008 0.05" pos="0 0 0.05"
            material="goal" contype="0" conaffinity="0"/>
      <geom name="goal_top"  type="sphere" size="0.018" pos="0 0 0.10"
            material="goal" contype="0" conaffinity="0"/>
    </body>

    <!-- Visual-only distractors, parked below the table by default. -->
    <body name="distractor1" pos="0 0 -5">
      <geom name="distractor1_geom" type="box" size="0.03 0.03 0.03"
            rgba="0.8 0.2 0.2 1" contype="0" conaffinity="0"/>
    </body>
    <body name="distractor2" pos="0 0 -5">
      <geom name="distractor2_geom" type="box" size="0.025 0.04 0.02"
            rgba="0.2 0.4 0.9 1" contype="0" conaffinity="0"/>
    </body>
    <body name="distractor3" pos="0 0 -5">
      <geom name="distractor3_geom" type="sphere" size="0.03"
            rgba="0.9 0.8 0.1 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <actuator>
    <position name="act1" joint="joint1" kp="120" kv="3.0" ctrlrange="-3.0 3.0"/>
    <position name="act2" joint="joint2" kp="120" kv="3.0" ctrlrange="-2.9 2.9"/>
  </actuator>
</mujoco>
```

### Why each non-obvious choice exists

**Arm geoms have `contype="0" conaffinity="0"`.** Otherwise the base cylinder and
the first link collide at the joint, and the joint appears stuck: the actuator
pushes, MuJoCo reports a large constraint force, and the joint barely moves. The
same can happen between adjacent links. The paddle is the only part of the robot
that needs contact with the puck.

**The paddle is a box, not a sphere.** A spherical tool gives a point contact and
the puck easily squirts sideways. A flat 9 cm paddle is more stable and gives the
demonstrations a clean pushing behaviour.

**The goal is a visual-only marker.** Its geoms have `contype="0"
conaffinity="0"`, so the puck can reach the goal without colliding with it. The
post and the small sphere at the top give the point cloud something visible; a
flat disc alone is nearly indistinguishable from the table in depth.

**Two cameras.** `top` is used for training. `top_shift` is a small translation
used in chapter 12 to test whether a policy relies on a fixed camera view. The
point cloud is reconstructed in world coordinates for either camera.

**Lights are named.** The robustness experiment in chapter 13 changes
`model.light_diffuse`; names make that intention clear.

**The table is a large plane.** The camera footprint (about 1.07 m at 1.15 m
height with a 50° field of view) lies inside the 2 m plane, so every rendered
pixel is table or an object. This matters for depth backprojection: background
depth and table depth are both at the far plane, but there is no background in
the image. If you make the table smaller than the view, filter points by a
workspace bounding box.

### Verify the model before writing Python

```python
import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("src/imitation_learning_reproduction/envs/assets/push_arm.xml")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print("nq, nv, nu:", m.nq, m.nv, m.nu)
print("qpos0:", m.qpos0)
print("EE:", d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee")])
```

Expected: `nq=9`, `nv=8`, `nu=2`; the puck free joint contributes seven `qpos`
and six `qvel` values. The two arm joints contribute one each.

> **MuJoCo version note.** MuJoCo 4.0 removed the `compiler inertia` attribute
> and the `Renderer(..., depth=True)` keyword. The XML and code here are tested
> with 4.0.1. If you use 3.x, remove the `inertia` attribute if present and use
> the depth-rendering toggle shown in chapter 01; the rest is compatible.

---

## 3. The environment wrapper

Create `envs/push_arm.py`. It has four responsibilities:

1. Own `MjModel`/`MjData` and reset them deterministically.
2. Map normalized actions to joint targets and step physics.
3. Build the observation dictionary for the requested modalities.
4. Render RGB/depth and backproject a point cloud.

Keep the camera math in `cameras.py` and the Gymnasium state wrapper in
`wrappers.py`; the environment should not grow into a 700-line file.

### 3.1 Constants and configuration

```python
# envs/push_arm.py
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
```

`obs_mode` is a string rather than three booleans because combinations that do
not include state are not useful: every policy gets proprioception.

### 3.2 Initialisation

```python
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
```

The `max_episode_steps` alias is there because many Gymnasium wrappers expect
that name. Pick one and use it consistently; the config field is
`episode_steps`.

### 3.3 Task sampling and reset

The task distribution is part of the benchmark. Sample the puck in an annulus,
then the goal farther out in roughly the same direction. This creates
demonstrations where the arm starts behind the puck and pushes outward, which a
2-DoF arm can actually solve.

```python
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
```

`ik2` is a small analytic two-link inverse-kinematics function. It belongs in
`envs/kinematics.py` or `envs/expert.py` (chapter 03). For now, put it in
`push_arm.py` and move it later if you prefer:

```python
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
```

The elbow branch matters. If you switch branches between steps, the arm flips
across its workspace. Always use the same branch for the expert, and use the
current joint configuration to choose the closer branch in a learned controller.

### 3.4 Step

```python
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
```

Reward design is discussed in chapter 06. For now, note that the environment
reward is **not** used by imitation-learning chapters 7–12 except as a diagnostic.
The demonstration data and the success metric are the shared currency.

`terminate_on_success=True` means the successful demonstration ends at the first
step whose puck is within 5 cm of the goal. This avoids recording the arm
accidentally knocking the puck away after success.

### 3.5 Observations and info

```python
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
```

### 3.6 Rendering and point clouds

```python
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
```

`self.camera_name` is initialised in `__init__` and reset to the configured
camera for every episode; the `shift` variant changes it in
`_apply_visual_variant`.

### 3.7 Visual variants

These are used in chapters 12–13. Implement them now so the evaluation protocol is
fixed; do not add a new environment per robustness test.

```python
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
```

The distractors are visual-only. In the point cloud they appear as geometry, so
DP3 can see them; DP sees their colours. That is a fair difference to analyse,
not a bug.

One subtlety: `_apply_visual_variant` uses `self.np_random`, which is also used
for task sampling. Changing the visual variant therefore changes the sequence of
random numbers used for subsequent resets. For a fixed evaluation seed, this is
reproducible. If you want visual randomness to be independent of task
randomness, create a separate `self.visual_rng = np.random.default_rng(seed + 1)`.

### 3.8 Camera math: `envs/cameras.py`

```python
from __future__ import annotations

import numpy as np


def camera_fovy(model, cam_id: int) -> float:
    """Field of view in degrees, as stored by MuJoCo."""
    return float(model.cam_fovy[cam_id])


def camera_pose(data, cam_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (position, rotation) of the camera in world coordinates.

    ``data.cam_xmat`` is a row-major 3x3 rotation whose columns are the camera's
    x, y, z axes in world coordinates. The camera looks along -z.
    """
    pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64).copy()
    mat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3).copy()
    return pos, mat


def backproject(
    depth: np.ndarray,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    fovy_deg: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Turn a depth image into a world-frame point cloud.

    Args:
        depth: (H, W) metres, camera z-depth (distance along the optical axis).
        cam_pos: (3,) world position of the camera.
        cam_mat: (3, 3) camera-to-world rotation (columns are camera axes).
        fovy_deg: vertical field of view in degrees.
        width, height: image dimensions.

    Returns:
        (H, W, 3) world-frame points.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.shape != (height, width):
        raise ValueError(f"depth shape {depth.shape} != {(height, width)}")

    fy = height / (2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0))
    fx = fy
    cx, cy = width / 2.0, height / 2.0

    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    z = depth
    x_cam = (u - cx) / fx * z
    y_cam = -(v - cy) / fy * z
    z_cam = -z

    cam_points = np.stack([x_cam, y_cam, z_cam], axis=-1)
    return cam_points @ cam_mat.T + cam_pos
```

**Why `y_cam` is negated:** image row index `v` increases downward, while the
camera's y axis points up. **Why `z_cam = -z`:** MuJoCo cameras look down their
local negative z axis, while the depth image stores positive distance along the
viewing direction. Both signs are easy to get wrong; the test below catches
them.

### 3.9 Gymnasium wrapper: `envs/wrappers.py`

PPO, GAIL, AIRL and Apprenticeship Learning use state-only observations. A
one-line wrapper keeps the vectorised code simple:

```python
import gymnasium as gym


class StateOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["state"]

    def observation(self, obs):
        return obs["state"]


def make_state_env(cfg=None, seed: int = 0):
    from .push_arm import PushArmConfig, PushArmEnv

    env = PushArmEnv(cfg=cfg or PushArmConfig(obs_mode="state"))
    env = StateOnlyWrapper(env)
    env.reset(seed=seed)
    return env
```

### 3.10 The complete reset order

The correct order is:

1. Reset `qpos`, `qvel`, `ctrl`.
2. Sample puck and goal.
3. Write puck into `qpos`; write goal into `model.body_pos`.
4. Solve IK for the arm and write `qpos`/`ctrl`.
5. Apply the visual variant.
6. `mj_forward`.
7. Compute observations.

If you render before `mj_forward` after moving the goal body, the renderer can
show the goal at its old position for one frame. If you apply the visual variant
before sampling, the task distribution changes with the variant. Keep the order
fixed.

---

## 4. Tests

Create `tests/test_env.py` and `tests/test_cameras.py`.

### Environment contract

```python
import numpy as np
import pytest

from imitation_learning_reproduction.envs.push_arm import (
    ACTION_DIM, STATE_DIM, PushArmConfig, PushArmEnv,
)


@pytest.mark.parametrize("obs_mode", ["state", "image", "point_cloud", "full"])
def test_observation_shapes(obs_mode):
    env = PushArmEnv(PushArmConfig(obs_mode=obs_mode, image_size=64, n_points=128))
    obs, info = env.reset(seed=0)
    assert obs["state"].shape == (STATE_DIM,)
    assert obs["state"].dtype == np.float32
    if obs_mode in {"image", "full"}:
        assert obs["image"].shape == (64, 64, 3)
        assert obs["image"].dtype == np.uint8
        assert obs["image"].std() > 1.0
    if obs_mode in {"point_cloud", "full"}:
        assert obs["point_cloud"].shape == (128, 3)
        assert np.isfinite(obs["point_cloud"]).all()
    env.close()


def test_step_and_bounds():
    env = PushArmEnv(PushArmConfig(obs_mode="state"))
    obs, info = env.reset(seed=1)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(np.ones(ACTION_DIM))
        assert obs["state"].shape == (STATE_DIM,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
    env.close()


def test_reset_determinism():
    env = PushArmEnv(PushArmConfig(obs_mode="state"))
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    np.testing.assert_allclose(a["state"], b["state"])
    env.close()


def test_task_distribution_is_reachable():
    env = PushArmEnv(PushArmConfig(obs_mode="state"))
    for seed in range(50):
        _, info = env.reset(seed=seed)
        assert np.linalg.norm(info["goal"] - np.array([-0.38, 0.0])) < 0.63
        assert np.linalg.norm(info["puck"] - info["goal"]) > 0.10
    env.close()
```

### Camera calibration against a known object

The strongest test is geometric: put a box at a known world position, render
depth, backproject, and check that the top face's centroid matches.

```python
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")

import mujoco
import numpy as np

from imitation_learning_reproduction.envs.cameras import backproject, camera_fovy, camera_pose


def test_backproject_known_box():
    xml = """
    <mujoco>
      <visual><global offwidth="64" offheight="64"/></visual>
      <worldbody>
        <light pos="0 0 3"/>
        <camera name="top" pos="0.05 0 1.1" xyaxes="1 0 0 0 1 0" fovy="50"/>
        <geom type="plane" size="2 2 .1" rgba=".4 .4 .4 1"/>
        <body pos="0.2 -0.1 0.05"><geom type="box" size=".05 .05 .05" rgba=".8 .1 .1 1"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")

    renderer = mujoco.Renderer(model, 64, 64)
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="top")
    depth = renderer.render()

    pos, mat = camera_pose(data, cam_id)
    points = backproject(depth, pos, mat, camera_fovy(model, cam_id), 64, 64)
    points = points.reshape(-1, 3)
    top = points[(points[:, 2] > 0.09) & (points[:, 2] < 0.11) & (points[:, 0] > 0.1)]
    assert len(top) > 10
    np.testing.assert_allclose(top.mean(axis=0), [0.2, -0.1, 0.1], atol=0.01)
```

If this test fails with a reflection or a constant offset, check the signs in
`backproject` before touching the environment.

### Point-cloud content test

```python
def test_point_cloud_contains_objects():
    env = PushArmEnv(PushArmConfig(obs_mode="point_cloud", image_size=64, n_points=256))
    obs, info = env.reset(seed=3)
    pc = obs["point_cloud"]
    assert pc.shape == (256, 3)
    # The puck top is around z=0.05; the arm is around z=0.06; the goal post is taller.
    assert pc[:, 2].max() > 0.04
    assert (pc[:, 2] > 0.004).sum() > 20
    env.close()
```

The exact counts depend on the camera and scene; keep the assertions loose
enough to survive small model changes but tight enough to catch an empty cloud.

---

## 5. A visual sanity check

Write `scripts/inspect_env.py` (or run this snippet) to render one episode with
the scripted expert from chapter 03, or with random actions for now:

```python
import imageio.v2 as imageio
import numpy as np

from imitation_learning_reproduction.envs.push_arm import PushArmConfig, PushArmEnv

env = PushArmEnv(PushArmConfig(obs_mode="full"))
obs, info = env.reset(seed=0)
frames = []
for _ in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(obs["image"])
    if terminated or truncated:
        break
env.close()
imageio.mimsave("env_preview.gif", frames, fps=10)
print("saved env_preview.gif", np.asarray(frames).shape)
```

Open the GIF. You should see the arm, the orange puck, and the green goal. If the
image is mostly one colour, check the camera position, lights, and the
`_render_rgb_depth` camera name. If the arm moves but the puck never reacts, check
that the paddle has collision enabled and the arm links do not.

---

## 6. Debugging notes

| Symptom | Likely cause |
|---|---|
| Joint does not move, actuator force large | Arm link colliding with base or another link; set `contype=0` on non-contact geoms |
| Arm flips between two poses | IK elbow branch changes between steps |
| Puck is jittery or tunnels | Timestep too large, paddle too thin, or frame skip too large; keep `timestep=0.002` |
| Goal appears at the old position | Forgot `mj_forward` after changing `model.body_pos` |
| Point cloud is mirrored | Camera y-axis sign or camera rotation transpose wrong |
| Point cloud is all zeros | Depth background/table not separated, or camera sees empty space |
| Image is all white | Lights too strong / materials too bright; start from the XML above |
| `Renderer` depth keyword error | MuJoCo 4.x API; use `enable_depth_rendering()` |
| Goal post collides with puck | Goal geoms need `contype=0 conaffinity=0` |
| Reset produces an unreachable task | Rejection sampler accepted a goal outside reach; tighten bounds |

---

## 7. Checklist

- [ ] The XML loads; `nq=9`, `nv=8`, `nu=2`.
- [ ] `reset(seed)` is deterministic for the same seed.
- [ ] State is 18-D with the documented layout.
- [ ] Actions are clipped to `[-1,1]` and map to joint targets.
- [ ] The arm starts behind the puck and can reach the goal region.
- [ ] RGB is 96×96, non-flat, and shows arm/puck/goal.
- [ ] Point cloud is 512×3 in world coordinates; a known box backprojects within 1 cm.
- [ ] The point cloud contains object points above the table.
- [ ] Visual variants `color`, `light`, `shift`, `distractors` run without changing physics.
- [ ] All tests in `tests/test_env.py` and `tests/test_cameras.py` pass.

### Exercises

1. Add a third arm joint and make the action 3-D. How does the IK branch choice
   change? What does this do to the demonstration distribution?
2. Replace the paddle with a spherical tool and measure the expert success rate.
   Explain the difference.
3. Add a joint-velocity penalty to the reward and see how PPO behaves.
4. Implement a `reset(options={"puck": ..., "goal": ...})` path for fixed
   evaluation tasks. You will need it for paired comparisons.
5. Add a second camera (wrist view) and verify that both cameras backproject to
   the same world-frame puck position.

Next: [03 · Expert demonstrations and the dataset](03_data.md).
