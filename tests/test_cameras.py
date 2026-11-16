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

