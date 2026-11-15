from dataclasses import dataclass, field

import mujoco
import numpy as np
import pytest
import torch

from imitation_learning_reproduction.utils.seed import resolve_device, seed_everything


def test_torch_works():
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 3, device=device)
    y = torch.randn(3, 2, device=device)
    assert torch.allclose(x @ y, (y.T @ x.T).T, atol=1e-6)
    assert x.device.type == device.type


def test_mujoco_render():
    xml = """
    <mujoco>
      <visual><global offwidth="64" offheight="64"/></visual>
      <worldbody>
        <light pos="0 0 2"/>
        <geom type="plane" size="1 1 .1" rgba=".5 .5 .5 1"/>
        <body pos="0 0 .1"><geom type="sphere" size=".05" rgba="1 0 0 1"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 64, 64)
    renderer.update_scene(data)
    image = renderer.render()
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8
    assert image.std() > 1.0, "the image is flat; rendering probably failed"


def test_device_resolution():
    assert resolve_device("cpu").type == "cpu"
    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is the training default; run this on a CUDA machine")
def test_cuda_device():
    assert resolve_device("cuda").type == "cuda"

