# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MuJoCo sim camera intrinsics constant, shared by sim connection modules
(Go2MujocoConnection, G1MujocoConnection) and by external readers that need
the value without pulling in the full mujoco transport class.
"""

from __future__ import annotations

import math

from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.simulation.mujoco.constants import (
    VIDEO_CAMERA_FOV,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)


def _compute_mujoco_camera_info() -> CameraInfo:
    """Pinhole model: f = height / (2 * tan(fovy / 2))."""
    fovy = math.radians(VIDEO_CAMERA_FOV)
    f = VIDEO_HEIGHT / (2 * math.tan(fovy / 2))
    cx = VIDEO_WIDTH / 2.0
    cy = VIDEO_HEIGHT / 2.0

    return CameraInfo(
        frame_id="camera_optical",
        height=VIDEO_HEIGHT,
        width=VIDEO_WIDTH,
        distortion_model="plumb_bob",
        D=[0.0, 0.0, 0.0, 0.0, 0.0],
        K=[f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0],
        R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        P=[f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    )


MUJOCO_CAMERA_INFO_STATIC: CameraInfo = _compute_mujoco_camera_info()
