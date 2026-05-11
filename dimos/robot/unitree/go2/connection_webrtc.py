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

"""Real-hardware Go2 connection over Unitree's WebRTC stack.

Self-contained: WebRTC transport, stream wiring, and RPC surface live in
one Module class. No shared base.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import Event, Thread, Timer
import time
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from reactivex import operators as ops
from reactivex.disposable import Disposable
from reactivex.observable import Observable
from reactivex.subject import Subject
import rerun.blueprint as rrb
from unitree_webrtc_connect.constants import (
    RTC_TOPIC,
    SPORT_CMD,
)
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection as LegionConnection,
    WebRTCConnectionMethod,
)

from dimos.agents.annotation import skill
from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.connection_registry import connection
from dimos.robot.unitree.go2.camera import _camera_info_static
from dimos.robot.unitree.go2.config import ConnectionConfig, Go2Mode
from dimos.robot.unitree.type.lidar import (
    pointcloud2_from_webrtc_lidar,
    repair_stale_ts,
)
from dimos.robot.unitree.type.odometry import Odometry as OdometryConverter
from dimos.spec.perception import Camera, Pointcloud
from dimos.utils.decorators.decorators import simple_mcache
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure, callback_to_observable

VideoMessage: TypeAlias = NDArray[np.uint8]

logger = setup_logger()


@dataclass
class SerializableVideoFrame:
    """Pickleable wrapper for av.VideoFrame with all metadata."""

    data: np.ndarray  # type: ignore[type-arg]
    pts: int | None = None
    time: float | None = None
    dts: int | None = None
    width: int | None = None
    height: int | None = None
    format: str | None = None

    @classmethod
    def from_av_frame(cls, frame: Any) -> SerializableVideoFrame:
        return cls(
            data=frame.to_ndarray(format="rgb24"),
            pts=frame.pts,
            time=frame.time,
            dts=frame.dts,
            width=frame.width,
            height=frame.height,
            format=frame.format.name if hasattr(frame, "format") and frame.format else None,
        )

    def to_ndarray(self, format: str | None = None) -> np.ndarray:  # type: ignore[type-arg]
        return self.data


_SPORT_API_ID_RAGEMODE: int = 2059


@connection(robot="go2", backend="webrtc")
class Go2WebRtcConnection(Module, Camera, Pointcloud):
    """Real-hardware Go2 connection over Unitree's WebRTC stack."""

    config: ConnectionConfig
    cmd_vel: In[Twist]
    pointcloud: Out[PointCloud2]
    odom: Out[PoseStamped]
    lidar: Out[PointCloud2]
    color_image: Out[Image]
    camera_info: Out[CameraInfo]

    camera_info_static: CameraInfo = _camera_info_static()
    _camera_info_thread: Thread | None = None
    _latest_video_frame: Image | None = None

    @classmethod
    def rerun_views(cls):  # type: ignore[no-untyped-def]
        return [
            rrb.Spatial2DView(
                name="Camera",
                origin="world/robot/camera/rgb",
            ),
        ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert self.config.ip is not None, "IP address must be provided"

        self.cmd_vel_timeout = 0.2
        self.stop_timer: Timer | None = None
        self.task: asyncio.Task[None] | None = None
        self.loop = asyncio.new_event_loop()
        self.connected_event = asyncio.Event()
        self.connection_ready = Event()

        self.conn = LegionConnection(WebRTCConnectionMethod.LocalSTA, ip=self.config.ip)

        mode_name = "ai"

        async def async_connect() -> None:
            await self.conn.connect()
            await self.conn.datachannel.disableTrafficSaving(True)
            self.conn.datachannel.set_decoder(decoder_type="native")
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": mode_name}},
            )
            self.connected_event.set()
            self.connection_ready.set()
            while True:
                await asyncio.sleep(1)

        def start_background_loop() -> None:
            asyncio.set_event_loop(self.loop)
            self.task = self.loop.create_task(async_connect())
            self.loop.run_forever()

        self.thread = Thread(target=start_background_loop, daemon=True)
        self.thread.start()
        self.connection_ready.wait()

    @rpc
    def start(self) -> None:
        super().start()

        def onimage(image: Image) -> None:
            self.color_image.publish(image)
            self._latest_video_frame = image

        self.register_disposable(self._lidar_stream().subscribe(self.lidar.publish))
        self.register_disposable(self._odom_stream().subscribe(self._publish_tf))
        self.register_disposable(self._video_stream().subscribe(onimage))
        self.register_disposable(Disposable(self.cmd_vel.subscribe(self.move)))

        self._camera_info_thread = Thread(
            target=self._publish_camera_info,
            daemon=True,
        )
        self._camera_info_thread.start()

        self.standup()
        time.sleep(3)
        self.balance_stand()

        if self.config.mode == Go2Mode.RAGE:
            self.enable_rage_mode()

        self.set_obstacle_avoidance(self.config.g.obstacle_avoidance)

    @rpc
    def stop(self) -> None:
        self.liedown()

        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

        if self.task:
            self.task.cancel()

        async def async_disconnect() -> None:
            try:
                self.conn.datachannel.pub_sub.publish_without_callback(
                    RTC_TOPIC["WIRELESS_CONTROLLER"],
                    data={"lx": 0, "ly": 0, "rx": 0, "ry": 0},
                )
                await self.conn.disconnect()
            except Exception:
                pass

        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(async_disconnect(), self.loop)
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.thread.is_alive():
            self.thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)

        if self._camera_info_thread and self._camera_info_thread.is_alive():
            self._camera_info_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)

        super().stop()

    def _unitree_sub_stream(self, topic_name: str):  # type: ignore[no-untyped-def]
        def subscribe_in_thread(cb) -> None:  # type: ignore[no-untyped-def]
            def run_subscription() -> None:
                self.conn.datachannel.pub_sub.subscribe(topic_name, cb)

            self.loop.call_soon_threadsafe(run_subscription)

        def unsubscribe_in_thread(cb) -> None:  # type: ignore[no-untyped-def]
            def run_unsubscription() -> None:
                self.conn.datachannel.pub_sub.unsubscribe(topic_name)

            self.loop.call_soon_threadsafe(run_unsubscription)

        return callback_to_observable(
            start=subscribe_in_thread,
            stop=unsubscribe_in_thread,
        )

    def _publish_request_raw(self, topic: str, data: dict[Any, Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(
            self.conn.datachannel.pub_sub.publish_request_new(topic, data), self.loop
        )
        return future.result()

    @simple_mcache
    def _raw_lidar_stream(self) -> Observable:  # type: ignore[type-arg]
        return backpressure(self._unitree_sub_stream(RTC_TOPIC["ULIDAR_ARRAY"]))

    @simple_mcache
    def _raw_odom_stream(self) -> Observable:  # type: ignore[type-arg]
        return backpressure(self._unitree_sub_stream(RTC_TOPIC["ROBOTODOM"]))

    @simple_mcache
    def _lidar_stream(self) -> Observable[PointCloud2]:
        return backpressure(
            self._raw_lidar_stream().pipe(
                ops.map(pointcloud2_from_webrtc_lidar),
                repair_stale_ts(),
            )
        )

    @simple_mcache
    def _odom_stream(self) -> Observable[PoseStamped]:
        return backpressure(self._raw_odom_stream().pipe(ops.map(OdometryConverter.from_msg)))

    @simple_mcache
    def _video_stream(self) -> Observable[Image]:
        return backpressure(
            self._raw_video_stream().pipe(
                ops.filter(lambda frame: frame is not None),
                ops.map(
                    lambda frame: Image.from_numpy(
                        frame.to_ndarray(format="rgb24"),
                        format=ImageFormat.RGB,
                        frame_id="camera_optical",
                    )
                ),
            )
        )

    @simple_mcache
    def _raw_video_stream(self) -> Observable[VideoMessage]:
        subject: Subject[VideoMessage] = Subject()
        stop_event = Event()

        from aiortc import MediaStreamTrack

        async def accept_track(track: MediaStreamTrack) -> None:
            while True:
                if stop_event.is_set():
                    return
                frame = await track.recv()
                serializable_frame = SerializableVideoFrame.from_av_frame(frame)
                subject.on_next(serializable_frame)  # type: ignore[arg-type]

        self.conn.video.add_track_callback(accept_track)

        def switch_video_channel() -> None:
            self.conn.video.switchVideoChannel(True)

        self.loop.call_soon_threadsafe(switch_video_channel)

        def stop() -> None:
            stop_event.set()
            self.conn.video.track_callbacks.remove(accept_track)

            def switch_video_channel_off() -> None:
                self.conn.video.switchVideoChannel(False)

            self.loop.call_soon_threadsafe(switch_video_channel_off)

        return subject.pipe(ops.finally_action(stop))

    def _stop_movement(self) -> None:
        """Cancel the auto-stop timer (used by move() for continuous commands)."""
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        """Send movement command to robot."""
        x, y, yaw = twist.linear.x, twist.linear.y, twist.angular.z

        # WebRTC coordinate mapping:
        # x - Positive right, negative left
        # y - positive forward, negative backwards
        # yaw - Positive rotate right, negative rotate left
        async def async_move() -> None:
            self.conn.datachannel.pub_sub.publish_without_callback(
                RTC_TOPIC["WIRELESS_CONTROLLER"],
                data={"lx": -y, "ly": x, "rx": -yaw, "ry": 0},
            )

        async def async_move_duration() -> None:
            """Send movement commands continuously for the specified duration."""
            start_time = time.time()
            while time.time() - start_time < duration:
                await async_move()
                await asyncio.sleep(0.01)

        if self.stop_timer:
            self.stop_timer.cancel()

        # Auto-stop after cmd_vel_timeout seconds if no new commands
        self.stop_timer = Timer(self.cmd_vel_timeout, self._stop_movement)
        self.stop_timer.daemon = True
        self.stop_timer.start()

        try:
            if duration > 0:
                future = asyncio.run_coroutine_threadsafe(async_move_duration(), self.loop)
                future.result()
                self._stop_movement()
            else:
                future = asyncio.run_coroutine_threadsafe(async_move(), self.loop)
                future.result()
            return True
        except Exception as e:
            logger.error(f"Failed to send movement command: {e}")
            return False

    @rpc
    def standup(self) -> bool:
        """Make the robot stand up."""
        return bool(
            self._publish_request_raw(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]})
        )

    @rpc
    def liedown(self) -> bool:
        """Make the robot lie down."""
        return bool(
            self._publish_request_raw(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandDown"]})
        )

    @rpc
    def balance_stand(self) -> bool:
        """Enter BalanceStand: neutral state for switching locomotion modes."""
        return bool(
            self._publish_request_raw(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]})
        )

    @rpc
    def enable_rage_mode(self) -> bool:
        """Enable Rage Mode (~2.5 m/s forward velocity envelope).

        Ensures BalanceStand precondition regardless of current FSM state.
        """
        # Force BalanceStand first.
        self._publish_request_raw(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]})
        time.sleep(0.3)

        rage_ok = bool(
            self._publish_request_raw(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": _SPORT_API_ID_RAGEMODE, "parameter": {"data": True}},
            )
        )
        time.sleep(2.0)

        joystick_ok = bool(
            self._publish_request_raw(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD["SwitchJoystick"], "parameter": {"data": True}},
            )
        )
        logger.info("Rage Mode enabled")
        return rage_ok and joystick_ok

    def set_obstacle_avoidance(self, enabled: bool = True) -> None:
        self._publish_request_raw(
            RTC_TOPIC["OBSTACLES_AVOID"],
            {"api_id": 1001, "parameter": {"enable": int(enabled)}},
        )

    @rpc
    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        """Publish a request to the underlying connection."""
        return self._publish_request_raw(topic, data)

    @classmethod
    def _odom_to_tf(cls, odom: PoseStamped) -> list[Transform]:
        camera_link = Transform(
            translation=Vector3(0.3, 0.0, 0.0),
            rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
            frame_id="base_link",
            child_frame_id="camera_link",
            ts=odom.ts,
        )

        camera_optical = Transform(
            translation=Vector3(0.0, 0.0, 0.0),
            rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
            frame_id="camera_link",
            child_frame_id="camera_optical",
            ts=odom.ts,
        )

        return [
            Transform.from_pose("base_link", odom),
            camera_link,
            camera_optical,
        ]

    def _publish_tf(self, msg: PoseStamped) -> None:
        transforms = self._odom_to_tf(msg)
        self.tf.publish(*transforms)
        if self.odom.transport:
            self.odom.publish(msg)

    def _publish_camera_info(self) -> None:
        while True:
            self.camera_info.publish(self.camera_info_static)
            time.sleep(1.0)

    @skill
    def observe(self) -> Image | None:
        """Returns the latest video frame from the robot camera. Use this skill for any visual world queries.

        This skill provides the current camera view for perception tasks.
        Returns None if no frame has been captured yet.
        """
        return self._latest_video_frame
