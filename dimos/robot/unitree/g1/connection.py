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

"""Real-hardware G1 connection over Unitree's WebRTC stack.

Self-contained: WebRTC transport, cmd_vel subscription, and RPC surface in
one Module class. No shared base.
"""

from __future__ import annotations

import asyncio
from threading import Event, Thread, Timer
import time
from typing import Any

from pydantic import Field
from reactivex.disposable import Disposable
from unitree_webrtc_connect.constants import RTC_TOPIC
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection as LegionConnection,
    WebRTCConnectionMethod,
)

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.robot.connection_registry import connection
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class G1Config(ModuleConfig):
    ip: str = Field(default_factory=lambda m: m["g"].robot_ip)


@connection(robot="g1", backend="webrtc")
class G1WebRtcConnection(Module):
    """Real-hardware G1 connection over Unitree's WebRTC stack."""

    config: G1Config
    cmd_vel: In[Twist]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cmd_vel_timeout = 0.2
        self.stop_timer: Timer | None = None
        self.task: asyncio.Task[None] | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: Thread | None = None
        self.connection_ready = Event()
        self.conn: LegionConnection | None = None

    @rpc
    def start(self) -> None:
        super().start()

        assert self.config.ip is not None, "IP address must be provided"

        self.loop = asyncio.new_event_loop()
        self.conn = LegionConnection(WebRTCConnectionMethod.LocalSTA, ip=self.config.ip)

        async def async_connect() -> None:
            assert self.conn is not None
            await self.conn.connect()
            await self.conn.datachannel.disableTrafficSaving(True)
            self.conn.datachannel.set_decoder(decoder_type="native")
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": "ai"}},
            )
            self.connection_ready.set()
            while True:
                await asyncio.sleep(1)

        def start_background_loop() -> None:
            assert self.loop is not None
            asyncio.set_event_loop(self.loop)
            self.task = self.loop.create_task(async_connect())
            self.loop.run_forever()

        self.thread = Thread(target=start_background_loop, daemon=True)
        self.thread.start()
        self.connection_ready.wait()

        self.register_disposable(Disposable(self.cmd_vel.subscribe(self.move)))

    @rpc
    def stop(self) -> None:
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

        if self.task:
            self.task.cancel()

        async def async_disconnect() -> None:
            try:
                assert self.conn is not None
                self.conn.datachannel.pub_sub.publish_without_callback(
                    RTC_TOPIC["WIRELESS_CONTROLLER"],
                    data={"lx": 0, "ly": 0, "rx": 0, "ry": 0},
                )
                await self.conn.disconnect()
            except Exception:
                pass

        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(async_disconnect(), self.loop)
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)

        super().stop()

    def _stop_movement(self) -> None:
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> None:
        assert self.conn is not None
        assert self.loop is not None

        x, y, yaw = twist.linear.x, twist.linear.y, twist.angular.z

        async def async_move() -> None:
            assert self.conn is not None
            self.conn.datachannel.pub_sub.publish_without_callback(
                RTC_TOPIC["WIRELESS_CONTROLLER"],
                data={"lx": -y, "ly": x, "rx": -yaw, "ry": 0},
            )

        async def async_move_duration() -> None:
            start_time = time.time()
            while time.time() - start_time < duration:
                await async_move()
                await asyncio.sleep(0.01)

        if self.stop_timer:
            self.stop_timer.cancel()

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
        except Exception as e:
            logger.error(f"Failed to send G1 movement command: {e}")

    @rpc
    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        logger.info(f"Publishing request to topic: {topic} with data: {data}")
        assert self.conn is not None
        assert self.loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self.conn.datachannel.pub_sub.publish_request_new(topic, data), self.loop
        )
        return future.result()  # type: ignore[no-any-return]
