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

"""Go2 Fleet Connection — manage multiple Go2 robots as a fleet.

The primary robot uses the full Go2WebRtcConnection (sensors + RPCs).
Additional robots use a minimal command-only client (no sensor streams)
embedded directly below.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import sys
from threading import Event, Thread, Timer
import time
from typing import TYPE_CHECKING, Any

from pydantic import Field, model_validator
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection as LegionConnection,
    WebRTCConnectionMethod,
)

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.robot.unitree.go2.config import ConnectionConfig
from dimos.robot.unitree.go2.connection_webrtc import Go2WebRtcConnection
from dimos.utils.logging_config import setup_logger

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from dimos.msgs.geometry_msgs.Twist import Twist

logger = setup_logger()


class FleetConnectionConfig(ConnectionConfig):
    ips: Sequence[str] = Field(
        default_factory=lambda m: [ip.strip() for ip in m["g"].robot_ips.split(",")]
    )

    @model_validator(mode="after")
    def set_ip_after_validation(self) -> Self:
        if self.ip is None:
            self.ip = self.ips[0]
        return self


class _FleetMemberClient:
    """Minimal command-only WebRTC client for extra fleet robots.

    Owns the LegionConnection + asyncio event loop, but exposes only the
    command surface (move/standup/liedown/balance_stand/set_obstacle_avoidance/
    publish_request). No sensor streams — fleet does not subscribe to extras.
    """

    def __init__(self, ip: str) -> None:
        self.ip = ip
        self.cmd_vel_timeout = 0.2
        self.stop_timer: Timer | None = None
        self.task: asyncio.Task[None] | None = None
        self.loop = asyncio.new_event_loop()
        self.connection_ready = Event()
        self.conn = LegionConnection(WebRTCConnectionMethod.LocalSTA, ip=ip)

        async def async_connect() -> None:
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
            asyncio.set_event_loop(self.loop)
            self.task = self.loop.create_task(async_connect())
            self.loop.run_forever()

        self.thread = Thread(target=start_background_loop, daemon=True)
        self.thread.start()
        self.connection_ready.wait()

    def start(self) -> None:
        # Handshake already happened in __init__.
        pass

    def stop(self) -> None:
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

    def _stop_movement(self) -> None:
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

    def _publish_request_raw(self, topic: str, data: dict[Any, Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(
            self.conn.datachannel.pub_sub.publish_request_new(topic, data), self.loop
        )
        return future.result()

    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        x, y, yaw = twist.linear.x, twist.linear.y, twist.angular.z

        async def async_move() -> None:
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
            return True
        except Exception as e:
            logger.error(f"Failed to send fleet-member movement command: {e}")
            return False

    def standup(self) -> bool:
        return bool(
            self._publish_request_raw(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]})
        )

    def liedown(self) -> bool:
        return bool(
            self._publish_request_raw(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandDown"]})
        )

    def balance_stand(self) -> bool:
        return bool(
            self._publish_request_raw(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]})
        )

    def set_obstacle_avoidance(self, enabled: bool = True) -> None:
        self._publish_request_raw(
            RTC_TOPIC["OBSTACLES_AVOID"],
            {"api_id": 1001, "parameter": {"enable": int(enabled)}},
        )

    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        return self._publish_request_raw(topic, data)  # type: ignore[no-any-return]


class Go2FleetConnection(Go2WebRtcConnection):
    """Inherits all single-robot behaviour from Go2WebRtcConnection for the
    primary (first) robot. Additional robots only receive broadcast commands
    (move, standup, liedown, balance_stand, set_obstacle_avoidance,
    publish_request) via _FleetMemberClient.

    Fleets are real-hardware only — there's no sim/replay equivalent.
    """

    config: FleetConnectionConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._extra_ips = self.config.ips[1:]
        self._extra_connections: list[_FleetMemberClient] = []

    @rpc
    def start(self) -> None:
        self._extra_connections.clear()
        for ip in self._extra_ips:
            client = _FleetMemberClient(ip)
            client.start()
            self._extra_connections.append(client)

        # Parent starts primary robot, subscribes sensors, calls standup() on it.
        super().start()
        for client in self._extra_connections:
            client.balance_stand()
            client.set_obstacle_avoidance(self.config.g.obstacle_avoidance)

    @rpc
    def stop(self) -> None:
        # One robot's error must not prevent others from stopping.
        for client in self._extra_connections:
            try:
                client.liedown()
            except Exception as e:
                logger.error(f"Error lying down fleet Go2: {e}")
            try:
                client.stop()
            except Exception as e:
                logger.error(f"Error stopping fleet Go2: {e}")
        self._extra_connections.clear()
        super().stop()

    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        results: list[bool] = [super().move(twist, duration)]
        for client in self._extra_connections:
            try:
                results.append(client.move(twist, duration))
            except Exception as e:
                logger.error(f"Fleet move failed: {e}")
                results.append(False)
        return all(results)

    @rpc
    def standup(self) -> bool:
        results: list[bool] = [super().standup()]
        for client in self._extra_connections:
            try:
                results.append(client.standup())
            except Exception as e:
                logger.error(f"Fleet standup failed: {e}")
                results.append(False)
        return all(results)

    @rpc
    def liedown(self) -> bool:
        results: list[bool] = [super().liedown()]
        for client in self._extra_connections:
            try:
                results.append(client.liedown())
            except Exception as e:
                logger.error(f"Fleet liedown failed: {e}")
                results.append(False)
        return all(results)

    @rpc
    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        """Publish a request to all robots, return primary's response."""
        for client in self._extra_connections:
            try:
                client.publish_request(topic, data)
            except Exception as e:
                logger.error(f"Fleet publish_request failed: {e}")
        return super().publish_request(topic, data)
