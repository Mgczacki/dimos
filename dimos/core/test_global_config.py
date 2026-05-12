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

from dimos.core.global_config import GlobalConfig


class TestGlobalConfigSecurityDefaults:
    """Network services must bind to localhost by default (not 0.0.0.0)."""

    def test_listen_host_defaults_to_localhost(self) -> None:
        config = GlobalConfig()
        assert config.listen_host == "127.0.0.1", (
            f"listen_host must default to 127.0.0.1, got {config.listen_host}"
        )


class TestSimulatorBackendResolution:
    """`--simulator` and `--simulation` translate into the connection backend."""

    def test_simulator_takes_precedence_over_simulation(self) -> None:
        config = GlobalConfig(simulation=True, simulator="simsim")
        assert config.effective_simulator == "simsim"
        assert config.unitree_connection_type == "simsim"

    def test_simulation_back_compat_resolves_to_mujoco(self) -> None:
        config = GlobalConfig(simulation=True)
        assert config.effective_simulator == "mujoco"
        assert config.unitree_connection_type == "mujoco"

    def test_neither_set_returns_none_and_webrtc(self) -> None:
        config = GlobalConfig(simulation=False, simulator=None)
        assert config.effective_simulator is None
        assert config.unitree_connection_type == "webrtc"

    def test_replay_overrides_simulator(self) -> None:
        config = GlobalConfig(replay=True, simulator="mujoco")
        assert config.unitree_connection_type == "replay"
