#!/usr/bin/env python3
"""Общие тестовые заглушки для самотестов bvs1_flight.py/bvs2_flight.py.

Оба сценария проверяют одну и ту же связку ``flight_core.FlightProxies`` +
``led_interface`` заглушками, отличающимися только начальным положением
дрона и картой поля — этот модуль вынесен, чтобы не дублировать фейковую
телеметрию/навигацию/ленту в каждом ``_self_test()``.

Не является боевым API полёта: импортируется только из ``_self_test()``
полётных сценариев, ``rospy``/``technic`` не использует.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import flight_core as fc
import led_interface as led


@dataclass
class FakeTelemetry:
    x: float
    y: float
    z: float
    armed: bool


class FakeClock:
    """Управляемые ``sleep_fn``/``time_fn`` для самотестов — время идёт
    только вперёд ровно на переданную длительность, без реальных задержек."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def sleep(self, duration: float) -> None:
        self._now += duration

    def time(self) -> float:
        return self._now


class FakeFlight:
    """Заглушка ``FlightProxies`` с учётом вызовов navigate()/arming() для
    самотестов полётных сценариев.

    ``navigate()`` мгновенно "телепортирует" дрон в целевую точку и переводит
    его в armed=True (как настоящий взлёт/перелёт), не имитируя физику
    полёта — самотесты проверяют порядок вызовов и параметры, а не динамику.
    """

    def __init__(self, start_x: float, start_y: float, start_z: float = 0.0) -> None:
        self.state: Dict[str, Any] = {
            "x": start_x,
            "y": start_y,
            "z": start_z,
            "armed": False,
        }
        self._last_target: Tuple[float, float, float] = (start_x, start_y, start_z)
        self.navigate_calls: List[Dict[str, Any]] = []
        self.disarm_calls: List[bool] = []
        self.proxies = fc.FlightProxies(
            get_telemetry=self._get_telemetry,
            navigate=self._navigate,
            land=self._land,
            arming=self._arming,
        )

    def _navigate(
        self,
        *,
        x: float,
        y: float,
        z: float,
        speed: float,
        frame_id: str,
        auto_arm: bool = False,
    ) -> None:
        self.navigate_calls.append(
            {"x": x, "y": y, "z": z, "frame_id": frame_id, "auto_arm": auto_arm}
        )
        self.state["x"], self.state["y"], self.state["z"] = x, y, z
        self.state["armed"] = True
        self._last_target = (x, y, z)

    def _get_telemetry(self, *, frame_id: str = "aruco_map", **_kwargs: object) -> FakeTelemetry:
        if frame_id == "navigate_target":
            # "телепорт" в _navigate() мгновенный, поэтому дрон уже в цели -
            # остаток пути в этом фрейме всегда (0, 0, 0), как и должно быть
            # у реального navigate_target сразу после прибытия.
            tx, ty, tz = self._last_target
            return FakeTelemetry(
                self.state["x"] - tx,
                self.state["y"] - ty,
                self.state["z"] - tz,
                self.state["armed"],
            )
        return FakeTelemetry(
            self.state["x"], self.state["y"], self.state["z"], self.state["armed"]
        )

    def _land(self) -> None:
        self.state["armed"] = False

    def _arming(self, value: bool) -> None:
        self.disarm_calls.append(value)
        self.state["armed"] = value


def instant_touchdown_rangefinder(_timeout: float) -> float:
    """Мгновенное «касание» дальномера — ускоряет самотесты управляемой
    посадки, не дожидаясь пошагового снижения."""
    return 0.05


def record_led_calls() -> Tuple[List[Tuple[str, Any]], led.LEDBackend]:
    """Подключить LED backend, который просто запоминает вызовы, и вернуть
    список записей вместе с самим backend."""
    recorded: List[Tuple[str, Any]] = []

    def effect_writer(effect: str, color: Any) -> None:
        recorded.append((effect, color))

    def pixels_writer(colors: Sequence[Any]) -> None:
        recorded.append(("pixels", tuple(colors)))

    backend = led.CallbackBackend(12, effect_writer, pixels_writer)
    led.configure_backend(backend)
    return recorded, backend
