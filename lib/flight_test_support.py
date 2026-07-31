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

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import flight_core as fc
import led_interface as led


@dataclass
class FakeTelemetry:
    x: float
    y: float
    z: float
    armed: bool
    # Курс дрона в запрошенном фрейме - нужен flight_nav для пересчёта
    # смещения из aruco_map в body (режим перелёта 'relative'). FakeFlight
    # курс не меняет: заглушка «телепортирует» дрон, не имитируя динамику.
    yaw: float = 0.0


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

    ``frame_id='body'`` при этом трактуется как **приращение** к текущему
    положению (с учётом курса), а не как абсолютная точка: иначе взлёт
    ``navigate(0, 0, z, frame_id='body')`` «переносил» бы заглушку в начало
    координат поля, и любая проверка позиции после взлёта проверяла бы не то.
    """

    def __init__(self, start_x: float, start_y: float, start_z: float = 0.0) -> None:
        self.state: Dict[str, Any] = {
            "x": start_x,
            "y": start_y,
            "z": start_z,
            "yaw": 0.0,
            "armed": False,
        }
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
        yaw: Optional[float] = None,
    ) -> None:
        # yaw=None в записи означает «параметр вообще не передавали» (сервис
        # подставил бы своё значение по умолчанию) — самотесты сценариев
        # отличают это от явного удержания курса yaw=NaN. На «полёт» заглушки
        # курс не влияет: перемещение в body считается по state['yaw'].
        self.navigate_calls.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "frame_id": frame_id,
                "auto_arm": auto_arm,
                "yaw": yaw,
            }
        )
        if frame_id == "body":
            drone_yaw = self.state["yaw"]
            self.state["x"] += x * math.cos(drone_yaw) - y * math.sin(drone_yaw)
            self.state["y"] += x * math.sin(drone_yaw) + y * math.cos(drone_yaw)
            self.state["z"] += z
        else:
            self.state["x"], self.state["y"], self.state["z"] = x, y, z
        self.state["armed"] = True

    def _get_telemetry(self, *, frame_id: str = "aruco_map", **_kwargs: object) -> FakeTelemetry:
        if frame_id == "navigate_target":
            # "телепорт" в _navigate() мгновенный, поэтому дрон уже в цели -
            # остаток пути в этом фрейме всегда (0, 0, 0), как и должно быть
            # у реального navigate_target сразу после прибытия.
            return FakeTelemetry(0.0, 0.0, 0.0, self.state["armed"])
        return FakeTelemetry(
            self.state["x"],
            self.state["y"],
            self.state["z"],
            self.state["armed"],
            self.state["yaw"],
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


def fixed_marker_reader(*marker_ids: int) -> Any:
    """Заглушка ``marker_id_reader`` — камера "всегда видит" ровно эти ID.

    Имитирует реальную сцену «дрон после взлёта видит метку под собой»,
    не поднимая ROS 1/камеру — используется вместо
    ``flight_core.read_visible_marker_ids_ros`` в самотестах."""

    def _reader(_timeout: float) -> Tuple[int, ...]:
        return tuple(marker_ids)

    return _reader


def visible_marker_reader(
    flight: FakeFlight,
    markers: Dict[int, Tuple[float, float, float]],
    *,
    radius: float = 1.5,
) -> Callable[[float], Tuple[int, ...]]:
    """Заглушка ``marker_id_reader``, привязанная к позиции дрона: «видны»
    метки в радиусе ``radius`` от неё (если ни одной — ближайшая).

    В отличие от ``fixed_marker_reader`` подходит для проверок
    согласованности позиции с тем, что реально видит камера
    (``flight_nav.read_localization``): камера смотрит вниз, поэтому далёкая
    метка в кадре — признак того, что локализация врёт, и заглушка должна
    вести себя так же, иначе самотест «пролетал» бы аварийные сценарии.
    """

    def _reader(_timeout: float) -> Tuple[int, ...]:
        x, y = flight.state["x"], flight.state["y"]

        def _distance(marker_id: int) -> float:
            mx, my, _mz = markers[marker_id]
            return math.hypot(mx - x, my - y)

        visible = tuple(sorted(mid for mid in markers if _distance(mid) <= radius))
        if visible:
            return visible
        return (min(markers, key=_distance),)

    return _reader


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
