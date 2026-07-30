#!/usr/bin/env python3
"""Общие примитивы автономного полёта для Skyris Technic 6S (ROS 1/rospy).

Используется обоими полётными сценариями («Энергоэстафета» — БВС-1 и БВС-2),
поэтому вынесено в отдельный модуль, а не продублировано в каждом скрипте.

Основано на официальных сервисах пакета ``technic``
(docs.skyris.ru/technic6S/AutonomousFlightOffboard.html,
docs.skyris.ru/technic6S/CodeExamples.html), карте меток ``aruco_map``
(docs.skyris.ru/technic6S/ArucoMap.html) и лазерном дальномере
(docs.skyris.ru/technic6S/LaserRangefinder.html)::

    import flight_core as fc

    proxies = fc.init_flight('bvs1_flight')
    markers = fc.read_map('field_map.txt')
    x, y = fc.marker_xy(markers, 48)
    fc.navigate_wait(proxies.navigate, proxies.get_telemetry,
                      x=x, y=y, z=1.5, frame_id='aruco_map', auto_arm=True)

Имя ROS-сервиса армирования (``mavros/cmd/arming``) — стандартное для
MAVROS, но не подтверждено явно документацией Skyris; проверить на площадке
в первый день вместе с остальными допущениями (см. TASK.md/PLAN.md).

Все функции принимают прокси (``navigate``, ``get_telemetry``, ``land``,
``arming``, читатель дальномера) явными аргументами, а не берут их из
глобального состояния — так модуль можно проверить самотестом без
установленного ROS 1, подставив обычные функции-заглушки.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from time import sleep
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class FlightProxies:
    """Набор ROS-прокси, которые использует один полётный узел."""

    get_telemetry: Callable[..., Any]
    navigate: Callable[..., Any]
    land: Callable[[], Any]
    arming: Callable[[bool], Any]


@dataclass(frozen=True)
class DescentResult:
    """Итог управляемого спуска на приподнятую поверхность."""

    final_z: float
    touchdown_detected: bool
    last_range: Optional[float]


def init_flight(node_name: str = "flight", *, init_node: bool = True) -> FlightProxies:
    """Поднять узел ROS 1 и собрать прокси navigate/get_telemetry/land/arming.

    Импорт ``rospy``/``technic``/``mavros_msgs`` выполняется только здесь —
    остальной модуль от ROS не зависит и тестируется без него.
    """
    try:
        import rospy
        from technic import srv
        from std_srvs.srv import Trigger
        from mavros_msgs.srv import CommandBool
    except ImportError as exc:
        raise RuntimeError(
            "Нужны rospy, technic, std_srvs и mavros_msgs. Выполните source "
            "/opt/ros/noetic/setup.bash и source своего catkin_ws"
        ) from exc

    if init_node:
        rospy.init_node(node_name)

    get_telemetry = rospy.ServiceProxy("get_telemetry", srv.GetTelemetry)
    navigate = rospy.ServiceProxy("navigate", srv.Navigate)
    land = rospy.ServiceProxy("land", Trigger)
    arming_proxy = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

    def arming(value: bool) -> Any:
        return arming_proxy(value)

    return FlightProxies(
        get_telemetry=get_telemetry, navigate=navigate, land=land, arming=arming
    )


def read_map(path: str) -> Dict[int, Tuple[float, float, float]]:
    """Разобрать карту поля в формате aruco_pose/aruco_map.

    Строка: ``id размер x y z угол_z угол_y угол_x`` (см.
    docs.skyris.ru/technic6S/ArucoMap.html). Возвращает ``{id: (x, y, z)}``.
    """
    markers: Dict[int, Tuple[float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(
                    "Некорректная строка {} в карте {}: {!r}".format(
                        line_number, path, raw_line
                    )
                )
            marker_id = int(parts[0])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            markers[marker_id] = (x, y, z)
    return markers


def marker_xy(
    markers: Dict[int, Tuple[float, float, float]], marker_id: int
) -> Tuple[float, float]:
    """Координаты x, y метки по её id (z из карты игнорируется — это высота
    самой метки на полу, а не высота полёта)."""
    if marker_id not in markers:
        raise KeyError("Метка {} отсутствует в карте поля".format(marker_id))
    x, y, _z = markers[marker_id]
    return x, y


def read_rangefinder_ros(timeout: float) -> float:
    """Прочитать дальность с ``/rangefinder/range`` (используется по
    умолчанию в полётных скриптах; в тестах подменяется заглушкой)."""
    import rospy
    from sensor_msgs.msg import Range

    message = rospy.wait_for_message("/rangefinder/range", Range, timeout=timeout)
    return float(message.range)


def _distance_3d(telemetry: Any, x: float, y: float, z: float) -> float:
    dx = float(telemetry.x) - x
    dy = float(telemetry.y) - y
    dz = float(telemetry.z) - z
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def navigate_wait(
    navigate: Callable[..., Any],
    get_telemetry: Callable[..., Any],
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    speed: float = 0.5,
    frame_id: str = "aruco_map",
    auto_arm: bool = False,
    tolerance: float = 0.2,
    timeout: float = 30.0,
    poll_interval: float = 0.2,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
) -> Any:
    """Отправить navigate() и заблокироваться до прибытия в точку.

    По образцу ``navigate_wait`` из docs.skyris.ru/technic6S/CodeExamples.html.
    """
    if tolerance <= 0:
        raise ValueError("tolerance должен быть больше нуля")
    if timeout <= 0:
        raise ValueError("timeout должен быть больше нуля")

    navigate(x=x, y=y, z=z, speed=speed, frame_id=frame_id, auto_arm=auto_arm)

    deadline = time_fn() + timeout
    while True:
        telemetry = get_telemetry(frame_id=frame_id)
        if _distance_3d(telemetry, x, y, z) <= tolerance:
            return telemetry
        if time_fn() >= deadline:
            raise TimeoutError(
                "navigate_wait: цель ({}, {}, {}) не достигнута за {}с".format(
                    x, y, z, timeout
                )
            )
        sleep_fn(poll_interval)


def land_wait(
    land: Callable[[], Any],
    get_telemetry: Callable[..., Any],
    *,
    timeout: float = 15.0,
    poll_interval: float = 0.2,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
) -> Any:
    """Вызвать land() и заблокироваться до дизарма."""
    if timeout <= 0:
        raise ValueError("timeout должен быть больше нуля")

    land()

    deadline = time_fn() + timeout
    while True:
        telemetry = get_telemetry()
        if not getattr(telemetry, "armed", True):
            return telemetry
        if time_fn() >= deadline:
            raise TimeoutError(
                "land_wait: дизарм не подтверждён за {}с".format(timeout)
            )
        sleep_fn(poll_interval)


def wait_until_stable(
    get_telemetry: Callable[..., Any],
    *,
    x: float,
    y: float,
    frame_id: str = "aruco_map",
    tolerance: float = 0.2,
    hold_time: float = 1.5,
    timeout: float = 10.0,
    poll_interval: float = 0.2,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
) -> Any:
    """Дождаться, пока x/y устоятся в пределах допуска ``hold_time`` секунд
    подряд, прежде чем начинать спуск («стабилизация» перед посадкой)."""
    if tolerance <= 0:
        raise ValueError("tolerance должен быть больше нуля")
    if hold_time <= 0:
        raise ValueError("hold_time должен быть больше нуля")
    if timeout <= 0:
        raise ValueError("timeout должен быть больше нуля")

    deadline = time_fn() + timeout
    stable_since: Optional[float] = None
    while True:
        telemetry = get_telemetry(frame_id=frame_id)
        dx = float(telemetry.x) - x
        dy = float(telemetry.y) - y
        now = time_fn()
        if (dx * dx + dy * dy) ** 0.5 <= tolerance:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= hold_time:
                return telemetry
        else:
            stable_since = None
        if now >= deadline:
            raise TimeoutError(
                "wait_until_stable: позиция не стабилизировалась за {}с".format(
                    timeout
                )
            )
        sleep_fn(poll_interval)


def controlled_descent_and_disarm(
    navigate: Callable[..., Any],
    arming: Callable[[bool], Any],
    rangefinder_reader: Callable[[float], float],
    *,
    x: float,
    y: float,
    start_z: float,
    min_z: float,
    step: float = 0.15,
    touchdown_threshold: float = 0.12,
    speed: float = 0.3,
    frame_id: str = "aruco_map",
    step_settle_time: float = 0.6,
    rangefinder_timeout: float = 1.0,
    max_steps: int = 200,
    sleep_fn: Callable[[float], None] = sleep,
) -> DescentResult:
    """Пошаговый спуск с контролем дальномера и ручной дизарм по касанию.

    Не полагается на штатный ``land()``: поведение платформы при посадке на
    приподнятую поверхность (куб зарядной станции) не описано в документации
    Skyris, поэтому высоту опускаем сами и подтверждаем касание показанием
    ``/rangefinder/range``. ``min_z`` — обязательный safety-предел (например,
    высота куба минус небольшой запас): если дальномер за ``max_steps`` так и
    не покажет касание, спуск всё равно остановится на этой высоте, а не
    продолжится бесконечно/до столкновения.
    """
    if step <= 0:
        raise ValueError("step должен быть больше нуля")
    if start_z <= min_z:
        raise ValueError("start_z должен быть больше min_z")
    if max_steps <= 0:
        raise ValueError("max_steps должен быть больше нуля")

    current_z = start_z
    last_range: Optional[float] = None
    touchdown = False

    for _ in range(max_steps):
        navigate(x=x, y=y, z=current_z, speed=speed, frame_id=frame_id)
        sleep_fn(step_settle_time)

        try:
            last_range = rangefinder_reader(rangefinder_timeout)
        except Exception:
            last_range = None

        if last_range is not None and last_range <= touchdown_threshold:
            touchdown = True
            break
        if current_z <= min_z:
            break
        current_z = max(min_z, current_z - step)

    arming(False)
    return DescentResult(
        final_z=current_z, touchdown_detected=touchdown, last_range=last_range
    )


def _self_test() -> int:
    import tempfile
    import os

    # read_map / marker_xy
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# comment line, should be skipped\n")
            handle.write("37 0.33 2.00 5.00 0 0 0 0\n")
            handle.write("48 0.33 6.00 6.00 0 0 0 0\n")
            handle.write("\n")
        markers = read_map(path)
        assert markers[37] == (2.0, 5.0, 0.0)
        assert markers[48] == (6.0, 6.0, 0.0)
        assert marker_xy(markers, 37) == (2.0, 5.0)
        try:
            marker_xy(markers, 99)
        except KeyError:
            pass
        else:
            raise AssertionError("Ожидался KeyError для отсутствующей метки")
    finally:
        os.remove(path)

    # navigate_wait: телеметрия постепенно приближается к цели
    clock = [0.0]

    def fake_time() -> float:
        return clock[0]

    def fake_sleep(duration: float) -> None:
        clock[0] += duration

    navigate_calls = []

    def fake_navigate(**kwargs: object) -> None:
        navigate_calls.append(kwargs)

    telemetry_positions = iter([(5.0, 5.0, 0.0), (1.5, 1.5, 0.0), (1.05, 0.98, 1.5)])

    class FakeTelemetry:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.x, self.y, self.z = x, y, z
            self.armed = True

    def fake_get_telemetry(**_kwargs: object) -> FakeTelemetry:
        x, y, z = next(telemetry_positions)
        return FakeTelemetry(x, y, z)

    result = navigate_wait(
        fake_navigate,
        fake_get_telemetry,
        x=1.0,
        y=1.0,
        z=1.5,
        frame_id="aruco_map",
        auto_arm=True,
        tolerance=0.2,
        timeout=5.0,
        sleep_fn=fake_sleep,
        time_fn=fake_time,
    )
    assert navigate_calls[-1]["auto_arm"] is True
    assert abs(result.x - 1.05) < 1e-9

    # navigate_wait: таймаут, если цель никогда не достигается
    def fake_get_telemetry_far(**_kwargs: object) -> FakeTelemetry:
        return FakeTelemetry(100.0, 100.0, 0.0)

    clock[0] = 0.0
    try:
        navigate_wait(
            fake_navigate,
            fake_get_telemetry_far,
            x=0.0,
            y=0.0,
            z=0.0,
            timeout=1.0,
            poll_interval=0.5,
            sleep_fn=fake_sleep,
            time_fn=fake_time,
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("Ожидался TimeoutError")

    # land_wait: телеметрия сообщает о дизарме после land()
    land_calls = []

    def fake_land() -> None:
        land_calls.append(True)

    armed_sequence = iter([True, True, False])

    def fake_get_telemetry_landing(**_kwargs: object) -> FakeTelemetry:
        telemetry = FakeTelemetry(0.0, 0.0, 0.0)
        telemetry.armed = next(armed_sequence)
        return telemetry

    clock[0] = 0.0
    land_wait(
        fake_land,
        fake_get_telemetry_landing,
        timeout=5.0,
        poll_interval=0.2,
        sleep_fn=fake_sleep,
        time_fn=fake_time,
    )
    assert land_calls == [True]

    # wait_until_stable: должен требовать hold_time подряд в допуске
    clock[0] = 0.0
    stable_positions = iter(
        [(0.0, 0.0), (1.0, 1.0), (0.05, -0.05), (0.02, 0.03), (0.0, 0.0)]
    )

    def fake_get_telemetry_stable(**_kwargs: object) -> FakeTelemetry:
        x, y = next(stable_positions)
        return FakeTelemetry(x, y, 1.5)

    wait_until_stable(
        fake_get_telemetry_stable,
        x=0.0,
        y=0.0,
        tolerance=0.2,
        hold_time=0.6,
        timeout=5.0,
        poll_interval=0.3,
        sleep_fn=fake_sleep,
        time_fn=fake_time,
    )

    # controlled_descent_and_disarm: касание определяется дальномером
    descent_navigate_calls = []

    def fake_descent_navigate(**kwargs: object) -> None:
        descent_navigate_calls.append(kwargs)

    disarm_calls = []

    def fake_arming(value: bool) -> None:
        disarm_calls.append(value)

    ranges = iter([2.0, 1.2, 0.5, 0.1])

    def fake_rangefinder(_timeout: float) -> float:
        return next(ranges)

    descent = controlled_descent_and_disarm(
        fake_descent_navigate,
        fake_arming,
        fake_rangefinder,
        x=2.0,
        y=5.0,
        start_z=1.5,
        min_z=0.7,
        step=0.3,
        touchdown_threshold=0.12,
        sleep_fn=lambda _duration: None,
    )
    assert descent.touchdown_detected is True
    assert descent.last_range == 0.1
    assert disarm_calls == [False]
    assert len(descent_navigate_calls) == 4

    # controlled_descent_and_disarm: дальномер молчит -> останов по min_z
    def fake_rangefinder_broken(_timeout: float) -> float:
        raise RuntimeError("нет данных")

    disarm_calls.clear()
    descent_safety = controlled_descent_and_disarm(
        fake_descent_navigate,
        fake_arming,
        fake_rangefinder_broken,
        x=2.0,
        y=5.0,
        start_z=1.0,
        min_z=0.85,
        step=0.1,
        sleep_fn=lambda _duration: None,
    )
    assert descent_safety.touchdown_detected is False
    assert descent_safety.final_z == 0.85
    assert disarm_calls == [False]

    print("SELF-TEST: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
