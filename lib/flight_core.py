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

import math
import time
from dataclasses import dataclass
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# «Сохранять текущий курс»: в Clover-совместимом API (пакет ``technic``)
# yaw=NaN означает «не разворачивать дрон, удерживать текущий курс», а
# yaw_rate по умолчанию 0 — то есть дрон летит боком/задом, но не крутится.
# Это значение по умолчанию во всех навигационных функциях модуля, потому
# что штатное значение сервиса (yaw=0) в кадре ``aruco_map`` означает
# «развернуться носом по оси X карты» — дрон доворачивался на каждой команде
# navigate(), что видно в полёте как постоянное вращение по курсу.
# Разворот по курсу не нужен ни на одном этапе миссии: камера смотрит вниз,
# метки распознаются при любой ориентации, а лишнее вращение сбивает
# локализацию по aruco_map и раскачивает крупный дрон.
HOLD_YAW = float("nan")


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


def read_visible_marker_ids_ros(timeout: float) -> Tuple[int, ...]:
    """Прочитать ID меток поля, видимых камерой прямо сейчас, с топика
    ``aruco_detect/markers`` (пакет ``aruco_pose``, см. TASK.md).

    Используется, чтобы подтвердить локализацию по ``aruco_map`` реальными
    метками перед навигацией по ней (см. ``wait_for_markers_visible``), а не
    просто число меток, как ``diagnosis.read_marker_count_ros``."""
    import rospy
    from aruco_pose.msg import MarkerArray

    message = rospy.wait_for_message("aruco_detect/markers", MarkerArray, timeout=timeout)
    return tuple(sorted(int(marker.id) for marker in message.markers))


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
    yaw: Optional[float] = HOLD_YAW,
    tolerance: float = 0.2,
    timeout: float = 30.0,
    poll_interval: float = 0.2,
    guard: Optional[Callable[[], None]] = None,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
    verbose: bool = False,
) -> Any:
    """Отправить navigate() и заблокироваться до прибытия в точку.

    По образцу ``navigate_wait`` из docs.skyris.ru/technic6S/CodeExamples.html.
    Прибытие проверяется телеметрией в ``frame_id='navigate_target'`` —
    специальном фрейме, который отдаёт вектор «дрон -> текущая цель
    navigate()» и сходится к (0, 0, 0) по мере приближения, независимо от
    того, в каком ``frame_id`` задавалась сама цель. Проверять это тем же
    ``frame_id``, что передан в ``navigate()``, некорректно для «подвижных»
    фреймов вроде ``body`` (система координат самого дрона) — телеметрия
    дрона относительно самого себя всегда около (0, 0, 0), поэтому цель
    в ней в принципе недостижима и цикл всегда упирается в timeout,
    независимо от реального поведения дрона (воспроизведено на площадке
    2026-08-01 на взлёте с ``frame_id='body'``).

    ``guard`` (если передан) вызывается на каждой итерации ожидания **до**
    проверки прибытия и должен бросить исключение, если продолжать полёт
    опасно (выход за геозону, потеря локализации — см.
    ``flight_nav.geofence_guard``). Это единственный способ прервать уже
    отданную команду navigate(): сам сервис асинхронный и без такого
    контроля дрон летит к цели до таймаута, даже если цель заведомо
    неверна (авария на площадке — вылет за поле).

    ``yaw`` по умолчанию — ``HOLD_YAW`` (NaN, «сохранять текущий курс»), см.
    комментарий у самой константы. Раньше yaw не передавался вовсе, и сервис
    подставлял своё значение по умолчанию (0 в указанном ``frame_id``): в
    ``body`` это безобидное «не менять курс», а в ``aruco_map`` —
    «развернуться носом по оси X карты», из-за чего дрон доворачивал на
    каждой команде navigate() и крутился в полёте. ``yaw=None`` возвращает
    прежнее поведение (не передавать yaw вообще) — это запасной вариант на
    случай, если NaN не поддержан прошивкой (см. ``--no-hold-yaw`` в
    bvs1_flight.py/bvs2_flight.py); числовое значение задаёт курс явно.
    """
    if tolerance <= 0:
        raise ValueError("tolerance должен быть больше нуля")
    if timeout <= 0:
        raise ValueError("timeout должен быть больше нуля")

    if verbose:
        print(
            "[navigate_wait] navigate(x={:.2f}, y={:.2f}, z={:.2f}, speed={}, "
            "frame_id={!r}, auto_arm={}, yaw={})".format(
                x, y, z, speed, frame_id, auto_arm, yaw
            )
        )
    request: Dict[str, Any] = {
        "x": x,
        "y": y,
        "z": z,
        "speed": speed,
        "frame_id": frame_id,
        "auto_arm": auto_arm,
    }
    if yaw is not None:
        request["yaw"] = yaw
    navigate(**request)

    deadline = time_fn() + timeout
    while True:
        if guard is not None:
            guard()
        telemetry = get_telemetry(frame_id="navigate_target")
        distance = _distance_3d(telemetry, 0.0, 0.0, 0.0)
        if verbose:
            print(
                "[navigate_wait] остаток до цели: dx={:.2f} dy={:.2f} dz={:.2f} "
                "(|d|={:.2f}, допуск={:.2f})".format(
                    telemetry.x, telemetry.y, telemetry.z, distance, tolerance
                )
            )
        if distance <= tolerance:
            return telemetry
        if time_fn() >= deadline:
            raise TimeoutError(
                "navigate_wait: цель ({}, {}, {}) не достигнута за {}с".format(
                    x, y, z, timeout
                )
            )
        sleep_fn(poll_interval)


def hover_in_place(
    navigate: Callable[..., Any],
    *,
    speed: float = 0.5,
    verbose: bool = False,
) -> None:
    """Немедленно погасить перемещение: удержание собственной текущей точки.

    ``navigate(x=0, y=0, z=0, frame_id='body')`` — это «цель = там, где я
    сейчас», то есть штатный способ остановиться, не садясь и не дизармя.
    Нужен как первая реакция на любую аварийную ситуацию в полёте (потеря
    локализации, выход за геозону): прервать уже отданную асинхронную команду
    navigate() иначе нечем — новая команда просто заменяет старую цель.

    ``yaw`` здесь сознательно не передаётся (в отличие от ``navigate_wait``):
    в кадре ``body`` значение сервиса по умолчанию (yaw=0) уже означает «не
    менять курс», а аварийный стоп не должен зависеть от того, поддержана ли
    прошивкой запись NaN.
    """
    if verbose:
        print("[hover_in_place] стоп: navigate(0, 0, 0, frame_id='body')")
    navigate(x=0.0, y=0.0, z=0.0, speed=speed, frame_id="body", auto_arm=False)


def land_wait(
    land: Callable[[], Any],
    get_telemetry: Callable[..., Any],
    *,
    timeout: float = 15.0,
    poll_interval: float = 0.2,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
    verbose: bool = False,
) -> Any:
    """Вызвать land() и заблокироваться до дизарма."""
    if timeout <= 0:
        raise ValueError("timeout должен быть больше нуля")

    if verbose:
        print("[land_wait] land()")
    land()

    deadline = time_fn() + timeout
    while True:
        telemetry = get_telemetry()
        armed = getattr(telemetry, "armed", True)
        if verbose:
            print("[land_wait] armed={}".format(armed))
        if not armed:
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
    verbose: bool = False,
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
        offset = (dx * dx + dy * dy) ** 0.5
        if verbose:
            print(
                "[wait_until_stable] offset={:.2f} (допуск={:.2f}, "
                "стабильно={})".format(
                    offset,
                    tolerance,
                    "-" if stable_since is None else "{:.1f}с".format(now - stable_since),
                )
            )
        if offset <= tolerance:
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


def wait_for_markers_visible(
    marker_id_reader: Callable[[float], Sequence[int]],
    *,
    timeout: float = 10.0,
    read_timeout: float = 1.0,
    poll_interval: float = 0.3,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
    verbose: bool = False,
) -> Tuple[int, ...]:
    """Дождаться, пока камера увидит хоть одну метку поля, и вернуть их ID.

    Сразу после взлёта TF ``aruco_map`` может быть ещё не готов или опираться
    на устаревшую/недостоверную оценку позиции — реальный кадр случай на
    площадке: дрон сразу после взлёта полетел на большой скорости вперёд и
    врезался в сетку, ни разу не подтвердив локализацию видимой меткой.
    Поэтому любая навигация в ``frame_id='aruco_map'`` должна идти только
    после того, как эта функция вернула непустой список — до этого доверять
    ``aruco_map`` нельзя.
    """
    if timeout <= 0:
        raise ValueError("timeout должен быть больше нуля")

    deadline = time_fn() + timeout
    while True:
        try:
            marker_ids: Tuple[int, ...] = tuple(marker_id_reader(read_timeout))
        except Exception as exc:
            marker_ids = ()
            if verbose:
                print(
                    "[wait_for_markers_visible] метки не прочитаны ({}: {})".format(
                        type(exc).__name__, exc
                    )
                )
        if verbose:
            print(
                "[wait_for_markers_visible] видимые метки поля: {}".format(
                    list(marker_ids)
                )
            )
        if marker_ids:
            return marker_ids
        if time_fn() >= deadline:
            raise TimeoutError(
                "wait_for_markers_visible: ни одной метки поля не видно за {}с - "
                "aruco_map недостоверен".format(timeout)
            )
        sleep_fn(poll_interval)


def nearest_marker_id(
    markers: Dict[int, Tuple[float, float, float]],
    candidate_ids: Sequence[int],
    x: float,
    y: float,
) -> int:
    """ID метки из ``candidate_ids``, ближайшей к (x, y) по карте поля.

    Используется, чтобы выбрать метку для привязки/стабилизации локализации
    среди реально видимых камерой ID (см. ``wait_for_markers_visible``), а не
    полагаться на первую попавшуюся - она может быть на другом конце поля,
    если по карте случайно совпали id."""
    known_candidates = [
        marker_id for marker_id in candidate_ids if marker_id in markers
    ]
    if not known_candidates:
        raise KeyError(
            "Ни одна из видимых меток {} не найдена в карте поля".format(
                list(candidate_ids)
            )
        )

    def _distance_sq(marker_id: int) -> float:
        mx, my, _mz = markers[marker_id]
        return (mx - x) ** 2 + (my - y) ** 2

    return min(known_candidates, key=_distance_sq)


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Целочисленные узлы решётки на прямой от (x0, y0) до (x1, y1) включительно."""
    points: List[Tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def marker_path(
    markers: Dict[int, Tuple[float, float, float]],
    start_id: int,
    goal_id: int,
    *,
    grid_step: float = 1.0,
) -> Tuple[int, ...]:
    """Путь по соседним меткам решётки поля от ``start_id`` до ``goal_id``
    включительно (``start_id`` - первый элемент, ``goal_id`` - последний).

    Поле - равномерная решётка ArUco с шагом ``grid_step`` (1 м, см.
    config/field_map.txt/TASK.md), поэтому путь растеризуется по узлам
    решётки алгоритмом Брезенхэма вместо одного длинного прыжка по
    диагонали: дрон идёт «метка за меткой» и не теряет локализацию по
    aruco_map между стартом и целью (см. docstring wait_for_markers_visible
    про то, к какой аварии на площадке привёл прямой прыжок).

    **Устарело для новых сценариев.** Маршрут по узлам решётки строится один
    раз от метки-якоря и дальше выполняется вслепую: ошибка в выборе якоря
    или расхождение aruco_map с полем разворачивают весь маршрут в сторону -
    так БВС-1 и вышел за пределы поля (авария 2026-08-01, разбор в докстринге
    ``flight_nav``). В ``bvs1_flight.py`` заменено на ``flight_nav.fly_to``:
    прямая к цели, нарезанная на короткие перегоны, с перепроверкой позиции
    перед каждым. Здесь остаётся, пока на новый способ не переведён
    ``bvs2_flight.py``.
    """
    if start_id not in markers:
        raise KeyError("Метка {} отсутствует в карте поля".format(start_id))
    if goal_id not in markers:
        raise KeyError("Метка {} отсутствует в карте поля".format(goal_id))
    if grid_step <= 0:
        raise ValueError("grid_step должен быть больше нуля")

    grid_index: Dict[Tuple[int, int], int] = {}
    for marker_id, (x, y, _z) in markers.items():
        grid_index[(round(x / grid_step), round(y / grid_step))] = marker_id

    start_x, start_y, _start_z = markers[start_id]
    goal_x, goal_y, _goal_z = markers[goal_id]
    gx0, gy0 = round(start_x / grid_step), round(start_y / grid_step)
    gx1, gy1 = round(goal_x / grid_step), round(goal_y / grid_step)

    path_ids: List[int] = []
    for gx, gy in _bresenham_line(gx0, gy0, gx1, gy1):
        marker_id = grid_index.get((gx, gy))
        if marker_id is None:
            raise KeyError(
                "На решётке карты нет метки в узле ({}, {}) на пути {} -> {}".format(
                    gx, gy, start_id, goal_id
                )
            )
        if not path_ids or path_ids[-1] != marker_id:
            path_ids.append(marker_id)
    return tuple(path_ids)


def fly_marker_path(
    navigate: Callable[..., Any],
    get_telemetry: Callable[..., Any],
    markers: Dict[int, Tuple[float, float, float]],
    path: Sequence[int],
    *,
    z: float,
    speed: float,
    yaw: Optional[float] = HOLD_YAW,
    navigate_tolerance: float = 0.2,
    navigate_timeout: float = 30.0,
    stabilize_tolerance: float = 0.2,
    stabilize_hold: float = 0.5,
    stabilize_timeout: float = 5.0,
    marker_id_reader: Optional[Callable[[float], Sequence[int]]] = None,
    markers_visible_timeout: float = 10.0,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
    verbose: bool = False,
) -> None:
    """Пролететь путь меток ``path`` (см. ``marker_path``) от второго элемента
    до последнего, стабилизируясь над каждой меткой перед следующим перегоном.

    **Устарело для новых сценариев** вместе с ``marker_path`` — см. примечание
    там же и ``flight_nav.fly_to``. Используется только ``bvs2_flight.py``.

    Дрон крупный: резкий перелёт большим прыжком (пропуская промежуточные
    метки) непредсказуем так же, как один длинный прыжок через всё поле -
    поэтому каждый перегон короткий (один шаг решётки) и завершается
    стабилизацией, а не только прибытием в допуск ``navigate_wait``.

    Если передан ``marker_id_reader``, после стабилизации на каждой
    промежуточной метке дополнительно подтверждается, что камера всё ещё
    видит хоть одну метку поля (``wait_for_markers_visible``) - без этого
    видимость проверялась только один раз сразу после взлёта, а дальше
    ``aruco_map`` мог стать недостоверным (например, из-за потери видимости
    в пути) незамеченным до самого прибытия. ``marker_id_reader=None``
    сохраняет прежнее поведение без проверки (например, для самотестов, где
    это не нужно).
    """
    for marker_id in path[1:]:
        leg_x, leg_y = marker_xy(markers, marker_id)
        navigate_wait(
            navigate,
            get_telemetry,
            x=leg_x,
            y=leg_y,
            z=z,
            speed=speed,
            frame_id="aruco_map",
            yaw=yaw,
            tolerance=navigate_tolerance,
            timeout=navigate_timeout,
            sleep_fn=sleep_fn,
            time_fn=time_fn,
            verbose=verbose,
        )
        wait_until_stable(
            get_telemetry,
            x=leg_x,
            y=leg_y,
            frame_id="aruco_map",
            tolerance=stabilize_tolerance,
            hold_time=stabilize_hold,
            timeout=stabilize_timeout,
            sleep_fn=sleep_fn,
            time_fn=time_fn,
            verbose=verbose,
        )
        if marker_id_reader is not None:
            wait_for_markers_visible(
                marker_id_reader,
                timeout=markers_visible_timeout,
                sleep_fn=sleep_fn,
                time_fn=time_fn,
                verbose=verbose,
            )


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
    yaw: Optional[float] = HOLD_YAW,
    step_settle_time: float = 0.6,
    rangefinder_timeout: float = 1.0,
    max_steps: int = 200,
    sleep_fn: Callable[[float], None] = sleep,
    verbose: bool = False,
) -> DescentResult:
    """Пошаговый спуск с контролем дальномера и ручной дизарм по касанию.

    Не полагается на штатный ``land()``: поведение платформы при посадке на
    приподнятую поверхность (куб зарядной станции) не описано в документации
    Skyris, поэтому высоту опускаем сами и подтверждаем касание показанием
    ``/rangefinder/range``. ``min_z`` — обязательный safety-предел (например,
    высота куба минус небольшой запас): если дальномер за ``max_steps`` так и
    не покажет касание, спуск всё равно остановится на этой высоте, а не
    продолжится бесконечно/до столкновения.

    ``yaw`` — как в ``navigate_wait`` (по умолчанию ``HOLD_YAW``): доворот по
    курсу прямо над кубом смещает дрон в горизонте и грозит штрафом за ножки
    за границей метки, поэтому курс на спуске не меняется.
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

    descent_request: Dict[str, Any] = {
        "speed": speed,
        "frame_id": frame_id,
    }
    if yaw is not None:
        descent_request["yaw"] = yaw

    for step_number in range(max_steps):
        navigate(x=x, y=y, z=current_z, **descent_request)
        sleep_fn(step_settle_time)

        try:
            last_range = rangefinder_reader(rangefinder_timeout)
        except Exception as exc:
            last_range = None
            if verbose:
                print(
                    "[controlled_descent] шаг {}: z={:.2f}, дальномер недоступен ({})".format(
                        step_number, current_z, exc
                    )
                )
        else:
            if verbose:
                print(
                    "[controlled_descent] шаг {}: z={:.2f}, дальность={:.2f} "
                    "(порог касания={:.2f})".format(
                        step_number, current_z, last_range, touchdown_threshold
                    )
                )

        if last_range is not None and last_range <= touchdown_threshold:
            touchdown = True
            break
        if current_z <= min_z:
            break
        current_z = max(min_z, current_z - step)

    if verbose:
        print(
            "[controlled_descent] итог: z={:.2f}, касание={}, дизарм".format(
                current_z, touchdown
            )
        )
    arming(False)
    return DescentResult(
        final_z=current_z, touchdown_detected=touchdown, last_range=last_range
    )


def simulate_charging(
    set_led_fn: Callable[[str, Any], None],
    *,
    total_s: float,
    green_before_takeoff_s: float,
    sleep_fn: Callable[[float], None] = sleep,
) -> None:
    """Имитация зарядки по TASK.md (алгоритм, шаги 3 и 6): красная мигающая
    лента, а за ``green_before_takeoff_s`` секунд до взлёта — зелёная лента
    (не мигающая — TASK.md называет её "зелёная лента" в отличие от явно
    "зелёного мигающего" сигнала возврата на старт).

    Общая для ``bvs1_flight.py`` и ``bvs2_flight.py`` (шаг «имитация
    зарядки» в обоих сценариях делает ровно это), поэтому вынесена сюда, а
    не продублирована в каждом скрипте. ``set_led_fn`` принимается явным
    аргументом (обычно ``led_interface.set_led``), а не импортируется
    напрямую — ``flight_core`` не должен знать про LED-модуль.
    """
    if total_s < 0:
        raise ValueError("total_s не может быть отрицательным")
    if green_before_takeoff_s < 0:
        raise ValueError("green_before_takeoff_s не может быть отрицательным")

    set_led_fn("blink", "red")
    hold_s = total_s - green_before_takeoff_s
    if hold_s > 0:
        sleep_fn(hold_s)
    set_led_fn("solid", "green")
    sleep_fn(green_before_takeoff_s)


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

    # Телеметрия в frame_id='navigate_target' - это остаток пути до цели,
    # сходящийся к (0, 0, 0) по мере приближения (см. docstring navigate_wait
    # про то, почему проверять нужно именно этот фрейм, а не тот, что был
    # передан в navigate()).
    navtarget_positions = iter([(4.0, 4.0, 1.5), (0.5, 0.5, 1.5), (0.05, -0.02, 0.0)])

    class FakeTelemetry:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.x, self.y, self.z = x, y, z
            self.armed = True

    def fake_get_telemetry(**_kwargs: object) -> FakeTelemetry:
        x, y, z = next(navtarget_positions)
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
    # navigate() сам вызывается с исходным (не navigate_target) frame_id
    assert navigate_calls[-1]["x"] == 1.0 and navigate_calls[-1]["frame_id"] == "aruco_map"
    assert navigate_calls[-1]["auto_arm"] is True
    # а результат navigate_wait - это последняя телеметрия в navigate_target
    assert abs(result.x - 0.05) < 1e-9 and abs(result.y - (-0.02)) < 1e-9

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

    # navigate_wait: по умолчанию курс удерживается (yaw=NaN), иначе дрон
    # доворачивает носом по оси X карты на каждой команде и крутится в полёте
    assert math.isnan(navigate_calls[-1]["yaw"])
    clock[0] = 0.0
    navigate_wait(
        fake_navigate,
        lambda **_kwargs: FakeTelemetry(0.0, 0.0, 0.0),
        x=1.0,
        y=1.0,
        z=1.5,
        yaw=None,
        timeout=1.0,
        sleep_fn=fake_sleep,
        time_fn=fake_time,
    )
    # yaw=None - запасной режим «не передавать yaw вовсе» (--no-hold-yaw)
    assert "yaw" not in navigate_calls[-1]
    clock[0] = 0.0
    navigate_wait(
        fake_navigate,
        lambda **_kwargs: FakeTelemetry(0.0, 0.0, 0.0),
        x=1.0,
        y=1.0,
        z=1.5,
        yaw=1.57,
        timeout=1.0,
        sleep_fn=fake_sleep,
        time_fn=fake_time,
    )
    assert navigate_calls[-1]["yaw"] == 1.57

    # navigate_wait: guard прерывает уже отданную команду navigate() - без
    # этого дрон летит к неверной цели до самого таймаута (авария на площадке)
    class GuardTripped(RuntimeError):
        pass

    guard_calls = [0]

    def tripping_guard() -> None:
        guard_calls[0] += 1
        if guard_calls[0] >= 2:
            raise GuardTripped("сторож сработал")

    clock[0] = 0.0
    try:
        navigate_wait(
            fake_navigate,
            fake_get_telemetry_far,  # цель никогда не достигается
            x=0.0,
            y=0.0,
            z=0.0,
            timeout=30.0,
            poll_interval=0.2,
            guard=tripping_guard,
            sleep_fn=fake_sleep,
            time_fn=fake_time,
        )
    except GuardTripped:
        pass
    else:
        raise AssertionError("Ожидался GuardTripped - guard должен прерывать ожидание")
    assert clock[0] < 30.0  # прервались раньше таймаута, а не после него

    # hover_in_place: удержание собственной точки, без посадки и дизарма
    hover_calls: List[Dict[str, Any]] = []
    hover_in_place(lambda **kwargs: hover_calls.append(kwargs))
    assert hover_calls == [
        {"x": 0.0, "y": 0.0, "z": 0.0, "speed": 0.5, "frame_id": "body", "auto_arm": False}
    ]

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

    # wait_for_markers_visible: ждёт, пока появится непустой список меток
    clock[0] = 0.0
    marker_reads = iter([(), (), (3, 7)])

    def fake_marker_reader(_timeout: float) -> Tuple[int, ...]:
        return next(marker_reads)

    seen = wait_for_markers_visible(
        fake_marker_reader,
        timeout=5.0,
        poll_interval=0.5,
        sleep_fn=fake_sleep,
        time_fn=fake_time,
    )
    assert seen == (3, 7)

    # wait_for_markers_visible: таймаут, если меток так и не видно
    clock[0] = 0.0
    try:
        wait_for_markers_visible(
            lambda _timeout: (),
            timeout=1.0,
            poll_interval=0.5,
            sleep_fn=fake_sleep,
            time_fn=fake_time,
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("Ожидался TimeoutError")

    # nearest_marker_id: выбирает ближайшую к (x, y) метку среди видимых
    markers_for_nearest = {1: (0.0, 0.0, 0.0), 2: (5.0, 0.0, 0.0), 3: (5.0, 5.0, 0.0)}
    assert nearest_marker_id(markers_for_nearest, [1, 2, 3], 4.5, 0.5) == 2
    try:
        nearest_marker_id(markers_for_nearest, [99], 0.0, 0.0)
    except KeyError:
        pass
    else:
        raise AssertionError("Ожидался KeyError - ни одна видимая метка не на карте")

    # marker_path: растеризует путь по решётке 7x7, не перепрыгивая метки
    grid_markers: Dict[int, Tuple[float, float, float]] = {}
    for row in range(7):
        for col in range(7):
            grid_markers[row * 7 + col] = (float(col), float(row), 0.0)

    path = marker_path(grid_markers, 48, 37)  # (6,6) -> (2,5), не одним прыжком
    assert path == (48, 47, 39, 38, 37)
    try:
        marker_path(grid_markers, 999, 37)
    except KeyError:
        pass
    else:
        raise AssertionError("Ожидался KeyError для отсутствующей стартовой метки")

    # fly_marker_path: летит по каждому перегону пути и стабилизируется на
    # каждой метке (не только прибывает в допуск navigate_wait) - "телепорт"
    # по тому же принципу, что FakeFlight в flight_test_support.py.
    fly_navigate_calls = []
    fly_stable_reads = 0
    fly_state = {"x": 0.0, "y": 0.0, "z": 0.0}

    def fake_fly_navigate(*, x: float, y: float, z: float, **_kwargs: object) -> None:
        fly_navigate_calls.append({"x": x, "y": y})
        fly_state["x"], fly_state["y"], fly_state["z"] = x, y, z

    def fake_fly_get_telemetry(*, frame_id: str = "aruco_map", **_kwargs: object) -> FakeTelemetry:
        nonlocal fly_stable_reads
        if frame_id == "navigate_target":
            return FakeTelemetry(0.0, 0.0, 0.0)  # мгновенное прибытие
        fly_stable_reads += 1
        return FakeTelemetry(fly_state["x"], fly_state["y"], fly_state["z"])

    clock[0] = 0.0
    fly_marker_path(
        fake_fly_navigate,
        fake_fly_get_telemetry,
        grid_markers,
        path,
        z=1.5,
        speed=0.3,
        stabilize_hold=0.3,
        stabilize_timeout=2.0,
        sleep_fn=fake_sleep,
        time_fn=fake_time,
    )
    assert len(fly_navigate_calls) == len(path) - 1  # первый элемент пути - старт, не перегон
    assert (fly_navigate_calls[-1]["x"], fly_navigate_calls[-1]["y"]) == grid_markers[37][:2]
    assert fly_stable_reads > 0  # действительно стабилизировался на каждой метке, а не просто прилетел

    # fly_marker_path: с marker_id_reader подтверждает видимость меток на
    # каждой промежуточной остановке, а не только сразу после взлёта.
    fly_navigate_calls.clear()
    fly_state["x"], fly_state["y"], fly_state["z"] = 0.0, 0.0, 0.0
    marker_reader_calls = 0

    def fake_marker_reader_always_visible(_timeout: float) -> Tuple[int, ...]:
        nonlocal marker_reader_calls
        marker_reader_calls += 1
        return (1,)

    clock[0] = 0.0
    fly_marker_path(
        fake_fly_navigate,
        fake_fly_get_telemetry,
        grid_markers,
        path,
        z=1.5,
        speed=0.3,
        stabilize_hold=0.3,
        stabilize_timeout=2.0,
        marker_id_reader=fake_marker_reader_always_visible,
        sleep_fn=fake_sleep,
        time_fn=fake_time,
    )
    assert marker_reader_calls == len(path) - 1  # проверка на каждом перегоне, не только один раз

    # fly_marker_path: если метки пропадают из виду в пути, останавливается
    # ошибкой вместо того, чтобы долететь до цели вслепую.
    clock[0] = 0.0
    try:
        fly_marker_path(
            fake_fly_navigate,
            fake_fly_get_telemetry,
            grid_markers,
            path,
            z=1.5,
            speed=0.3,
            stabilize_hold=0.3,
            stabilize_timeout=2.0,
            marker_id_reader=lambda _timeout: (),
            markers_visible_timeout=1.0,
            sleep_fn=fake_sleep,
            time_fn=fake_time,
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("Ожидался TimeoutError - метки пропали из виду в пути")

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

    # simulate_charging: красный на всё время, кроме последних N секунд —
    # зелёный; суммарное время сна равно total_s
    led_calls = []

    def fake_set_led(pattern: str, color: object) -> None:
        led_calls.append((pattern, color))

    sleep_calls = []

    def recording_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    simulate_charging(
        fake_set_led,
        total_s=15.0,
        green_before_takeoff_s=5.0,
        sleep_fn=recording_sleep,
    )
    assert led_calls == [("blink", "red"), ("solid", "green")]
    assert sleep_calls == [10.0, 5.0]

    # simulate_charging: green_before_takeoff_s >= total_s -> без красной фазы
    led_calls.clear()
    sleep_calls.clear()
    simulate_charging(
        fake_set_led,
        total_s=5.0,
        green_before_takeoff_s=5.0,
        sleep_fn=recording_sleep,
    )
    assert led_calls == [("blink", "red"), ("solid", "green")]
    assert sleep_calls == [5.0]  # красная фаза пропущена (hold_s <= 0)

    print("SELF-TEST: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
