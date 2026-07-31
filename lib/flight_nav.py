#!/usr/bin/env python3
"""Безопасное перемещение по полю: геозона, контроль локализации, перелёты.

Появился после аварии на площадке: БВС-1 взлетел, «пошёл по меткам» в
сторону станции, но фактическая траектория увела его за пределы поля и он
упал. Разбор причин и что изменено:

1. **Путь строился по узлам решётки ArUco** (``flight_core.marker_path``,
   растеризация Брезенхэмом), а не по прямой к цели. Любая ошибка в выборе
   стартовой метки-якоря разворачивала весь маршрут в сторону — дрон
   честно летел «по меткам», но не туда, куда нужно. Здесь маршрут
   строится **от фактической позиции к цели по прямой** и нарезается на
   короткие перегоны (``plan_legs``), а промежуточные метки — просто то,
   что под дроном оказывается по дороге, а не узлы маршрута.
2. **Маршрут считался один раз** — от точки старта, и дальше выполнялся
   вслепую. Здесь перед каждым перегоном позиция читается заново
   (``fly_to``), то есть контур замкнут: накопленная ошибка не уводит
   дрон, а компенсируется на следующем шаге.
3. **Никто не проверял, что локализация вообще правдоподобна.** Проверка
   «камера видит хоть одну метку» это не ловит: ``aruco_map`` может
   уверенно отдавать позицию с другого конца поля. Здесь позиция
   сверяется с реально видимыми метками (``read_localization``): если
   ``aruco_map`` утверждает, что дрон в 4 метрах от метки, которую камера
   прямо сейчас видит под собой, — локализации нельзя верить и лететь по
   ней нельзя.
4. **Команду navigate() ничто не могло прервать.** Сервис асинхронный:
   отдав неверную цель, скрипт просто ждал прибытия до таймаута, пока
   дрон вылетал за сетку. Здесь на каждый перегон вешается ``guard``
   (``geofence_guard``), который на каждой итерации ожидания проверяет
   геозону и наличие позиции и бросает исключение, — а полётный сценарий
   по этому исключению гасит движение (``flight_core.hover_in_place``) и
   садится.

Модуль не импортирует ``rospy``: все ROS-вызовы приходят прокси-функциями
(см. ``flight_core.FlightProxies``), поэтому логика проверяется самотестом
на десктопе::

    python3 lib/flight_nav.py --self-test
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import flight_core as fc


class NavigationAbort(RuntimeError):
    """Полёт прерван защитой — продолжать миссию нельзя."""


class GeofenceBreach(NavigationAbort):
    """Дрон вышел (или собирается выйти) за пределы разрешённой зоны."""


class LocalizationLost(NavigationAbort):
    """``aruco_map`` недоступен или расходится с видимыми метками."""


class NavigationStalled(NavigationAbort):
    """Перегоны выполняются, но дрон не приближается к цели."""


@dataclass(frozen=True)
class Geofence:
    """Разрешённая зона полёта в координатах ``aruco_map``.

    Считается от карты поля с запасом ``margin``: поле 7x7 с шагом 1 м даёт
    x, y в [0, 6], запас 0.5 м — это половина шага решётки, дальше дрон
    уже над зоной, где меток нет и локализация деградирует. Верхняя граница
    ``max_z`` — не «потолок сетки» (≥4 м по регламенту), а рабочая высота с
    запасом: подниматься выше в этой миссии незачем.
    """

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float = 0.0
    max_z: float = 3.0

    @classmethod
    def from_markers(
        cls,
        markers: Dict[int, Tuple[float, float, float]],
        *,
        margin: float = 0.5,
        min_z: float = 0.0,
        max_z: float = 3.0,
    ) -> "Geofence":
        if not markers:
            raise ValueError("Карта поля пуста — геозону построить не из чего")
        xs = [x for x, _y, _z in markers.values()]
        ys = [y for _x, y, _z in markers.values()]
        return cls(
            min_x=min(xs) - margin,
            max_x=max(xs) + margin,
            min_y=min(ys) - margin,
            max_y=max(ys) + margin,
            min_z=min_z,
            max_z=max_z,
        )

    def contains(self, x: float, y: float, z: Optional[float] = None) -> bool:
        if math.isnan(x) or math.isnan(y):
            return False
        if not (self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y):
            return False
        if z is not None and not math.isnan(z):
            if not (self.min_z <= z <= self.max_z):
                return False
        return True

    def clamp(self, x: float, y: float) -> Tuple[float, float]:
        """Ближайшая точка внутри зоны — для «подрезки» цели перегона."""
        return (
            min(max(x, self.min_x), self.max_x),
            min(max(y, self.min_y), self.max_y),
        )

    def describe(self) -> str:
        return "x∈[{:.2f}, {:.2f}], y∈[{:.2f}, {:.2f}], z∈[{:.2f}, {:.2f}]".format(
            self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z
        )


@dataclass(frozen=True)
class LocalizationStatus:
    """Состояние локализации в момент проверки (``read_localization``)."""

    ok: bool
    reason: str
    x: float
    y: float
    z: float
    yaw: float
    visible_ids: Tuple[int, ...]
    nearest_id: Optional[int]
    marker_distance: Optional[float]
    expected_error: Optional[float]

    def describe(self) -> str:
        return (
            "позиция=({:.2f}, {:.2f}, {:.2f}), видимые метки={}, ближайшая={}, "
            "расхождение с ней={}, {}".format(
                self.x,
                self.y,
                self.z,
                list(self.visible_ids),
                self.nearest_id,
                "-" if self.marker_distance is None else "{:.2f}м".format(self.marker_distance),
                "OK" if self.ok else "НЕ ОК: " + self.reason,
            )
        )


def read_localization(
    get_telemetry: Callable[..., Any],
    marker_id_reader: Callable[[float], Sequence[int]],
    markers: Dict[int, Tuple[float, float, float]],
    *,
    expected_xy: Optional[Tuple[float, float]] = None,
    expected_tolerance: float = 1.0,
    max_marker_distance: float = 2.5,
    read_timeout: float = 1.0,
    verbose: bool = False,
) -> LocalizationStatus:
    """Проверить, что позиции по ``aruco_map`` можно верить прямо сейчас.

    Проверяется по порядку:

    1. камера видит хоть одну метку поля (иначе ``aruco_map`` работает по
       устаревшей оценке — именно так дрон и улетел в сетку);
    2. телеметрия в ``aruco_map`` вообще отдаёт числа (не NaN — так
       платформа сообщает «фрейма нет»);
    3. **позиция согласована с тем, что видно**: расстояние от заявленной
       позиции до ближайшей из видимых меток не больше
       ``max_marker_distance``. Камера смотрит вниз, поэтому метка в кадре
       физически не может быть далеко — если ``aruco_map`` говорит, что
       может, значит он врёт. Порог зависит от высоты и угла обзора
       камеры: 2.5 м — начальное значение для крейсерских 1.5–2 м,
       калибруется на площадке (``--max-marker-distance``);
    4. если задан ``expected_xy`` — позиция совпадает с ожидаемой (например,
       с координатами стартовой метки сразу после взлёта) в пределах
       ``expected_tolerance``.

    Исключений не бросает: возвращает статус, а решение (ждать, садиться,
    лететь) принимает вызывающий — ``wait_for_localization``/``fly_to``.
    """
    try:
        visible_ids: Tuple[int, ...] = tuple(sorted(int(i) for i in marker_id_reader(read_timeout)))
    except Exception as exc:  # noqa: BLE001 — любая ошибка чтения = меток не видно
        visible_ids = ()
        if verbose:
            print(
                "[read_localization] метки не прочитаны ({}: {})".format(
                    type(exc).__name__, exc
                )
            )

    try:
        telemetry = get_telemetry(frame_id="aruco_map")
        x = float(telemetry.x)
        y = float(telemetry.y)
        z = float(getattr(telemetry, "z", float("nan")))
        yaw = float(getattr(telemetry, "yaw", 0.0))
    except Exception as exc:  # noqa: BLE001
        return LocalizationStatus(
            ok=False,
            reason="телеметрия aruco_map недоступна ({}: {})".format(
                type(exc).__name__, exc
            ),
            x=float("nan"),
            y=float("nan"),
            z=float("nan"),
            yaw=0.0,
            visible_ids=visible_ids,
            nearest_id=None,
            marker_distance=None,
            expected_error=None,
        )

    def _status(ok: bool, reason: str, nearest_id: Optional[int],
                marker_distance: Optional[float],
                expected_error: Optional[float]) -> LocalizationStatus:
        status = LocalizationStatus(
            ok=ok,
            reason=reason,
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            visible_ids=visible_ids,
            nearest_id=nearest_id,
            marker_distance=marker_distance,
            expected_error=expected_error,
        )
        if verbose:
            print("[read_localization] {}".format(status.describe()))
        return status

    if not visible_ids:
        return _status(False, "камера не видит ни одной метки поля", None, None, None)
    if math.isnan(x) or math.isnan(y):
        return _status(False, "aruco_map не отдаёт позицию (NaN)", None, None, None)

    known_ids = [marker_id for marker_id in visible_ids if marker_id in markers]
    if not known_ids:
        return _status(
            False,
            "видимые метки {} отсутствуют в карте поля".format(list(visible_ids)),
            None,
            None,
            None,
        )

    nearest_id = fc.nearest_marker_id(markers, known_ids, x, y)
    nearest_x, nearest_y = fc.marker_xy(markers, nearest_id)
    marker_distance = math.hypot(nearest_x - x, nearest_y - y)
    if marker_distance > max_marker_distance:
        return _status(
            False,
            "aruco_map расходится с видимыми метками: до ближайшей видимой "
            "метки {} по карте {:.2f}м > {:.2f}м".format(
                nearest_id, marker_distance, max_marker_distance
            ),
            nearest_id,
            marker_distance,
            None,
        )

    expected_error: Optional[float] = None
    if expected_xy is not None:
        expected_error = math.hypot(expected_xy[0] - x, expected_xy[1] - y)
        if expected_error > expected_tolerance:
            return _status(
                False,
                "позиция ({:.2f}, {:.2f}) расходится с ожидаемой ({:.2f}, {:.2f}) "
                "на {:.2f}м > {:.2f}м".format(
                    x, y, expected_xy[0], expected_xy[1], expected_error,
                    expected_tolerance,
                ),
                nearest_id,
                marker_distance,
                expected_error,
            )

    return _status(True, "локализация подтверждена", nearest_id, marker_distance, expected_error)


def wait_for_localization(
    get_telemetry: Callable[..., Any],
    marker_id_reader: Callable[[float], Sequence[int]],
    markers: Dict[int, Tuple[float, float, float]],
    *,
    timeout: float = 8.0,
    poll_interval: float = 0.3,
    expected_xy: Optional[Tuple[float, float]] = None,
    expected_tolerance: float = 1.0,
    max_marker_distance: float = 2.5,
    read_timeout: float = 1.0,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
    verbose: bool = False,
) -> LocalizationStatus:
    """Дождаться достоверной локализации (``read_localization``) или бросить
    ``LocalizationLost``.

    Метки могут пропасть из кадра на секунду (наклон при торможении, блик),
    поэтому первый неудачный опрос — не повод прекращать миссию; но если за
    ``timeout`` локализация так и не подтвердилась, лететь по ``aruco_map``
    нельзя ни в каком виде.
    """
    if timeout <= 0:
        raise ValueError("timeout должен быть больше нуля")

    deadline = time_fn() + timeout
    last: Optional[LocalizationStatus] = None
    while True:
        last = read_localization(
            get_telemetry,
            marker_id_reader,
            markers,
            expected_xy=expected_xy,
            expected_tolerance=expected_tolerance,
            max_marker_distance=max_marker_distance,
            read_timeout=read_timeout,
            verbose=verbose,
        )
        if last.ok:
            return last
        if time_fn() >= deadline:
            raise LocalizationLost(
                "локализация не подтверждена за {}с: {}".format(timeout, last.reason)
            )
        sleep_fn(poll_interval)


def geofence_guard(
    get_telemetry: Callable[..., Any],
    fence: Geofence,
    *,
    max_blind_time: float = 2.0,
    time_fn: Callable[[], float] = time.monotonic,
    verbose: bool = False,
) -> Callable[[], None]:
    """Собрать функцию-сторож для ``flight_core.navigate_wait(guard=...)``.

    Вызывается на каждой итерации ожидания прибытия (~5 раз в секунду) и
    бросает:

    * ``GeofenceBreach`` — позиция вышла за ``fence``. Это последний рубеж:
      если дрон уже за границей поля, дальше сетка;
    * ``LocalizationLost`` — позиции нет (NaN/ошибка сервиса) дольше
      ``max_blind_time``. Короткие провалы допускаются: при потере TF на
      доли секунды дрон продолжает лететь к прежней цели, и это нормально,
      а вот несколько секунд вслепую на скорости — это уже полёт в
      неизвестность.

    Намеренно не читает видимость меток: ``aruco_detect/markers`` читается
    блокирующим ``wait_for_message`` и на такой частоте опроса тормозил бы
    сам цикл ожидания. Согласованность позиции с метками проверяется между
    перегонами (``read_localization``), где это дёшево.
    """
    blind_since: List[Optional[float]] = [None]

    def guard() -> None:
        try:
            telemetry = get_telemetry(frame_id="aruco_map")
            x, y = float(telemetry.x), float(telemetry.y)
            z = float(getattr(telemetry, "z", float("nan")))
        except Exception:  # noqa: BLE001 — сервис мог не ответить, это не сразу авария
            x = y = z = float("nan")

        if math.isnan(x) or math.isnan(y):
            now = time_fn()
            if blind_since[0] is None:
                blind_since[0] = now
            elif now - blind_since[0] >= max_blind_time:
                raise LocalizationLost(
                    "позиция в aruco_map недоступна {:.1f}с в полёте".format(
                        now - blind_since[0]
                    )
                )
            return

        blind_since[0] = None
        if not fence.contains(x, y, z):
            raise GeofenceBreach(
                "дрон вне геозоны: позиция ({:.2f}, {:.2f}, {:.2f}), зона {}".format(
                    x, y, z, fence.describe()
                )
            )
        if verbose:
            print("[geofence_guard] позиция ({:.2f}, {:.2f}, {:.2f}) в зоне".format(x, y, z))

    return guard


def plan_legs(
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    *,
    leg_length: float = 1.0,
) -> List[Tuple[float, float]]:
    """Разбить прямую «текущая позиция -> цель» на перегоны не длиннее
    ``leg_length`` (последняя точка — ровно цель).

    Это замена ``flight_core.marker_path``: маршрут ведёт **к цели**, а не
    по узлам решётки меток. Дрон крупный, поэтому длинный перегон опасен
    сам по себе (разгон, наклон, потеря меток из кадра) — но дробить его
    надо по прямой, а не заворачивать по решётке.
    """
    if leg_length <= 0:
        raise ValueError("leg_length должен быть больше нуля")

    dx = goal_xy[0] - start_xy[0]
    dy = goal_xy[1] - start_xy[1]
    distance = math.hypot(dx, dy)
    if distance <= leg_length:
        return [(goal_xy[0], goal_xy[1])]

    steps = int(math.ceil(distance / leg_length))
    return [
        (start_xy[0] + dx * i / steps, start_xy[1] + dy * i / steps)
        for i in range(1, steps + 1)
    ]


def map_vector_to_body(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    """Перевести смещение из координат ``aruco_map`` в координаты ``body``
    (система самого дрона) при текущем курсе ``yaw``.

    Нужно для режима ``relative`` в ``fly_to``: команда «пролететь столько-то
    вперёд/вбок» не зависит от того, дёргается ли TF карты в этот момент,
    тогда как цель в ``aruco_map`` пересчитывается платформой непрерывно и
    прыгает вместе с ней.
    """
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return dx * cos_yaw + dy * sin_yaw, -dx * sin_yaw + dy * cos_yaw


@dataclass(frozen=True)
class FlightLimits:
    """Пороги перелёта — вынесены отдельно, чтобы не тащить 15 аргументов
    через каждый вызов ``fly_to`` в полётных сценариях."""

    leg_length_m: float = 1.0
    speed: float = 0.4
    arrival_tolerance_m: float = 0.25
    navigate_tolerance_m: float = 0.25
    navigate_timeout_s: float = 20.0
    stabilize_tolerance_m: float = 0.3
    stabilize_hold_s: float = 0.7
    stabilize_timeout_s: float = 8.0
    settle_s: float = 1.0
    localization_timeout_s: float = 8.0
    localization_read_timeout_s: float = 1.0
    max_marker_distance_m: float = 2.5
    max_blind_time_s: float = 2.0
    # Режим 'relative': насколько дрону разрешено скорректировать высоту за
    # один перегон. Высота там задаётся приращением, поэтому расхождение с
    # крейсерской гасится постепенно — рывок вверх/вниз на ошибочных данных
    # карты опаснее, чем лишний перегон с неточной высотой.
    max_climb_per_leg_m: float = 0.5
    max_legs: int = 40
    min_progress_m: float = 0.15
    stall_legs: int = 3


def fly_to(
    navigate: Callable[..., Any],
    get_telemetry: Callable[..., Any],
    marker_id_reader: Callable[[float], Sequence[int]],
    markers: Dict[int, Tuple[float, float, float]],
    fence: Geofence,
    *,
    target_x: float,
    target_y: float,
    z: float,
    mode: str = "map",
    limits: FlightLimits = FlightLimits(),
    yaw: Optional[float] = None,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
    verbose: bool = False,
) -> LocalizationStatus:
    """Долететь до точки ``(target_x, target_y)`` короткими перегонами с
    перепроверкой локализации перед каждым.

    Замыкает контур, которого не было в прежнем алгоритме: позиция читается
    заново перед каждым перегоном и вектор к цели пересчитывается от
    фактического места, а не от изначального плана. Поэтому промах на одном
    перегоне гасится на следующем, а не накапливается до вылета за поле.

    Режимы (``--nav-mode`` в полётных сценариях):

    * ``map`` — перегон задаётся абсолютной точкой в ``frame_id='aruco_map'``.
      Точнее всего, пока карта достоверна; платформа сама доводит дрон в
      точку.
    * ``relative`` — та же точка, но пересчитанная в смещение и отданная в
      ``frame_id='body'`` («пролететь X вперёд, Y влево»). Карта участвует
      только в расчёте вектора между перегонами, а не в контуре управления,
      поэтому рывки TF не превращаются в рывки дрона. Платой идёт дрейф по
      высоте и курсу — за перегон в 1 м он мал, но проверять надо на
      площадке.

    Возвращает последний подтверждённый ``LocalizationStatus``. Бросает
    ``NavigationAbort`` (``LocalizationLost``/``GeofenceBreach``/
    ``NavigationStalled``) — в этом случае движение уже погашено
    ``hover_in_place``, а решение о посадке принимает полётный сценарий.
    """
    if mode not in ("map", "relative"):
        raise ValueError("Неизвестный режим перелёта: {!r}".format(mode))
    if not fence.contains(target_x, target_y):
        raise GeofenceBreach(
            "цель ({:.2f}, {:.2f}) вне геозоны {} — проверьте --station-marker/"
            "карту поля".format(target_x, target_y, fence.describe())
        )

    guard = geofence_guard(
        get_telemetry,
        fence,
        max_blind_time=limits.max_blind_time_s,
        time_fn=time_fn,
    )

    status: Optional[LocalizationStatus] = None
    best_distance: Optional[float] = None
    legs_without_progress = 0

    for leg_number in range(1, limits.max_legs + 1):
        try:
            status = wait_for_localization(
                get_telemetry,
                marker_id_reader,
                markers,
                timeout=limits.localization_timeout_s,
                max_marker_distance=limits.max_marker_distance_m,
                read_timeout=limits.localization_read_timeout_s,
                sleep_fn=sleep_fn,
                time_fn=time_fn,
                verbose=verbose,
            )
        except LocalizationLost:
            fc.hover_in_place(navigate, speed=limits.speed, verbose=verbose)
            raise

        distance = math.hypot(target_x - status.x, target_y - status.y)
        if verbose:
            print(
                "[fly_to] перегон {}: позиция ({:.2f}, {:.2f}), до цели ({:.2f}, "
                "{:.2f}) {:.2f}м".format(
                    leg_number, status.x, status.y, target_x, target_y, distance
                )
            )
        if distance <= limits.arrival_tolerance_m:
            return status

        if best_distance is None or distance < best_distance - limits.min_progress_m:
            best_distance = distance
            legs_without_progress = 0
        else:
            legs_without_progress += 1
            if legs_without_progress >= limits.stall_legs:
                fc.hover_in_place(navigate, speed=limits.speed, verbose=verbose)
                raise NavigationStalled(
                    "за {} перегонов дрон не приблизился к цели ({:.2f}, {:.2f}): "
                    "остаток {:.2f}м".format(
                        legs_without_progress, target_x, target_y, distance
                    )
                )

        step = min(limits.leg_length_m, distance)
        leg_x = status.x + (target_x - status.x) / distance * step
        leg_y = status.y + (target_y - status.y) / distance * step
        if not fence.contains(leg_x, leg_y):
            # Сама цель внутри зоны (проверено выше), значит за границу
            # вылезла промежуточная точка — «подрезаем» её, а не отменяем
            # перелёт.
            leg_x, leg_y = fence.clamp(leg_x, leg_y)

        try:
            if mode == "map":
                fc.navigate_wait(
                    navigate,
                    get_telemetry,
                    x=leg_x,
                    y=leg_y,
                    z=z,
                    speed=limits.speed,
                    frame_id="aruco_map",
                    yaw=yaw,
                    tolerance=limits.navigate_tolerance_m,
                    timeout=limits.navigate_timeout_s,
                    guard=guard,
                    sleep_fn=sleep_fn,
                    time_fn=time_fn,
                    verbose=verbose,
                )
                fc.wait_until_stable(
                    get_telemetry,
                    x=leg_x,
                    y=leg_y,
                    frame_id="aruco_map",
                    tolerance=limits.stabilize_tolerance_m,
                    hold_time=limits.stabilize_hold_s,
                    timeout=limits.stabilize_timeout_s,
                    sleep_fn=sleep_fn,
                    time_fn=time_fn,
                    verbose=verbose,
                )
            else:
                body_x, body_y = map_vector_to_body(
                    leg_x - status.x, leg_y - status.y, status.yaw
                )
                # В body z — это приращение к текущей высоте, поэтому
                # крейсерская высота выдерживается не одной командой, а
                # порционной коррекцией (не более max_climb_per_leg_m за
                # перегон): так ошибка высоты в карте не превращается в рывок.
                body_z = 0.0
                if not math.isnan(status.z):
                    body_z = max(
                        -limits.max_climb_per_leg_m,
                        min(limits.max_climb_per_leg_m, z - status.z),
                    )
                fc.navigate_wait(
                    navigate,
                    get_telemetry,
                    x=body_x,
                    y=body_y,
                    z=body_z,
                    speed=limits.speed,
                    frame_id="body",
                    tolerance=limits.navigate_tolerance_m,
                    timeout=limits.navigate_timeout_s,
                    guard=guard,
                    sleep_fn=sleep_fn,
                    time_fn=time_fn,
                    verbose=verbose,
                )
                # Стабилизироваться по абсолютной точке нельзя — в этом режиме
                # мы сознательно не верим карте в контуре управления; просто
                # даём дрону успокоиться перед следующим замером.
                sleep_fn(limits.settle_s)
        except (TimeoutError, NavigationAbort):
            fc.hover_in_place(navigate, speed=limits.speed, verbose=verbose)
            raise

    fc.hover_in_place(navigate, speed=limits.speed, verbose=verbose)
    raise NavigationStalled(
        "цель ({:.2f}, {:.2f}) не достигнута за {} перегонов".format(
            target_x, target_y, limits.max_legs
        )
    )


def _self_test() -> int:
    markers = {}
    for row in range(7):
        for col in range(7):
            markers[row * 7 + col] = (float(col), float(row), 0.0)

    # --- Geofence ---
    fence = Geofence.from_markers(markers, margin=0.5)
    assert (fence.min_x, fence.max_x, fence.min_y, fence.max_y) == (-0.5, 6.5, -0.5, 6.5)
    assert fence.contains(0.0, 0.0, 1.5)
    assert fence.contains(6.5, 6.5)
    assert not fence.contains(6.8, 3.0)  # вылет за поле - ровно тот случай аварии
    assert not fence.contains(3.0, 3.0, 4.5)  # выше рабочей высоты
    assert not fence.contains(float("nan"), 3.0)
    assert fence.clamp(8.0, -2.0) == (6.5, -0.5)

    # --- read_localization ---
    class Telemetry:
        def __init__(self, x: float, y: float, z: float = 1.5, yaw: float = 0.0) -> None:
            self.x, self.y, self.z, self.yaw = x, y, z, yaw
            self.armed = True

    def telemetry_at(x: float, y: float, yaw: float = 0.0) -> Callable[..., Telemetry]:
        return lambda **_kwargs: Telemetry(x, y, 1.5, yaw)

    # позиция согласована с видимой меткой 24 (3, 3)
    ok_status = read_localization(
        telemetry_at(3.1, 2.9), lambda _t: (24,), markers
    )
    assert ok_status.ok and ok_status.nearest_id == 24

    # метка видна, но aruco_map утверждает, что мы на другом конце поля -
    # это и есть «уверенно врущая» локализация, которую прежний код пропускал
    lying_status = read_localization(
        telemetry_at(0.0, 0.0), lambda _t: (48,), markers
    )
    assert not lying_status.ok and "расходится с видимыми метками" in lying_status.reason

    # меток не видно вовсе
    blind_status = read_localization(telemetry_at(3.0, 3.0), lambda _t: (), markers)
    assert not blind_status.ok and "не видит" in blind_status.reason

    # NaN в телеметрии = фрейма нет
    nan_status = read_localization(
        telemetry_at(float("nan"), float("nan")), lambda _t: (24,), markers
    )
    assert not nan_status.ok and "NaN" in nan_status.reason

    # позиция не совпала с ожидаемой стартовой меткой
    wrong_start = read_localization(
        telemetry_at(3.0, 3.0), lambda _t: (24,), markers,
        expected_xy=(6.0, 6.0), expected_tolerance=1.0,
    )
    assert not wrong_start.ok and "расходится с ожидаемой" in wrong_start.reason

    # --- wait_for_localization: моргание меток допустимо, полное отсутствие - нет ---
    clock = [0.0]

    def fake_time() -> float:
        return clock[0]

    def fake_sleep(duration: float) -> None:
        clock[0] += duration

    reads = iter([(), (), (24,)])
    status = wait_for_localization(
        telemetry_at(3.0, 3.0),
        lambda _t: next(reads),
        markers,
        timeout=5.0,
        poll_interval=0.3,
        sleep_fn=fake_sleep,
        time_fn=fake_time,
    )
    assert status.ok and status.visible_ids == (24,)

    clock[0] = 0.0
    try:
        wait_for_localization(
            telemetry_at(3.0, 3.0),
            lambda _t: (),
            markers,
            timeout=1.0,
            poll_interval=0.3,
            sleep_fn=fake_sleep,
            time_fn=fake_time,
        )
    except LocalizationLost:
        pass
    else:
        raise AssertionError("Ожидался LocalizationLost")

    # --- geofence_guard ---
    clock[0] = 0.0
    guard_inside = geofence_guard(telemetry_at(3.0, 3.0), fence, time_fn=fake_time)
    guard_inside()  # внутри зоны - молча

    guard_outside = geofence_guard(telemetry_at(7.2, 3.0), fence, time_fn=fake_time)
    try:
        guard_outside()
    except GeofenceBreach:
        pass
    else:
        raise AssertionError("Ожидался GeofenceBreach за границей поля")

    # короткая потеря позиции допустима, длительная - нет
    guard_blind = geofence_guard(
        telemetry_at(float("nan"), float("nan")), fence,
        max_blind_time=1.0, time_fn=fake_time,
    )
    clock[0] = 0.0
    guard_blind()
    clock[0] = 0.5
    guard_blind()  # ещё в пределах max_blind_time
    clock[0] = 1.6
    try:
        guard_blind()
    except LocalizationLost:
        pass
    else:
        raise AssertionError("Ожидался LocalizationLost после max_blind_time")

    # --- plan_legs: прямая к цели, а не обход по решётке ---
    assert plan_legs((6.0, 6.0), (6.0, 5.4), leg_length=1.0) == [(6.0, 5.4)]
    legs = plan_legs((6.0, 6.0), (2.0, 5.0), leg_length=1.0)
    assert legs[-1] == (2.0, 5.0)
    assert len(legs) == 5  # длина 4.12м -> 5 перегонов
    for leg_x, leg_y in legs:
        # все точки лежат на прямой старт->цель (в отличие от узлов решётки)
        assert abs((leg_x - 6.0) * (5.0 - 6.0) - (leg_y - 6.0) * (2.0 - 6.0)) < 1e-9

    # --- map_vector_to_body ---
    bx, by = map_vector_to_body(1.0, 0.0, 0.0)
    assert abs(bx - 1.0) < 1e-9 and abs(by) < 1e-9
    bx, by = map_vector_to_body(1.0, 0.0, math.pi / 2)  # дрон развёрнут на +90°
    assert abs(bx) < 1e-9 and abs(by + 1.0) < 1e-9

    # --- fly_to: замкнутый контур, перегоны ведут к цели ---
    class FakeDrone:
        """Летит куда сказали, но каждый перегон промахивается на 0.2м вбок -
        замкнутый контур обязан это компенсировать, а не накопить."""

        def __init__(self, x: float, y: float) -> None:
            self.x, self.y, self.z, self.yaw = x, y, 1.5, 0.0
            self.calls: List[Dict[str, Any]] = []

        def navigate(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)
            if kwargs["frame_id"] == "aruco_map":
                self.x, self.y = kwargs["x"], kwargs["y"] + 0.2
                self.z = kwargs["z"]
            else:  # body: приращение к текущему положению
                bx, by = kwargs["x"], kwargs["y"]
                self.x += bx * math.cos(self.yaw) - by * math.sin(self.yaw)
                self.y += bx * math.sin(self.yaw) + by * math.cos(self.yaw) + 0.2
                self.z += kwargs["z"]

        def get_telemetry(self, *, frame_id: str = "aruco_map", **_kwargs: Any) -> Telemetry:
            if frame_id == "navigate_target":
                return Telemetry(0.0, 0.0, 0.0)  # прибытие мгновенное
            return Telemetry(self.x, self.y, self.z, self.yaw)

    def marker_under(drone: FakeDrone) -> Callable[[float], Tuple[int, ...]]:
        """Имитация камеры, смотрящей вниз: видна метка, над которой дрон
        сейчас находится. Заодно проверяет, что контроль согласованности
        (``read_localization``) не отвергает нормальный полёт."""

        def reader(_timeout: float) -> Tuple[int, ...]:
            col = min(6, max(0, int(round(drone.x))))
            row = min(6, max(0, int(round(drone.y))))
            return (row * 7 + col,)

        return reader

    limits = FlightLimits(
        leg_length_m=1.0,
        stabilize_hold_s=0.2,
        stabilize_timeout_s=3.0,
        stabilize_tolerance_m=0.3,
        settle_s=0.1,
    )

    for mode in ("map", "relative"):
        drone = FakeDrone(6.0, 6.0)
        clock[0] = 0.0
        final = fly_to(
            drone.navigate,
            drone.get_telemetry,
            marker_under(drone),
            markers,
            fence,
            target_x=2.0,
            target_y=5.0,
            z=1.5,
            mode=mode,
            limits=limits,
            sleep_fn=fake_sleep,
            time_fn=fake_time,
            verbose=False,
        )
        assert math.hypot(final.x - 2.0, final.y - 5.0) <= limits.arrival_tolerance_m, mode
        # ни один перегон не длиннее leg_length с запасом на промах
        assert drone.calls, mode
        assert all(call["frame_id"] in ("aruco_map", "body") for call in drone.calls), mode

    # fly_to (relative): высота выравнивается к целевой порционно - в body
    # z это приращение, и одной командой её выставить нельзя
    climbing = FakeDrone(6.0, 6.0)
    climbing.z = 2.6
    clock[0] = 0.0
    fly_to(
        climbing.navigate,
        climbing.get_telemetry,
        marker_under(climbing),
        markers,
        fence,
        target_x=2.0,
        target_y=5.0,
        z=1.5,
        mode="relative",
        limits=limits,
        sleep_fn=fake_sleep,
        time_fn=fake_time,
    )
    assert abs(climbing.z - 1.5) < 1e-6
    assert all(abs(call["z"]) <= limits.max_climb_per_leg_m + 1e-9 for call in climbing.calls)

    # fly_to: цель вне геозоны - ошибка до единой команды навигации
    drone = FakeDrone(6.0, 6.0)
    try:
        fly_to(
            drone.navigate,
            drone.get_telemetry,
            marker_under(drone),
            markers,
            fence,
            target_x=9.0,
            target_y=5.0,
            z=1.5,
            limits=limits,
            sleep_fn=fake_sleep,
            time_fn=fake_time,
        )
    except GeofenceBreach:
        pass
    else:
        raise AssertionError("Ожидался GeofenceBreach для цели вне поля")
    assert drone.calls == []

    # fly_to: метки пропали -> hover и LocalizationLost, а не полёт вслепую
    drone = FakeDrone(6.0, 6.0)
    clock[0] = 0.0
    try:
        fly_to(
            drone.navigate,
            drone.get_telemetry,
            lambda _t: (),
            markers,
            fence,
            target_x=2.0,
            target_y=5.0,
            z=1.5,
            limits=FlightLimits(localization_timeout_s=1.0),
            sleep_fn=fake_sleep,
            time_fn=fake_time,
        )
    except LocalizationLost:
        pass
    else:
        raise AssertionError("Ожидался LocalizationLost без видимых меток")
    assert drone.calls and drone.calls[-1]["frame_id"] == "body"  # погасили движение
    assert (drone.calls[-1]["x"], drone.calls[-1]["y"], drone.calls[-1]["z"]) == (0.0, 0.0, 0.0)

    # fly_to: дрон принимает команды, но с места не двигается (упёрся, ветер,
    # не тот режим PX4). В relative-режиме прибытие подтверждается фреймом
    # navigate_target, поэтому «застревание» ловится только по отсутствию
    # прогресса к цели - иначе миссия молча крутилась бы до max_legs.
    class StuckDrone(FakeDrone):
        def navigate(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)

    stuck = StuckDrone(6.0, 6.0)
    clock[0] = 0.0
    try:
        fly_to(
            stuck.navigate,
            stuck.get_telemetry,
            marker_under(stuck),
            markers,
            fence,
            target_x=2.0,
            target_y=5.0,
            z=1.5,
            mode="relative",
            limits=limits,
            sleep_fn=fake_sleep,
            time_fn=fake_time,
        )
    except NavigationStalled:
        pass
    else:
        raise AssertionError("Ожидался NavigationStalled для неподвижного дрона")
    assert stuck.calls[-1]["frame_id"] == "body" and stuck.calls[-1]["x"] == 0.0  # hover

    # тот же неподвижный дрон в режиме map упирается в стабилизацию над точкой
    # перегона - тоже аварийный останов, но по таймауту, а не по прогрессу
    stuck_map = StuckDrone(6.0, 6.0)
    clock[0] = 0.0
    try:
        fly_to(
            stuck_map.navigate,
            stuck_map.get_telemetry,
            marker_under(stuck_map),
            markers,
            fence,
            target_x=2.0,
            target_y=5.0,
            z=1.5,
            mode="map",
            limits=limits,
            sleep_fn=fake_sleep,
            time_fn=fake_time,
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("Ожидался TimeoutError стабилизации над точкой перегона")
    assert stuck_map.calls[-1]["frame_id"] == "body"  # движение погашено hover_in_place

    print("SELF-TEST: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Безопасное перемещение по полю (геозона, контроль локализации)"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="проверить логику без ROS 1/дрона"
    )
    parser.parse_args()
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())
