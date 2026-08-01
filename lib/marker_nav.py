#!/usr/bin/env python3
"""Навигация БВС-1 строго по видимым ArUco-меткам: зрение, карта, геометрия.

Подход перенесён из проверенного в полёте ``fly_head.py`` (репозиторий
Bl1tz2200/Snake) и главным отличается от остального кода этого репозитория
тем, что **телеметрия не используется вообще**: ни ``get_telemetry``, ни
``aruco_map``, ни TF. Всё, что нужно для полёта, читается с кадра камеры:

* **масштаб и высота** — по стороне метки в пикселях. Метка известного
  размера (0.33 м) даёт и «пиксель → метр», и оценку высоты: сторона
  обратно пропорциональна высоте, эталон снимается на рабочей высоте сразу
  после взлёта. Это лечит известную беду ``navigate(frame_id='body', z=0)``:
  он держит не высоту, а «сколько было в момент команды», и на разгонах
  ошибка копится только вверх;
* **курс** — по тому, как повёрнуты в кадре оси поля. Меряется по паре
  видимых меток (их разность в кадре против разности координат по карте),
  поэтому не нужно допущение о том, каким боком дрон поставлен на старте;
* **место на поле** — координаты видимой метки по карте плюс её смещение от
  центра кадра. Нужно только для проверок (геозона, деревья) и лога: в
  контуре управления глобальной позиции нет, дрон наводится на то, что
  видно в кадре прямо сейчас.

Модуль не импортирует ROS и не шлёт команд — он только считает. ``cv2``
импортируется внутри ``MarkerDetector`` (см. CLAUDE.md, раздел
«Архитектура»), поэтому чистая математика проверяется на десктопе::

    python3 lib/marker_nav.py --self-test
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

# Деревья высотой ~1 м стоят на стыках четырёх меток; здесь перечислены сами
# метки, а координата ствола считается из карты как их центр — раскладку поля
# объявляют перед попыткой, и пересчитывать координаты руками не придётся.
DEFAULT_TREE_GROUPS: Tuple[Tuple[int, ...], ...] = (
    (12, 13, 19, 20),
    (38, 39, 45, 46),
    (9, 10, 16, 17),
    (28, 29, 35, 36),
)


class NavigationError(RuntimeError):
    """Лететь дальше нельзя: обстановка не соответствует ожидаемой."""


class Marker(NamedTuple):
    """Метка на кадре: место и размер в пикселях, поворот в радианах."""

    mid: int
    x: float
    y: float
    side: float
    angle: float


# ═══════════════════════════════════════════════════════════════════════
#  КАРТА ПОЛЯ
# ═══════════════════════════════════════════════════════════════════════


def read_field_map(path: str) -> Dict[int, Tuple[float, float]]:
    """Карта поля в формате aruco_pose: ``id size x y z rot_z rot_y rot_x``.

    Возвращает ``{id: (x, y)}``. Высота меток не нужна: они все лежат на полу.
    """
    field: Dict[int, Tuple[float, float]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = line.split("#", 1)[0].split()
            if not row:
                continue
            if len(row) < 4:
                raise ValueError(
                    "{}:{}: ожидалось 'id size x y ...', получено {!r}".format(
                        path, line_number, line.strip()
                    )
                )
            field[int(row[0])] = (float(row[2]), float(row[3]))
    if not field:
        raise ValueError("Карта поля пуста: " + path)
    return field


def marker_xy(field: Dict[int, Tuple[float, float]], mid: int) -> Tuple[float, float]:
    """Координаты метки по карте. Нет такой метки — понятная ошибка, не KeyError."""
    if mid not in field:
        raise NavigationError("Метки {} нет в карте поля".format(mid))
    return field[mid]


def nearest_marker_id(field: Dict[int, Tuple[float, float]], x: float, y: float) -> int:
    """Ближайшая к точке метка — для печати «у метки N» и старта маршрута."""
    return min(field, key=lambda mid: math.hypot(field[mid][0] - x, field[mid][1] - y))


def field_step(field: Dict[int, Tuple[float, float]]) -> float:
    """Шаг решётки по карте: минимальное ненулевое расстояние между метками."""
    points = list(field.values())
    best = float("inf")
    for index, (ax, ay) in enumerate(points):
        for bx, by in points[index + 1 :]:
            distance = math.hypot(ax - bx, ay - by)
            if 1e-6 < distance < best:
                best = distance
    if not math.isfinite(best):
        raise NavigationError("В карте поля меньше двух различимых меток")
    return best


# ═══════════════════════════════════════════════════════════════════════
#  ГЕОЗОНА И ДЕРЕВЬЯ
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Geofence:
    """Разрешённая зона: габариты поля по крайним меткам плюс запас."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @classmethod
    def from_field(cls, field: Dict[int, Tuple[float, float]], margin: float = 0.5) -> "Geofence":
        xs = [x for x, _y in field.values()]
        ys = [y for _x, y in field.values()]
        return cls(min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)

    def contains(self, x: float, y: float) -> bool:
        if math.isnan(x) or math.isnan(y):
            return False
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def describe(self) -> str:
        return "x {:.1f}..{:.1f}, y {:.1f}..{:.1f}".format(
            self.min_x, self.max_x, self.min_y, self.max_y
        )


def tree_positions(
    field: Dict[int, Tuple[float, float]],
    groups: Iterable[Sequence[int]] = DEFAULT_TREE_GROUPS,
) -> Tuple[Tuple[float, float], ...]:
    """Стволы деревьев: центр каждой группы меток, в углу которых стоит дерево."""
    trees: List[Tuple[float, float]] = []
    for group in groups:
        ids = list(group)
        if not ids:
            continue
        points = [marker_xy(field, mid) for mid in ids]
        trees.append(
            (
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            )
        )
    return tuple(trees)


def point_to_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Расстояние от точки до отрезка — им и меряется зазор до ствола."""
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def segment_is_clear(
    start: Tuple[float, float],
    end: Tuple[float, float],
    trees: Sequence[Tuple[float, float]],
    clearance: float,
) -> bool:
    """Отрезок проходит не ближе ``clearance`` к каждому стволу.

    Сравнение с допуском в микрон: типовые зазоры на решётке — ровно половина
    шага (0.5 м), и зазор по умолчанию задан таким же. Строгое сравнение
    сделало бы допустимость такого прохода зависящей от последнего бита
    арифметики с плавающей точкой, то есть от координат в карте поля.
    """
    return all(
        point_to_segment(tx, ty, start[0], start[1], end[0], end[1]) >= clearance - 1e-6
        for tx, ty in trees
    )


def plan_route(
    field: Dict[int, Tuple[float, float]],
    start_id: int,
    goal_id: int,
    *,
    trees: Sequence[Tuple[float, float]] = (),
    clearance: float = 0.5,
    step: Optional[float] = None,
) -> List[int]:
    """Кратчайший путь по решётке меток в обход деревьев: список ID без стартового.

    Рёбра — только соседние метки по осям (никаких диагоналей: команда полёта
    и так отдаётся по одной оси). Ребро отбрасывается, если проходит ближе
    ``clearance`` к любому стволу.

    Деревья стоят в углах ячеек, поэтому зазор до ствола у рёбер решётки
    принимает всего несколько значений: 0.5 шага у четырёх рёбер, идущих
    вплотную вдоль дерева, ~0.71 у соседних с ними и дальше. Отсюда и смысл
    ``clearance``: 0.5 разрешает проход вплотную (маршрут короче), всё, что
    больше, отбрасывает эти четыре ребра и уводит маршрут на ячейку в сторону.
    """
    if start_id == goal_id:
        return []
    marker_xy(field, start_id)
    marker_xy(field, goal_id)
    grid = field_step(field) if step is None else step

    def neighbours(mid: int) -> List[int]:
        x, y = field[mid]
        found = []
        for other, (ox, oy) in field.items():
            if other == mid:
                continue
            dx, dy = abs(ox - x), abs(oy - y)
            axis_aligned = (dx < 1e-6 and abs(dy - grid) < 1e-6) or (
                dy < 1e-6 and abs(dx - grid) < 1e-6
            )
            if axis_aligned and segment_is_clear((x, y), (ox, oy), trees, clearance):
                found.append(other)
        return sorted(found)

    previous: Dict[int, Optional[int]] = {start_id: None}
    queue = deque([start_id])
    while queue:
        current = queue.popleft()
        if current == goal_id:
            break
        for neighbour in neighbours(current):
            if neighbour not in previous:
                previous[neighbour] = current
                queue.append(neighbour)

    if goal_id not in previous:
        raise NavigationError(
            "Маршрут {} -> {} не строится: все проходы перекрыты деревьями "
            "(зазор {:.2f} м — попробуйте меньше)".format(start_id, goal_id, clearance)
        )

    route: List[int] = []
    node: Optional[int] = goal_id
    while node is not None and node != start_id:
        route.append(node)
        node = previous[node]
    route.reverse()
    return route


def tree_hit(
    x: float, y: float, trees: Sequence[Tuple[float, float]], clearance: float
) -> Optional[Tuple[float, float]]:
    """Ствол, в цилиндр которого попала точка, или None. Сторож на каждый шаг."""
    for tree in trees:
        if math.hypot(x - tree[0], y - tree[1]) < clearance:
            return tree
    return None


# ═══════════════════════════════════════════════════════════════════════
#  ЗРЕНИЕ
# ═══════════════════════════════════════════════════════════════════════


class MarkerDetector:
    """Метки на кадре через ``cv2.aruco``. Единственное место с OpenCV.

    Импорт спрятан в конструктор: остальной модуль считается без OpenCV и
    проверяется самотестом на любой машине.
    """

    def __init__(
        self,
        dictionary_name: str = "DICT_4X4_50",
        *,
        max_marker_id: int = 48,
        min_side_px: float = 8.0,
    ) -> None:
        try:
            import cv2  # noqa: F401  (нужен только внутри детектора)
        except ImportError as exc:  # pragma: no cover - зависит от машины
            raise RuntimeError(
                "Нужен opencv-contrib-python (модуль cv2.aruco)"
            ) from exc
        if not hasattr(cv2, "aruco"):  # pragma: no cover
            raise RuntimeError("В установленном OpenCV нет cv2.aruco (нужен contrib-пакет)")
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError("Неизвестный словарь ArUco: " + dictionary_name)

        factory = getattr(
            cv2.aruco, "getPredefinedDictionary", getattr(cv2.aruco, "Dictionary_get", None)
        )
        if factory is None:  # pragma: no cover
            raise RuntimeError("OpenCV не умеет создавать словарь ArUco")
        self._cv2 = cv2
        self._dictionary = factory(getattr(cv2.aruco, dictionary_name))
        params_factory = getattr(
            cv2.aruco, "DetectorParameters", getattr(cv2.aruco, "DetectorParameters_create", None)
        )
        self._params = params_factory() if params_factory is not None else None
        self._detector = (
            cv2.aruco.ArucoDetector(self._dictionary, self._params)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self.max_marker_id = max_marker_id
        self.min_side_px = min_side_px

    def detect(self, image: Any) -> Dict[int, Marker]:
        """``{ID: Marker}`` по кадру BGR."""
        cv2 = self._cv2
        import numpy as np

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._detector is not None:
            corners, ids, _rejected = self._detector.detectMarkers(gray)
        else:  # pragma: no cover - старое API OpenCV
            corners, ids, _rejected = cv2.aruco.detectMarkers(
                gray, self._dictionary, parameters=self._params
            )
        if ids is None or len(ids) == 0:
            return {}

        seen: Dict[int, Marker] = {}
        for quad, mid in zip(corners, ids.flatten()):
            points = np.asarray(quad, np.float32).reshape(-1, 2)
            centre = points.mean(axis=0)
            # Сторона метки — САМОЕ ДЛИННОЕ ребро, а не среднее: наклонённый дрон
            # видит квадрат прямоугольником, поперечные рёбра сжимаются в косинус
            # наклона. Среднее занижало сторону, а по ней считается высота — дрон
            # решал, что он выше, чем есть, и получал ложную команду вниз.
            side = float(
                max(
                    float(np.linalg.norm(points[i] - points[(i + 1) % 4]))
                    for i in range(4)
                )
            )
            edge = points[1] - points[0]
            marker_id = int(mid)
            if side >= self.min_side_px and 0 <= marker_id <= self.max_marker_id:
                seen[marker_id] = Marker(
                    marker_id,
                    float(centre[0]),
                    float(centre[1]),
                    side,
                    math.atan2(float(edge[1]), float(edge[0])),
                )
        return seen


def nearest_to_centre(
    seen: Dict[int, Marker], centre: Tuple[float, float]
) -> Optional[Marker]:
    """Метка, над которой висим: ближайшая к центру кадра."""
    if not seen:
        return None
    return min(
        seen.values(), key=lambda m: math.hypot(m.x - centre[0], m.y - centre[1])
    )


# ═══════════════════════════════════════════════════════════════════════
#  ВЫСОТА ПО МЕТКЕ
# ═══════════════════════════════════════════════════════════════════════


def alt_by_side(side: float, side_ref: float, alt_ref: float) -> float:
    """Высота по стороне метки в кадре, м. 0.0 — эталон ещё не снят."""
    if side_ref <= 1.0 or side <= 1.0:
        return 0.0
    return alt_ref * side_ref / side


def hold_alt(
    side: float,
    side_ref: float,
    alt_ref: float,
    *,
    limit: float = 0.3,
    dead_zone: float = 0.07,
) -> float:
    """На сколько подняться (+) или опуститься (−) до рабочей высоты."""
    now = alt_by_side(side, side_ref, alt_ref)
    if not now:
        return 0.0
    correction = alt_ref - now
    if abs(correction) < dead_zone:  # не дёргаем дрон по мелочи
        return 0.0
    return max(-limit, min(limit, correction))


def metres_per_pixel(side: float, marker_size: float) -> float:
    """Масштаб кадра по стороне метки известного размера."""
    if side <= 1.0:
        raise NavigationError("Метка слишком мелкая в кадре, масштаб не посчитать")
    return marker_size / side


# ═══════════════════════════════════════════════════════════════════════
#  КУРС: КАК ПОВЁРНУТЫ ОСИ ПОЛЯ В КАДРЕ
# ═══════════════════════════════════════════════════════════════════════
#
# Камера смотрит вниз, поэтому верх кадра — «вперёд» по корпусу, левый край —
# «влево»: смещение метки от центра кадра сразу даёт смещение дрона в осях
# корпуса. А вот вектор, заданный ПО КАРТЕ («цель на два узла левее»), задан в
# осях поля, и его надо развернуть — на угол, показывающий, как дрон стоит
# относительно поля. Этот угол и есть frame_yaw: 0 означает «нос дрона смотрит
# вдоль оси +Y карты» (при таком положении ось +Y поля идёт вперёд, ось +X —
# вправо, то есть влево = −X).


def frame_vector(marker: Marker, centre: Tuple[float, float], scale: float) -> Tuple[float, float]:
    """Куда лететь по корпусу, чтобы встать над меткой: (вперёд, влево), м."""
    return -(marker.y - centre[1]) * scale, -(marker.x - centre[0]) * scale


def field_to_body(dx: float, dy: float, frame_yaw: float) -> Tuple[float, float]:
    """Вектор из осей поля в оси корпуса при известном ``frame_yaw``."""
    forward, left = dy, -dx  # положение «нос вдоль +Y карты»
    cos_a, sin_a = math.cos(frame_yaw), math.sin(frame_yaw)
    return forward * cos_a - left * sin_a, forward * sin_a + left * cos_a


def body_to_field(forward: float, left: float, frame_yaw: float) -> Tuple[float, float]:
    """Обратное к ``field_to_body``: вектор корпуса в оси поля."""
    cos_a, sin_a = math.cos(-frame_yaw), math.sin(-frame_yaw)
    base_forward = forward * cos_a - left * sin_a
    base_left = forward * sin_a + left * cos_a
    return -base_left, base_forward


def frame_yaw_from_pairs(
    seen: Dict[int, Marker], field: Dict[int, Tuple[float, float]], marker_size: float
) -> Optional[float]:
    """Поворот осей поля в кадре по всем парам видимых меток, радианы.

    Самый надёжный способ: разность двух меток в кадре против их разности по
    карте. Не требует ни эталона на взлёте, ни допущения о том, каким боком
    дрон поставлен на старте. Углы усредняются векторно (иначе разрыв ±180°
    дал бы бессмысленное среднее).
    """
    known = [m for m in seen.values() if m.mid in field]
    if len(known) < 2:
        return None

    sin_sum = cos_sum = 0.0
    used = 0
    for index, first in enumerate(known):
        for second in known[index + 1 :]:
            fx, fy = field[first.mid]
            sx, sy = field[second.mid]
            dx, dy = sx - fx, sy - fy
            if math.hypot(dx, dy) < 1e-6:
                continue
            scale = metres_per_pixel(min(first.side, second.side), marker_size)
            measured = (
                -(second.y - first.y) * scale,
                -(second.x - first.x) * scale,
            )
            if math.hypot(*measured) < 1e-6:
                continue
            base = field_to_body(dx, dy, 0.0)
            angle = math.atan2(measured[1], measured[0]) - math.atan2(base[1], base[0])
            sin_sum += math.sin(angle)
            cos_sum += math.cos(angle)
            used += 1
    if not used or math.hypot(sin_sum, cos_sum) < 1e-9:
        return None
    return math.atan2(sin_sum, cos_sum)


def wrap_angle(angle: float) -> float:
    """Угол в диапазон −π…π."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def estimate_position(
    base: Marker,
    field: Dict[int, Tuple[float, float]],
    frame_yaw: float,
    centre: Tuple[float, float],
    marker_size: float,
) -> Tuple[float, float]:
    """Где дрон на поле: координаты видимой метки минус её смещение в кадре."""
    scale = metres_per_pixel(base.side, marker_size)
    forward, left = frame_vector(base, centre, scale)
    dx, dy = body_to_field(forward, left, frame_yaw)
    mx, my = marker_xy(field, base.mid)
    return mx - dx, my - dy


def measured_grid_step(
    seen: Dict[int, Marker], field: Dict[int, Tuple[float, float]], marker_size: float
) -> float:
    """Шаг решётки, измеренный по двум соседним меткам в кадре, м. 0.0 — нет пары.

    Самопроверка ``marker_size``: если замер сильно расходится с картой, значит
    сторона метки в настройках не та, и по ней неверно считается вся геометрия.
    """
    known = [m for m in seen.values() if m.mid in field]
    grid = field_step(field) if len(field) > 1 else 0.0
    for index, first in enumerate(known):
        for second in known[index + 1 :]:
            fx, fy = field[first.mid]
            sx, sy = field[second.mid]
            if abs(math.hypot(sx - fx, sy - fy) - grid) > 1e-6:
                continue  # не соседи по решётке
            gap = math.hypot(second.x - first.x, second.y - first.y)
            return gap * marker_size / min(first.side, second.side)
    return 0.0


# ═══════════════════════════════════════════════════════════════════════
#  ОДНА ОСЬ ЗА КОМАНДУ
# ═══════════════════════════════════════════════════════════════════════


class Hop(NamedTuple):
    """Одна команда полёта: ось и величина. Больше в команде ничего нет."""

    axis: str
    value: float


def hop_candidates(
    forward: float,
    left: float,
    *,
    limit: float = 1.5,
    dead_zone: float = 0.05,
) -> List[Hop]:
    """Варианты ОДНОЙ команды в порядке предпочтения: ось с большей ошибкой первой.

    Вторая ось не теряется — она гасится следующим кадром: контур замкнут по
    меткам, позиция перечитывается перед каждой командой. ``limit`` режет
    слишком длинный перелёт: он почти всегда означает, что опорная метка
    опознана неверно, и лететь по нему целиком опасно.

    Вариантов несколько, потому что команда идёт по осям КОРПУСА, а маршрут
    проложен по осям ПОЛЯ: у развёрнутого дрона перелёт получается косым и
    может пройти рядом с деревом. Тогда вызывающий берёт следующий вариант —
    другую ось или укороченный шаг, — а не летит напролом.
    """
    order = (
        (("x", forward), ("y", left))
        if abs(forward) >= abs(left)
        else (("y", left), ("x", forward))
    )
    variants: List[Hop] = []
    for factor in (1.0, 0.5, 0.25):
        for axis, value in order:
            clipped = max(-limit, min(limit, value * factor))
            if abs(clipped) >= dead_zone:
                variants.append(Hop(axis, clipped))
    return variants


# ═══════════════════════════════════════════════════════════════════════
#  САМОТЕСТ: только чистая математика, без ROS, OpenCV и имитации полёта
# ═══════════════════════════════════════════════════════════════════════


def _fake_marker(
    mid: int,
    field: Dict[int, Tuple[float, float]],
    drone: Tuple[float, float],
    frame_yaw: float,
    side: float,
    centre: Tuple[float, float],
    marker_size: float,
) -> Marker:
    """Как метка легла бы в кадр при таком положении дрона — для проверки формул."""
    mx, my = field[mid]
    forward, left = field_to_body(mx - drone[0], my - drone[1], frame_yaw)
    scale = marker_size / side
    return Marker(mid, centre[0] - left / scale, centre[1] - forward / scale, side, 0.0)


def _self_test() -> None:
    from pathlib import Path

    map_path = Path(__file__).resolve().parent.parent / "config" / "field_map.txt"
    field = read_field_map(str(map_path))
    assert len(field) == 49, len(field)
    assert marker_xy(field, 5) == (5.0, 0.0)
    assert marker_xy(field, 48) == (6.0, 6.0)
    assert abs(field_step(field) - 1.0) < 1e-9
    assert nearest_marker_id(field, 5.4, 0.1) == 5

    # Деревья: ствол — центр четырёх меток, вокруг которых он стоит.
    trees = tree_positions(field)
    assert trees[0] == (5.5, 1.5), trees[0]
    assert set(trees) == {(5.5, 1.5), (3.5, 5.5), (2.5, 1.5), (0.5, 4.5)}
    assert tree_hit(5.6, 1.6, trees, 0.5) == (5.5, 1.5)
    assert tree_hit(3.0, 3.0, trees, 0.5) is None

    # Прямая «старт -> станция» проходит в 0.25 м от дерева (5.5, 1.5) — ради
    # этого и нужен маршрут по решётке, где зазор до ствола не меньше половины
    # ячейки.
    assert not segment_is_clear((6.0, 6.0), (5.0, 0.0), trees, 0.5)

    for clearance, expected in ((0.5, 7), (0.8, 9)):
        route = plan_route(field, 48, 5, trees=trees, clearance=clearance)
        assert route[-1] == 5, route
        assert len(route) == expected, (clearance, route)
        previous = marker_xy(field, 48)
        for mid in route:
            current = marker_xy(field, mid)
            moved = (abs(current[0] - previous[0]), abs(current[1] - previous[1]))
            assert abs(moved[0] + moved[1] - 1.0) < 1e-9 and min(moved) < 1e-9, (mid, moved)
            assert segment_is_clear(previous, current, trees, clearance), (clearance, mid)
            previous = current
    # Зазор по умолчанию (0.5) пускает дрон вплотную вдоль ствола — маршрут
    # получается кратчайшим манхэттенским, как если бы деревьев не было.
    assert len(plan_route(field, 48, 5, trees=(), clearance=0.5)) == 7
    try:
        plan_route(field, 48, 5, trees=trees, clearance=5.0)
    except NavigationError:
        pass
    else:  # слишком большой зазор перекрывает поле — об этом надо узнать до взлёта
        raise AssertionError("ожидалась ошибка: маршрут не строится")

    fence = Geofence.from_field(field, margin=0.5)
    assert fence.contains(0.0, 0.0) and fence.contains(6.5, 6.5)
    assert not fence.contains(-0.6, 3.0) and not fence.contains(float("nan"), 3.0)

    # Высота по стороне метки: вдвое мельче метка — вдвое выше дрон.
    assert abs(alt_by_side(100.0, 100.0, 1.5) - 1.5) < 1e-9
    assert abs(alt_by_side(50.0, 100.0, 1.5) - 3.0) < 1e-9
    assert alt_by_side(0.0, 100.0, 1.5) == 0.0
    assert hold_alt(100.0, 100.0, 1.5) == 0.0  # мёртвая зона
    assert abs(hold_alt(50.0, 100.0, 1.5) + 0.3) < 1e-9  # вниз, урезано до limit
    assert abs(hold_alt(200.0, 100.0, 1.5) - 0.3) < 1e-9  # вверх, урезано до limit

    # Оси: при frame_yaw=0 нос смотрит вдоль +Y карты, значит +X карты — вправо.
    forward, left = field_to_body(1.0, 0.0, 0.0)
    assert abs(forward) < 1e-9 and abs(left + 1.0) < 1e-9, (forward, left)
    forward, left = field_to_body(1.0, 0.0, math.pi / 2)
    assert abs(forward - 1.0) < 1e-9 and abs(left) < 1e-9, (forward, left)
    for yaw in (0.0, 0.7, -2.5):
        back = body_to_field(*field_to_body(1.3, -0.4, yaw), frame_yaw=yaw)
        assert abs(back[0] - 1.3) < 1e-9 and abs(back[1] + 0.4) < 1e-9, back

    # Курс и место по синтетическому кадру: дрон между метками, кадр повёрнут.
    centre, marker_size, side = (640.0, 360.0), 0.33, 120.0
    drone, yaw = (4.3, 2.1), 0.6
    seen = {
        mid: _fake_marker(mid, field, drone, yaw, side, centre, marker_size)
        for mid in (18, 19, 25)
    }
    measured_yaw = frame_yaw_from_pairs(seen, field, marker_size)
    assert measured_yaw is not None and abs(wrap_angle(measured_yaw - yaw)) < 1e-6
    assert frame_yaw_from_pairs({18: seen[18]}, field, marker_size) is None
    base = nearest_to_centre(seen, centre)
    assert base is not None and base.mid == 18, base  # метка (4, 2) ближе всех к дрону
    x, y = estimate_position(base, field, measured_yaw, centre, marker_size)
    assert abs(x - drone[0]) < 1e-6 and abs(y - drone[1]) < 1e-6, (x, y)
    assert abs(measured_grid_step(seen, field, marker_size) - 1.0) < 1e-6

    # Команда — ровно одна ось, длиннее предела не бывает, первый вариант —
    # ось с большей ошибкой, запасные — другая ось и укороченный шаг.
    variants = hop_candidates(1.0, 0.2)
    assert variants[0] == Hop("x", 1.0), variants[0]
    assert variants[1] == Hop("y", 0.2), variants[1]
    assert all(abs(hop.value) <= 1.5 + 1e-9 for hop in variants)
    assert hop_candidates(0.2, -2.0, limit=1.5)[0] == Hop("y", -1.5)
    assert hop_candidates(0.01, -0.02) == []

    print("marker_nav: самотест пройден")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Навигация по видимым ArUco-меткам: расчётная часть без ROS"
    )
    parser.add_argument("--self-test", action="store_true", help="прогнать самотест")
    parser.add_argument("--map", default="config/field_map.txt", help="карта поля")
    parser.add_argument("--start-marker", type=int, default=48)
    parser.add_argument("--station-marker", type=int, default=5)
    parser.add_argument("--tree-clearance", type=float, default=0.5)
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0

    # Без флага — показать маршрут для заданной раскладки: этим удобно
    # проверить обход деревьев, не запуская ничего на дроне.
    field = read_field_map(args.map)
    trees = tree_positions(field)
    route = plan_route(
        field,
        args.start_marker,
        args.station_marker,
        trees=trees,
        clearance=args.tree_clearance,
    )
    print("деревья: " + ", ".join("({:.1f}, {:.1f})".format(x, y) for x, y in trees))
    print(
        "маршрут {} -> {}: {}".format(
            args.start_marker, args.station_marker, " -> ".join(str(mid) for mid in route)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
