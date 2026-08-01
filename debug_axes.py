#!/usr/bin/env python3
"""Проверка осей карты: совпадают ли X и Y у aruco_map с физическим полем.

Моторы не включаются. Дрон держат руками **над** тремя метками по очереди:
начало координат, «метка по оси X» и «метка по оси Y». Скрипт снимает позицию
с ``get_telemetry(frame_id='aruco_map')`` в каждой точке и сам ставит диагноз:
оси совпадают, перепутаны местами, направлены в обратную сторону, карта
зеркальная, повёрнутая, чужая или врёт масштаб.

Метки берутся **незакрытые**: по умолчанию 41 (начало), 34 (ось X) и 40 (ось
Y) — они рядом со стартовой площадкой и ни подо что не попадают. Метку 48 для
этого брать нельзя: её закрывает площадка «Н», а соседние метки с малой высоты
в кадр не попадают (на этом поле две метки сразу видны только метров с двух),
и над ней платформа просто не даст позиции.

Держать дрон нужно примерно в полуметре над меткой, камерой ровно над её
центром. Одной метки в кадре платформе достаточно: ``aruco_map`` считает
позицию по любым видимым меткам поля, поэтому вопрос к ней стоит не «видишь ли
ты метку N», а «какие у тебя сейчас координаты».

Зачем это нужно. Кадр карты в ROS — ENU, правая тройка (docs Clover,
«Map-based navigation with ArUco markers»). Если карта на платформе задана
зеркально или повёрнуто, дрон получает координаты, которые не соответствуют
полю: команда «вперёд по X» уводит вбок, и никакие настройки скрипта этого
не лечат. Проверка занимает две минуты и делается до первого взлёта.

Запуск на дроне::

    python3 debug_axes.py --map config/field_map.txt

Проверка самой логики диагноза, без ROS и дрона::

    python3 debug_axes.py --self-test
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import marker_nav as nav  # noqa: E402  (после sys.path выше)

Pose = Tuple[float, float, float, float]  # x, y, z, yaw


# ═══════════════════════════════════════════════════════════════════════
#  ROS — единственное место с rospy
# ═══════════════════════════════════════════════════════════════════════


def make_pose_reader(node_name: str = "debug_axes") -> Callable[[], Optional[Pose]]:
    """Читалка позиции из кадра карты. Импорт ROS спрятан здесь намеренно."""
    try:
        import rospy
        from technic import srv
    except ImportError as exc:
        raise RuntimeError(
            "Нужны rospy и technic. Выполните source /opt/ros/noetic/setup.bash "
            "и source своего catkin_ws"
        ) from exc

    rospy.init_node(node_name)
    get_telemetry = rospy.ServiceProxy("get_telemetry", srv.GetTelemetry)

    def read() -> Optional[Pose]:
        try:
            telemetry = get_telemetry(frame_id="aruco_map")
        except Exception as exc:  # noqa: BLE001 - следующая попытка может пройти
            print("    get_telemetry не ответил: {}".format(exc))
            return None
        pose = (float(telemetry.x), float(telemetry.y),
                float(getattr(telemetry, "z", float("nan"))),
                float(getattr(telemetry, "yaw", float("nan"))))
        # По документации Clover телеметрия в кадре карты отдаёт NaN, если
        # карта не распознавалась 0.5 с. Это «не вижу», а не «ошибка».
        if math.isnan(pose[0]) or math.isnan(pose[1]):
            return None
        return pose

    return read


# ═══════════════════════════════════════════════════════════════════════
#  ЗАМЕР
# ═══════════════════════════════════════════════════════════════════════


def average_pose(
    read: Callable[[], Optional[Pose]],
    samples: int,
    pause: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Optional[Tuple[Pose, float]]:
    """Среднее из ``samples`` замеров и разброс по горизонтали.

    Разброс важнее самого среднего: если он большой, позиция скачет, и
    сравнивать направления осей ещё рано — сначала разбираться с картой,
    освещением и калибровкой.
    """
    poses: List[Pose] = []
    for _ in range(samples):
        pose = read()
        if pose is not None:
            poses.append(pose)
        sleep_fn(pause)
    if not poses:
        return None
    mean = tuple(sum(p[i] for p in poses) / len(poses) for i in range(4))
    spread = max(math.hypot(p[0] - mean[0], p[1] - mean[1]) for p in poses)
    return (mean[0], mean[1], mean[2], mean[3]), spread  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════
#  ДИАГНОЗ
# ═══════════════════════════════════════════════════════════════════════


def _angle_between(vector: Tuple[float, float], reference: Tuple[float, float]) -> float:
    """Угол от ``reference`` к ``vector`` в градусах, −180…180."""
    angle = math.atan2(vector[1], vector[0]) - math.atan2(reference[1], reference[0])
    return math.degrees(nav.wrap_angle(angle))


def diagnose(
    measured_x: Tuple[float, float],
    measured_y: Tuple[float, float],
    expected_x: Tuple[float, float],
    expected_y: Tuple[float, float],
    *,
    origin_error: Optional[float] = None,
    angle_tolerance: float = 20.0,
    scale_tolerance: float = 0.2,
    origin_tolerance: float = 0.5,
) -> Tuple[bool, List[str]]:
    """Сравнить измеренные направления осей с ожидаемыми по карте.

    ``measured_*`` — векторы «из начала координат к контрольной метке», как их
    показала телеметрия; ``expected_*`` — те же векторы по нашей карте.
    Возвращает (всё ли в порядке, список строк отчёта).
    """
    report: List[str] = []
    ok = True

    # Смещение начала координат — самый прямой признак «на платформе другая
    # карта». Направления осей могут совпасть случайно (обе карты правые), а
    # вот начало отсчёта у чужой карты почти наверняка в другом месте.
    wrong_origin = origin_error is not None and origin_error > origin_tolerance
    if wrong_origin:
        ok = False
        report.append(
            "НАЧАЛО КООРДИНАТ не совпало: над стартовой меткой платформа "
            "показывает точку в {:.2f} м от нуля карты. Это значит, что нода "
            "aruco_map читает ДРУГОЙ файл карты, а не наш".format(origin_error))

    len_measured_x = math.hypot(*measured_x)
    len_measured_y = math.hypot(*measured_y)
    len_expected_x = math.hypot(*expected_x)
    len_expected_y = math.hypot(*expected_y)
    if min(len_measured_x, len_measured_y) < 1e-6:
        return False, ["дрон не сдвинулся между замерами — точки совпали"]

    # 1. Масштаб: во сколько раз измеренное расстояние отличается от карты.
    for name, measured, expected in (("X", len_measured_x, len_expected_x),
                                     ("Y", len_measured_y, len_expected_y)):
        ratio = measured / expected
        if abs(ratio - 1.0) > scale_tolerance:
            ok = False
            report.append(
                "МАСШТАБ по оси {}: замер {:.2f} м вместо {:.2f} м (в {:.2f} раза). "
                "Обычно это некалиброванная камера или неверный размер метки".format(
                    name, measured, expected, ratio))

    # 2. Зеркальность: знак векторного произведения должен совпадать.
    measured_cross = measured_x[0] * measured_y[1] - measured_x[1] * measured_y[0]
    expected_cross = expected_x[0] * expected_y[1] - expected_x[1] * expected_y[0]
    if measured_cross * expected_cross < 0:
        ok = False
        report.append(
            "ЗЕРКАЛЬНАЯ карта: оси X и Y образуют тройку противоположного знака. "
            "Никакой поворот это не исправит — карту на платформе надо заменить")

    # 3. Направления осей.
    angle_x = _angle_between(measured_x, expected_x)
    angle_y = _angle_between(measured_y, expected_y)
    report.append("ось X: замер повёрнут на {:+.0f}° от ожидаемого".format(angle_x))
    report.append("ось Y: замер повёрнут на {:+.0f}° от ожидаемого".format(angle_y))

    if abs(angle_x) <= angle_tolerance and abs(angle_y) <= angle_tolerance:
        if ok:
            report.append("ОСИ СОВПАДАЮТ: карта на платформе соответствует полю")
        elif wrong_origin:
            report.append("направления осей верные — расходится только начало отсчёта")
        return ok, report

    ok = False
    if measured_cross * expected_cross < 0 and abs(abs(angle_x) - 90.0) <= angle_tolerance:
        report.append("X и Y ПЕРЕПУТАНЫ МЕСТАМИ (это и есть зеркальность через обмен осей)")
    elif abs(nav.wrap_angle(math.radians(angle_x - angle_y))) <= math.radians(angle_tolerance):
        turn = (angle_x + angle_y) / 2
        if wrong_origin:
            report.append(
                "карта ПОВЁРНУТА на {:+.0f}° целиком и начало отсчёта другое — "
                "это чужая карта (наша дала бы 0° и совпавший ноль)".format(turn))
        else:
            report.append(
                "карта ПОВЁРНУТА на {:+.0f}° целиком (обе оси уехали одинаково), "
                "а начало отсчёта совпало — поле физически развёрнуто "
                "относительно нумерации".format(turn))
    if abs(abs(angle_x) - 180.0) <= angle_tolerance:
        report.append("ось X направлена В ОБРАТНУЮ сторону")
    if abs(abs(angle_y) - 180.0) <= angle_tolerance:
        report.append("ось Y направлена В ОБРАТНУЮ сторону")
    return ok, report


# ═══════════════════════════════════════════════════════════════════════
#  СЦЕНАРИЙ
# ═══════════════════════════════════════════════════════════════════════


def run(
    read: Callable[[], Optional[Pose]],
    field: Dict[int, Tuple[float, float]],
    markers: Sequence[int],
    *,
    samples: int,
    pause: float,
    wait_fn: Callable[[str], Any] = input,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    """Три замера и диагноз. ``markers`` = (начало, по оси X, по оси Y)."""
    origin_id, axis_x_id, axis_y_id = markers
    labels = ("НАЧАЛО КООРДИНАТ", "ОСЬ X", "ОСЬ Y")
    poses: List[Pose] = []

    print("Проверка осей карты. Моторы не включаются, дрон носим руками.")
    print("Держите дрон примерно в 0.5 м НАД меткой, камерой над её центром.")
    print("Платформе хватает одной метки в кадре — это не полёт по паре меток.\n")
    for label, mid in zip(labels, markers):
        expected = nav.marker_xy(field, mid)
        wait_fn("  {}: держите дрон над меткой {} (ожидаем {:.2f}, {:.2f}) "
                "и нажмите Enter...".format(label, mid, expected[0], expected[1]))
        measured = average_pose(read, samples, pause, sleep_fn=sleep_fn)
        if measured is None:
            print("    карта не видна — позиции нет. Проверьте ноду aruco_map, "
                  "map1.txt и освещение")
            return False
        pose, spread = measured
        print("    замер: ({:+.2f}, {:+.2f}, {:+.2f}), курс {:+.0f}°, разброс {:.2f} м"
              .format(pose[0], pose[1], pose[2], math.degrees(pose[3]), spread))
        if not math.isnan(pose[2]) and pose[2] < -0.05:
            print("    !!! z отрицательный ({:+.2f} м): дрон физически ВЫШЕ плоскости "
                  "меток, а "
                  "карта считает его ниже. Карта на платформе задана для потолка "
                  "или это вообще не наша карта".format(pose[2]))
        if spread > 0.15:
            print("    !!! позиция скачет на {:.2f} м — сравнивать оси рано, "
                  "сначала разберитесь с этим".format(spread))
        poses.append(pose)

    origin, along_x, along_y = poses
    measured_x = (along_x[0] - origin[0], along_x[1] - origin[1])
    measured_y = (along_y[0] - origin[0], along_y[1] - origin[1])
    map_origin = nav.marker_xy(field, origin_id)
    map_x = nav.marker_xy(field, axis_x_id)
    map_y = nav.marker_xy(field, axis_y_id)
    expected_x = (map_x[0] - map_origin[0], map_x[1] - map_origin[1])
    expected_y = (map_y[0] - map_origin[0], map_y[1] - map_origin[1])

    print("\n--- ИТОГ ---")
    print("по карте:  X {} -> {} = ({:+.2f}, {:+.2f}), Y {} -> {} = ({:+.2f}, {:+.2f})".format(
        origin_id, axis_x_id, expected_x[0], expected_x[1],
        origin_id, axis_y_id, expected_y[0], expected_y[1]))
    print("по замеру: X = ({:+.2f}, {:+.2f}), Y = ({:+.2f}, {:+.2f})".format(
        measured_x[0], measured_x[1], measured_y[0], measured_y[1]))
    origin_error = math.hypot(origin[0] - map_origin[0], origin[1] - map_origin[1])
    print("начало координат: по карте ({:+.2f}, {:+.2f}), по замеру ({:+.2f}, {:+.2f}), "
          "расхождение {:.2f} м".format(
              map_origin[0], map_origin[1], origin[0], origin[1], origin_error))

    ok, report = diagnose(measured_x, measured_y, expected_x, expected_y,
                          origin_error=origin_error)
    for line in report:
        print("  " + line)
    if not ok:
        print("\nЧто делать: скопировать config/field_map.txt в "
              "~/technic_ws/src/technic/aruco_pose/map/map1.txt и перезапустить ноду "
              "(rosnode kill /aruco_map).")
    return ok


# ═══════════════════════════════════════════════════════════════════════
#  САМОТЕСТ ЛОГИКИ ДИАГНОЗА (без ROS и дрона)
# ═══════════════════════════════════════════════════════════════════════


def _self_test() -> None:
    expected_x, expected_y = (1.0, 0.0), (0.0, 1.0)

    ok, report = diagnose((1.0, 0.0), (0.0, 1.0), expected_x, expected_y)
    assert ok, report
    assert any("ОСИ СОВПАДАЮТ" in line for line in report), report

    # Оси перепутаны местами: это же и зеркальность.
    ok, report = diagnose((0.0, 1.0), (1.0, 0.0), expected_x, expected_y)
    assert not ok
    assert any("ЗЕРКАЛЬНАЯ" in line for line in report), report
    assert any("ПЕРЕПУТАНЫ МЕСТАМИ" in line for line in report), report

    # Карта развёрнута на 180°: обе оси уехали одинаково, зеркальности нет.
    ok, report = diagnose((-1.0, 0.0), (0.0, -1.0), expected_x, expected_y)
    assert not ok
    assert not any("ЗЕРКАЛЬНАЯ" in line for line in report), report
    assert any("ПОВЁРНУТА" in line for line in report), report
    assert any("ось X направлена В ОБРАТНУЮ" in line for line in report), report

    # Поворот на 90° целиком (поле стоит боком к карте).
    ok, report = diagnose((0.0, 1.0), (-1.0, 0.0), expected_x, expected_y)
    assert not ok
    assert any("ПОВЁРНУТА" in line for line in report), report

    # Масштаб: замер в 1.4 раза больше — некалиброванная камера.
    ok, report = diagnose((1.4, 0.0), (0.0, 1.4), expected_x, expected_y)
    assert not ok
    assert any("МАСШТАБ" in line for line in report), report

    # Чужая карта: направления совпали, но начало отсчёта в другом месте.
    ok, report = diagnose((1.0, 0.0), (0.0, 1.0), expected_x, expected_y,
                          origin_error=1.7)
    assert not ok
    assert any("НАЧАЛО КООРДИНАТ не совпало" in line for line in report), report
    assert any("направления осей верные" in line for line in report), report

    # Реальный замер с площадки 2026-08-01: карта повёрнута на +92° И начало
    # другое — значит нода читает чужой файл, а не наш.
    ok, report = diagnose((-0.02, 0.90), (-1.20, -0.07), expected_x, expected_y,
                          origin_error=1.65)
    assert not ok
    assert any("чужая карта" in line for line in report), report
    assert not any("ЗЕРКАЛЬНАЯ" in line for line in report), report

    # Шум в пределах допуска не должен поднимать тревогу.
    ok, report = diagnose((0.98, 0.05), (-0.04, 1.03), expected_x, expected_y)
    assert ok, report

    # Дрон не двигали между замерами.
    ok, report = diagnose((0.0, 0.0), (0.0, 1.0), expected_x, expected_y)
    assert not ok and "не сдвинулся" in report[0], report

    # Усреднение и разброс.
    poses = iter([(0.0, 0.0, 1.0, 0.0), (0.2, 0.0, 1.0, 0.0), None])
    result = average_pose(lambda: next(poses), 3, 0.0, sleep_fn=lambda _s: None)
    assert result is not None
    mean, spread = result
    assert abs(mean[0] - 0.1) < 1e-9 and abs(spread - 0.1) < 1e-9, (mean, spread)
    assert average_pose(lambda: None, 3, 0.0, sleep_fn=lambda _s: None) is None

    # Полный проход сценария на подставных данных: карта верная.
    field = nav.read_field_map(
        str(Path(__file__).resolve().parent / "config" / "field_map.txt"))
    truth = {48: (0.0, 0.0), 41: (1.0, 0.0), 47: (0.0, 1.0)}
    order = iter([truth[48], truth[41], truth[47]])
    current = {"pose": (0.0, 0.0)}

    def fake_wait(_prompt: str) -> None:
        current["pose"] = next(order)

    def fake_read() -> Optional[Pose]:
        return (current["pose"][0], current["pose"][1], 1.5, 0.0)

    assert run(fake_read, field, (48, 41, 47), samples=2, pause=0.0,
               wait_fn=fake_wait, sleep_fn=lambda _s: None)

    print("debug_axes: самотест пройден")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Проверка соответствия осей X/Y карты aruco_map физическому полю")
    parser.add_argument("--map", default="config/field_map.txt", help="карта поля")
    parser.add_argument("--markers", default="41,34,40",
                        help="метки «начало, по оси X, по оси Y». По умолчанию 41,34,40 — "
                             "незакрытые метки рядом со стартом; метку 48 брать нельзя, "
                             "её закрывает площадка")
    parser.add_argument("--samples", type=int, default=10,
                        help="сколько замеров усреднять в каждой точке")
    parser.add_argument("--pause", type=float, default=0.2,
                        help="пауза между замерами, с")
    parser.add_argument("--self-test", action="store_true",
                        help="проверить логику диагноза без ROS и дрона")
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0

    markers = tuple(int(part) for part in args.markers.split(","))
    if len(markers) != 3:
        parser.error("--markers: нужно ровно три метки: начало, по оси X, по оси Y")

    field = nav.read_field_map(args.map)
    for mid in markers:
        nav.marker_xy(field, mid)  # понятная ошибка, если метки нет в карте

    read = make_pose_reader()
    return 0 if run(read, field, markers, samples=args.samples, pause=args.pause) else 1


if __name__ == "__main__":
    raise SystemExit(main())
