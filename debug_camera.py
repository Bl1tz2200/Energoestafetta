#!/usr/bin/env python3
"""Что видит камера: метки, масштаб кадра, оценка места и курса.

Моторы не включаются. Дрон держат руками на известной высоте над полем (или
кладут рядом с метками) и смотрят, сходится ли то, что распознаёт камера, с
картой поля. Проверяются три вещи, каждая из которых ломает полёт по-своему:

* **распознавание** — те ли ID видны и не путается ли словарь ArUco;
* **размер метки** — замеренный по кадру шаг решётки должен совпасть с
  картой. Замер идёт отношением пикселей (зазор между метками к стороне
  метки), поэтому от фокусного расстояния он не зависит и ловит именно
  неверный ``--marker-size`` или неверный физический размер метки. Ошибку
  фокусного (калибровку) ловит ``debug_axes.py`` — там дрон переносят на
  известное расстояние и сверяют с показаниями платформы;
* **место и курс** — расчёт по кадру сравнивается с позицией платформы
  (``aruco_map``), если она доступна. Расхождение показывает, кому верить.

Запуск на дроне::

    python3 debug_camera.py --map config/field_map.txt

Проверка расчётной части без ROS, дрона и камеры::

    python3 debug_camera.py --self-test
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


# ═══════════════════════════════════════════════════════════════════════
#  ROS — единственное место с rospy/cv_bridge
# ═══════════════════════════════════════════════════════════════════════


def make_readers(
    camera_topic: str, timeout: float, node_name: str = "debug_camera"
) -> Tuple[Callable[[], Any], Callable[[], Optional[Tuple[float, float]]]]:
    """(кадр, позиция от aruco_map). Импорт ROS спрятан здесь намеренно."""
    try:
        import rospy
        from technic import srv
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
    except ImportError as exc:
        raise RuntimeError(
            "Нужны rospy, technic, sensor_msgs и cv_bridge. Выполните source "
            "/opt/ros/noetic/setup.bash и source своего catkin_ws"
        ) from exc

    rospy.init_node(node_name)
    bridge = CvBridge()
    get_telemetry = rospy.ServiceProxy("get_telemetry", srv.GetTelemetry)

    def read_frame() -> Any:
        message = rospy.wait_for_message(camera_topic, Image, timeout=timeout)
        return bridge.imgmsg_to_cv2(message, "bgr8")

    def read_map_pose() -> Optional[Tuple[float, float]]:
        try:
            telemetry = get_telemetry(frame_id="aruco_map")
        except Exception:  # noqa: BLE001 - платформа может не отвечать, это не фатально
            return None
        x, y = float(telemetry.x), float(telemetry.y)
        return None if math.isnan(x) or math.isnan(y) else (x, y)

    return read_frame, read_map_pose


# ═══════════════════════════════════════════════════════════════════════
#  РАЗБОР ОДНОГО КАДРА
# ═══════════════════════════════════════════════════════════════════════


def describe_frame(
    seen: Dict[int, nav.Marker],
    field: Dict[int, Tuple[float, float]],
    centre: Tuple[float, float],
    marker_size: float,
    grid_step: float,
    *,
    scale_tolerance: float = 0.15,
) -> List[str]:
    """Строки отчёта по одному кадру: что видно, масштаб, место и курс."""
    lines: List[str] = []
    if not seen:
        return ["меток в кадре нет"]

    known = sorted(mid for mid in seen if mid in field)
    unknown = sorted(mid for mid in seen if mid not in field)
    lines.append("видно меток: {}{}".format(
        " ".join(str(mid) for mid in known) or "нет из карты",
        " | ЧУЖИЕ ID: " + " ".join(str(mid) for mid in unknown) if unknown else ""))
    for mid in known:
        marker = seen[mid]
        lines.append("  метка {:2d}: центр ({:6.1f}, {:6.1f}) px, сторона {:5.1f} px, "
                     "поворот {:+.0f}°".format(
                         mid, marker.x, marker.y, marker.side, math.degrees(marker.angle)))

    step = nav.measured_grid_step(seen, field, marker_size)
    if not step:
        lines.append("шаг решётки: нет пары соседних меток в кадре")
    else:
        ratio = step / grid_step
        note = "" if abs(ratio - 1.0) <= scale_tolerance else (
            "  <-- МАСШТАБ ВРЁТ в {:.2f} раза: калибровка камеры или --marker-size"
            .format(ratio))
        lines.append("шаг решётки: замер {:.2f} м, по карте {:.2f} м{}".format(
            step, grid_step, note))

    frame_yaw = nav.frame_yaw_from_pairs(seen, field, marker_size)
    if frame_yaw is None:
        lines.append("поворот поля: нужны две метки в кадре, курс не измерить")
        return lines
    lines.append("поворот поля в кадре: {:+.0f}°".format(math.degrees(frame_yaw)))

    base = nav.nearest_to_centre(seen, centre)
    if base is None or base.mid not in field:
        return lines
    position = nav.estimate_position(base, field, frame_yaw, centre, marker_size)
    lines.append("место по камере: ({:+.2f}, {:+.2f}), опора — метка {}".format(
        position[0], position[1], base.mid))
    return lines


def compare_with_map(
    camera_xy: Optional[Tuple[float, float]],
    map_xy: Optional[Tuple[float, float]],
    *,
    tolerance: float = 0.3,
) -> str:
    """Одна строка: сходятся ли оценка по камере и позиция платформы."""
    if map_xy is None:
        return "aruco_map: позиции нет (NaN) — платформа карту сейчас не видит"
    if camera_xy is None:
        return "aruco_map: ({:+.2f}, {:+.2f}); по камере места нет, сравнить не с чем".format(
            *map_xy)
    error = math.hypot(camera_xy[0] - map_xy[0], camera_xy[1] - map_xy[1])
    verdict = "сходятся" if error <= tolerance else (
        "РАСХОДЯТСЯ — карта на платформе не та же, что config/field_map.txt")
    return "aruco_map: ({:+.2f}, {:+.2f}), расхождение с камерой {:.2f} м — {}".format(
        map_xy[0], map_xy[1], error, verdict)


# ═══════════════════════════════════════════════════════════════════════
#  СЦЕНАРИЙ
# ═══════════════════════════════════════════════════════════════════════


def run(
    read_frame: Callable[[], Any],
    detect: Callable[[Any], Dict[int, nav.Marker]],
    read_map_pose: Callable[[], Optional[Tuple[float, float]]],
    field: Dict[int, Tuple[float, float]],
    marker_size: float,
    *,
    cycles: int = 0,
    pause: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    grid_step = nav.field_step(field)
    count = 0
    while cycles <= 0 or count < cycles:
        count += 1
        print("--- кадр {} ---".format(count))
        try:
            image = read_frame()
        except Exception as exc:  # noqa: BLE001 - следующий кадр может быть хорошим
            print("кадр не получен: {}".format(exc))
            sleep_fn(pause)
            continue
        if image is None or getattr(image, "ndim", 0) != 3:
            print("кадр негодный (не BGR)")
            sleep_fn(pause)
            continue

        height, width = image.shape[:2]
        centre = (width / 2.0, height / 2.0)
        seen = detect(image)
        for line in describe_frame(seen, field, centre, marker_size, grid_step):
            print("  " + line)

        camera_xy = None
        frame_yaw = nav.frame_yaw_from_pairs(seen, field, marker_size)
        base = nav.nearest_to_centre(seen, centre)
        if frame_yaw is not None and base is not None and base.mid in field:
            camera_xy = nav.estimate_position(base, field, frame_yaw, centre, marker_size)
        print("  " + compare_with_map(camera_xy, read_map_pose()))
        sleep_fn(pause)


# ═══════════════════════════════════════════════════════════════════════
#  САМОТЕСТ (без ROS, камеры и OpenCV)
# ═══════════════════════════════════════════════════════════════════════


def _fake_seen(
    field: Dict[int, Tuple[float, float]],
    ids: Sequence[int],
    drone: Tuple[float, float],
    frame_yaw: float,
    side: float,
    centre: Tuple[float, float],
    marker_size: float,
) -> Dict[int, nav.Marker]:
    """Как метки легли бы в кадр при таком положении дрона."""
    scale = marker_size / side
    seen: Dict[int, nav.Marker] = {}
    for mid in ids:
        mx, my = field[mid]
        forward, left = nav.field_to_body(mx - drone[0], my - drone[1], frame_yaw)
        seen[mid] = nav.Marker(mid, centre[0] - left / scale, centre[1] - forward / scale,
                               side, 0.0)
    return seen


def _self_test() -> None:
    field = nav.read_field_map(
        str(Path(__file__).resolve().parent / "config" / "field_map.txt"))
    centre, marker_size, side = (640.0, 360.0), 0.33, 120.0
    grid = nav.field_step(field)
    drone, yaw = nav.marker_xy(field, 41), 0.4

    seen = _fake_seen(field, (41, 47, 48), drone, yaw, side, centre, marker_size)
    lines = describe_frame(seen, field, centre, marker_size, grid)
    text = "\n".join(lines)
    assert "видно меток: 41 47 48" in text, text
    assert "МАСШТАБ ВРЁТ" not in text, text
    assert "место по камере: (+1.00, +0.00)" in text, text

    # Неверный размер метки: все расстояния врут в одно и то же число раз.
    lines = describe_frame(seen, field, centre, marker_size * 1.5, grid)
    assert any("МАСШТАБ ВРЁТ в 1.50 раза" in line for line in lines), lines

    # Одна метка: курс не измерить, места нет.
    lines = describe_frame({41: seen[41]}, field, centre, marker_size, grid)
    assert any("нужны две метки" in line for line in lines), lines

    # Чужие ID в кадре (не из карты поля) должны быть названы.
    stranger = dict(seen)
    stranger[99] = nav.Marker(99, 10.0, 10.0, side, 0.0)
    lines = describe_frame(stranger, field, centre, marker_size, grid)
    assert any("ЧУЖИЕ ID: 99" in line for line in lines), lines

    assert "меток в кадре нет" in describe_frame({}, field, centre, marker_size, grid)[0]

    # Сравнение с платформой.
    assert "сходятся" in compare_with_map((1.0, 0.0), (1.05, 0.02))
    assert "РАСХОДЯТСЯ" in compare_with_map((1.0, 0.0), (3.0, 0.0))
    assert "позиции нет" in compare_with_map((1.0, 0.0), None)
    assert "сравнить не с чем" in compare_with_map(None, (1.0, 0.0))

    # Полный проход по одному кадру с подставными читалками.
    class FakeImage:
        ndim = 3
        shape = (720, 1280, 3)

    printed: List[str] = []
    run(lambda: FakeImage(), lambda _img: seen, lambda: (1.0, 0.0), field, marker_size,
        cycles=1, pause=0.0, sleep_fn=lambda _s: None)
    del printed

    print("debug_camera: самотест пройден")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Отладка камеры: метки, масштаб кадра, место и курс по кадру")
    parser.add_argument("--map", default="config/field_map.txt", help="карта поля")
    parser.add_argument("--marker-size", type=float, default=0.33,
                        help="сторона метки, м — по ней считается масштаб кадра")
    parser.add_argument("--aruco-dict", default="DICT_4X4_50", help="словарь ArUco")
    parser.add_argument("--camera-topic", default="main_camera/image_raw")
    parser.add_argument("--frame-timeout", type=float, default=2.0)
    parser.add_argument("--cycles", type=int, default=0,
                        help="сколько кадров разобрать (0 — до Ctrl+C)")
    parser.add_argument("--pause", type=float, default=1.0, help="пауза между кадрами, с")
    parser.add_argument("--self-test", action="store_true",
                        help="проверить расчётную часть без ROS и камеры")
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0

    field = nav.read_field_map(args.map)
    detector = nav.MarkerDetector(args.aruco_dict, max_marker_id=max(field))
    read_frame, read_map_pose = make_readers(args.camera_topic, args.frame_timeout)
    try:
        run(read_frame, detector.detect, read_map_pose, field, args.marker_size,
            cycles=args.cycles, pause=args.pause)
    except KeyboardInterrupt:
        print("отладка окончена")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
