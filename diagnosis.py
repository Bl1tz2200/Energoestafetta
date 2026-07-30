#!/usr/bin/env python3
"""Диагностика Skyris Technic 6S на площадке — БЕЗ включения моторов.

Проверяет связь со всеми подсистемами, которые нужны полётным сценариям
(``bvs1_flight.py``/``bvs2_flight.py``), не вызывая ``navigate()``,
``arming(True)`` или ``land()`` — двигатели не включаются вообще, скрипт
безопасен на стенде без пропеллеров и на площадке до первого взлёта.

Что проверяется (каждый пункт независим — сбой одного не прерывает
остальные, чтобы получить полную картину, а не первую же ошибку):

1. Карта поля — `config/field_map.txt` читается и парсится (`flight_core.read_map`).
2. Базовая связь ROS 1/MAVROS — `get_telemetry(frame_id='body')`.
3. Локализация по карте поля — `get_telemetry(frame_id='aruco_map')`
   (docs.skyris.ru/technic6S/ArucoMap.html); читает текущую позицию, не
   двигая дрон.
4. Видимость меток поля — топик `aruco_detect/markers` (пакет `aruco_pose`):
   если меток не видно, `aruco_map` из проверки выше не может быть верным —
   это самая частая причина «ошибок с aruco_map» на площадке.
5. Лазерный дальномер — топик `/rangefinder/range`.
6. Камера — топик `main_camera/image_raw`, и если установлены
   `opencv-contrib-python`/`numpy` — дополнительно прогон через
   `energy_relay_vision.StationVision` на полученном кадре.
7. LED-лента — короткая тестовая последовательность цветов
   (`led_interface.set_led`); безопасна без пропеллеров.
8. Захват груза — ТОЛЬКО если передан ``--test-gripper`` (двигает
   сервопривод, поэтому по умолчанию выключено).

Запуск на дроне::

    python3 diagnosis.py --map config/field_map.txt

Проверка модуля без ROS 1/дрона::

    python3 diagnosis.py --self-test
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import flight_core as fc
import led_interface as led


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _run_check(name: str, action: Callable[[], str]) -> CheckResult:
    """Выполнить одну проверку, изолируя её сбой от остальных.

    Диагностика должна вернуть полную картину по всем подсистемам, поэтому
    падение одной проверки не должно обрывать скрипт и скрывать результаты
    остальных (в отличие от обычного полётного сценария).
    """
    try:
        detail = action()
    except Exception as exc:  # noqa: BLE001 - хотим отчитаться о любой ошибке
        return CheckResult(name, False, "{}: {}".format(type(exc).__name__, exc))
    return CheckResult(name, True, detail)


def read_marker_count_ros(timeout: float) -> int:
    """Прочитать число видимых меток поля с топика ``aruco_detect/markers``
    (пакет ``aruco_pose``) — сколько ArUco-меток сейчас видит камера."""
    import rospy
    from aruco_pose.msg import MarkerArray

    message = rospy.wait_for_message("aruco_detect/markers", MarkerArray, timeout=timeout)
    return len(message.markers)


def read_camera_frame_ros(timeout: float) -> Any:
    """Прочитать один BGR-кадр с ``main_camera/image_raw`` через ``cv_bridge``
    (тот же топик, что использует ``energy_relay_vision.TechnicROS1Vision``)."""
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image

    message = rospy.wait_for_message("main_camera/image_raw", Image, timeout=timeout)
    return CvBridge().imgmsg_to_cv2(message, "bgr8")


def run_diagnosis(
    *,
    map_path: str,
    get_telemetry: Callable[..., Any],
    marker_count_reader: Callable[[float], int],
    rangefinder_reader: Callable[[float], float],
    set_led_fn: Callable[[str, Any], None],
    image_reader: Optional[Callable[[float], Any]] = None,
    vision_factory: Optional[Callable[[], Any]] = None,
    gripper_actions: Optional[Tuple[Callable[[], None], Callable[[], None]]] = None,
    rangefinder_timeout: float = 2.0,
    markers_timeout: float = 2.0,
    camera_timeout: float = 2.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> List[CheckResult]:
    """Прогнать все проверки и вернуть список результатов.

    Все внешние вызовы приходят через параметры (по образцу
    ``flight_core.FlightProxies``), поэтому логику можно проверить
    самотестом с заглушками без ROS 1/дрона (см. ``_self_test``).
    """
    results: List[CheckResult] = []

    def _check_field_map() -> str:
        markers = fc.read_map(map_path)
        if not markers:
            raise ValueError("карта поля пуста")
        return "меток в карте: {}".format(len(markers))

    results.append(_run_check("Карта поля ({})".format(map_path), _check_field_map))

    def _check_body_telemetry() -> str:
        telemetry = get_telemetry(frame_id="body")
        return "armed={} x={:.2f} y={:.2f} z={:.2f} (frame=body)".format(
            getattr(telemetry, "armed", "?"), telemetry.x, telemetry.y, telemetry.z
        )

    results.append(_run_check("Телеметрия ROS 1/MAVROS (frame=body)", _check_body_telemetry))

    def _check_aruco_map_telemetry() -> str:
        telemetry = get_telemetry(frame_id="aruco_map")
        return "x={:.2f} y={:.2f} z={:.2f} (frame=aruco_map)".format(
            telemetry.x, telemetry.y, telemetry.z
        )

    results.append(_run_check("Локализация по aruco_map", _check_aruco_map_telemetry))

    def _check_markers_visible() -> str:
        count = marker_count_reader(markers_timeout)
        if count == 0:
            raise RuntimeError(
                "камера не видит ни одной метки поля — aruco_map выше не может "
                "быть верным без этого"
            )
        return "видно меток: {}".format(count)

    results.append(_run_check("Видимость меток (aruco_detect/markers)", _check_markers_visible))

    def _check_rangefinder() -> str:
        distance = rangefinder_reader(rangefinder_timeout)
        return "дальность: {:.2f} м".format(distance)

    results.append(_run_check("Дальномер (/rangefinder/range)", _check_rangefinder))

    frame_box: List[Any] = []
    if image_reader is not None:

        def _check_camera() -> str:
            frame = image_reader(camera_timeout)
            frame_box.append(frame)
            height, width = frame.shape[:2]
            return "кадр {}x{}".format(width, height)

        results.append(_run_check("Камера (main_camera/image_raw)", _check_camera))

        def _check_vision() -> str:
            if not frame_box:
                raise RuntimeError("кадр не получен — проверка камеры выше провалена")
            factory = vision_factory or _default_vision_factory
            detector = factory()
            result = detector.detect(frame_box[0])
            return "меток: {}, станций: {}".format(len(result.markers), len(result.stations))

        results.append(_run_check("Распознавание (energy_relay_vision)", _check_vision))

    def _check_led() -> str:
        for pattern, color in (("solid", "red"), ("solid", "green"), ("solid", "blue"), ("solid", "off")):
            set_led_fn(pattern, color)
            sleep_fn(0.5)
        return "лента прогнала red/green/blue/off"

    results.append(_run_check("LED-лента", _check_led))

    if gripper_actions is not None:
        open_fn, close_fn = gripper_actions

        def _check_gripper() -> str:
            open_fn()
            sleep_fn(0.5)
            close_fn()
            return "захват открыт/закрыт без ошибок"

        results.append(_run_check("Захват груза", _check_gripper))

    return results


def _default_vision_factory() -> Any:
    import energy_relay_vision as erv

    return erv.StationVision()


def _print_report(results: List[CheckResult]) -> int:
    for result in results:
        status = "OK  " if result.ok else "FAIL"
        print("[{}] {} — {}".format(status, result.name, result.detail))

    failed = [result for result in results if not result.ok]
    print()
    print("Итог: {}/{} проверок пройдено".format(len(results) - len(failed), len(results)))
    if failed:
        print("Провалено: " + ", ".join(result.name for result in failed))
    return len(failed)


def _self_test() -> int:
    import flight_test_support as fts

    telemetry = fts.FakeTelemetry(x=2.0, y=5.0, z=0.0, armed=False)
    led_calls, _led_backend = fts.record_led_calls()

    results = run_diagnosis(
        map_path="config/field_map.txt",
        get_telemetry=lambda **_kwargs: telemetry,
        marker_count_reader=lambda _timeout: 6,
        rangefinder_reader=lambda _timeout: 0.85,
        set_led_fn=led.set_led,
        sleep_fn=lambda _duration: None,
    )
    by_name = {result.name: result for result in results}
    assert by_name["Карта поля (config/field_map.txt)"].ok is True
    assert by_name["Телеметрия ROS 1/MAVROS (frame=body)"].ok is True
    assert by_name["Локализация по aruco_map"].ok is True
    assert by_name["Видимость меток (aruco_detect/markers)"].ok is True
    assert "видно меток: 6" in by_name["Видимость меток (aruco_detect/markers)"].detail
    assert by_name["Дальномер (/rangefinder/range)"].ok is True
    assert by_name["LED-лента"].ok is True
    assert ("solid", (0, 0, 0)) == led_calls[-1] or ("fill", (0, 0, 0)) == led_calls[-1]

    # Отсутствие меток в кадре — RuntimeError -> FAIL, но остальные проверки
    # всё равно выполняются (падение одной проверки не рушит остальные)
    results_no_markers = run_diagnosis(
        map_path="config/field_map.txt",
        get_telemetry=lambda **_kwargs: telemetry,
        marker_count_reader=lambda _timeout: 0,
        rangefinder_reader=lambda _timeout: 0.85,
        set_led_fn=led.set_led,
        sleep_fn=lambda _duration: None,
    )
    by_name_no_markers = {result.name: result for result in results_no_markers}
    assert by_name_no_markers["Видимость меток (aruco_detect/markers)"].ok is False
    assert by_name_no_markers["Дальномер (/rangefinder/range)"].ok is True

    # Некорректный путь к карте -> FAIL только по этой проверке
    results_bad_map = run_diagnosis(
        map_path="does-not-exist.txt",
        get_telemetry=lambda **_kwargs: telemetry,
        marker_count_reader=lambda _timeout: 6,
        rangefinder_reader=lambda _timeout: 0.85,
        set_led_fn=led.set_led,
        sleep_fn=lambda _duration: None,
    )
    by_name_bad_map = {result.name: result for result in results_bad_map}
    assert by_name_bad_map["Карта поля (does-not-exist.txt)"].ok is False
    assert by_name_bad_map["Телеметрия ROS 1/MAVROS (frame=body)"].ok is True

    # Камера + распознавание: заглушка кадра/детектора, без cv2/numpy
    class FakeFrame:
        shape = (480, 640, 3)

    class FakeVisionResult:
        markers = (1, 2)
        stations = (1,)

    class FakeDetector:
        def detect(self, _frame: Any) -> FakeVisionResult:
            return FakeVisionResult()

    results_with_camera = run_diagnosis(
        map_path="config/field_map.txt",
        get_telemetry=lambda **_kwargs: telemetry,
        marker_count_reader=lambda _timeout: 6,
        rangefinder_reader=lambda _timeout: 0.85,
        set_led_fn=led.set_led,
        image_reader=lambda _timeout: FakeFrame(),
        vision_factory=lambda: FakeDetector(),
        sleep_fn=lambda _duration: None,
    )
    by_name_camera = {result.name: result for result in results_with_camera}
    assert by_name_camera["Камера (main_camera/image_raw)"].ok is True
    assert "640x480" in by_name_camera["Камера (main_camera/image_raw)"].detail  # width x height
    assert by_name_camera["Распознавание (energy_relay_vision)"].ok is True
    assert "меток: 2, станций: 1" in by_name_camera["Распознавание (energy_relay_vision)"].detail

    # Захват груза — только если явно передан gripper_actions
    gripper_calls: List[str] = []
    results_with_gripper = run_diagnosis(
        map_path="config/field_map.txt",
        get_telemetry=lambda **_kwargs: telemetry,
        marker_count_reader=lambda _timeout: 6,
        rangefinder_reader=lambda _timeout: 0.85,
        set_led_fn=led.set_led,
        gripper_actions=(
            lambda: gripper_calls.append("open"),
            lambda: gripper_calls.append("close"),
        ),
        sleep_fn=lambda _duration: None,
    )
    by_name_gripper = {result.name: result for result in results_with_gripper}
    assert by_name_gripper["Захват груза"].ok is True
    assert gripper_calls == ["open", "close"]

    exit_code = _print_report(results)
    assert exit_code == 0

    print("SELF-TEST: OK")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Диагностика Skyris Technic 6S без включения моторов (Энергоэстафета)"
    )
    parser.add_argument("--map", default="config/field_map.txt", help="путь к карте поля")
    parser.add_argument("--node-name", default="diagnosis")
    parser.add_argument("--rangefinder-timeout", type=float, default=2.0)
    parser.add_argument("--markers-timeout", type=float, default=2.0)
    parser.add_argument("--camera-timeout", type=float, default=2.0)
    parser.add_argument(
        "--skip-camera",
        action="store_true",
        help="не проверять камеру/распознавание (например, если opencv не установлен)",
    )
    parser.add_argument(
        "--test-gripper",
        action="store_true",
        help="дополнительно открыть/закрыть захват — двигает сервопривод, по умолчанию выключено",
    )
    parser.add_argument("--gripper-pin", type=int, default=17)
    parser.add_argument("--gripper-open-pulse", type=int, default=1000)
    parser.add_argument("--gripper-close-pulse", type=int, default=2000)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="проверить логику диагностики без ROS 1/дрона",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.self_test:
        return _self_test()

    proxies = fc.init_flight(args.node_name)
    led.use_technic_ros1_backend()

    gripper_actions = None
    gripper_backend = None
    if args.test_gripper:
        import gripper_control as gripper

        gripper_backend = gripper.use_technic_gpio_gripper(
            args.gripper_pin,
            open_pulse=args.gripper_open_pulse,
            close_pulse=args.gripper_close_pulse,
        )
        gripper_actions = (gripper.gripper_open, gripper.gripper_close)

    try:
        results = run_diagnosis(
            map_path=args.map,
            get_telemetry=proxies.get_telemetry,
            marker_count_reader=read_marker_count_ros,
            rangefinder_reader=fc.read_rangefinder_ros,
            set_led_fn=led.set_led,
            image_reader=None if args.skip_camera else read_camera_frame_ros,
            gripper_actions=gripper_actions,
            rangefinder_timeout=args.rangefinder_timeout,
            markers_timeout=args.markers_timeout,
            camera_timeout=args.camera_timeout,
        )
    finally:
        led.close_led_backend()
        if gripper_backend is not None:
            gripper_backend.shutdown()

    return _print_report(results)


if __name__ == "__main__":
    raise SystemExit(main())
