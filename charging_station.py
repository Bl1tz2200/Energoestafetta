#!/usr/bin/env python3
"""Автономный контроллер зарядной станции «Энергоэстафеты».

Файл предназначен для Raspberry Pi станции и совместим с интерфейсами
репозитория Bl1tz2200/Energoestafetta:

* ``energy_relay_vision.StationVision`` распознаёт цвет ленты БВС;
* ``station_protocol.Signal.LANDING_GRANTED`` задаёт сигнал разрешения;
* ``led_interface.set_led(pattern, color)`` управляет лентой станции.

Алгоритм:

1. Камера ищет заданный экспертами цвет ленты БВС.
2. Цвет должен быть подтверждён несколько кадров подряд.
3. Станция включает зелёную ленту — приглашение на посадку/зарядку.
4. После заданного времени лента выключается.
5. Перед новым приглашением станция ждёт исчезновения БВС из кадра.

Файл следует положить в корень Energoestafetta рядом с ``lib/`` либо передать
путь к репозиторию через ``--repo-root``. GitHub/ROS станции не требуются.

Проверка без камеры и Raspberry Pi:

    python3 charging_station.py --repo-root . --self-test

Полевой запуск сначала с безопасным выводом команд в консоль:

    python3 charging_station.py --repo-root . --camera 0 \
        --drone-colors red --led-backend console --once

Запуск с лентой WS281x на GPIO18:

    sudo python3 charging_station.py --repo-root . --camera 0 \
        --drone-colors red --led-backend ws281x \
        --led-pin 18 --led-count 12 --brightness 128
"""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, Optional, Sequence, Tuple


RGB = Tuple[int, int, int]
HSVRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


# Красный/зелёный совпадают с energy_relay_vision.DEFAULT_COLOR_RANGES.
# Жёлтый/синий добавлены, потому что конкретный цвет БВС назначают эксперты.
DRONE_COLOR_RANGES: Dict[str, Tuple[HSVRange, ...]] = {
    "red": (
        ((0, 100, 60), (12, 255, 255)),
        ((168, 100, 60), (179, 255, 255)),
    ),
    "green": (
        ((35, 70, 50), (90, 255, 255)),
    ),
    "yellow": (
        ((18, 90, 70), (38, 255, 255)),
    ),
    "blue": (
        ((90, 80, 50), (135, 255, 255)),
    ),
}


@dataclass(frozen=True)
class DroneDetection:
    """Лучшее цветовое пятно, принятое за ленту подлетевшего БВС."""

    color: str
    confidence: float
    frame_ratio: float
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class StationConfig:
    """Параметры фильтрации и конечного автомата станции."""

    expected_colors: Tuple[str, ...] = ("red",)
    required_frames: int = 5
    reset_frames: int = 10
    min_confidence: float = 0.45
    min_frame_ratio: float = 0.003
    max_frame_ratio: float = 0.35
    poll_interval_s: float = 0.05
    grant_hold_s: float = 15.0
    once: bool = False


class OpenCVCamera:
    """Камера Raspberry Pi/USB, доступная через OpenCV VideoCapture."""

    def __init__(
        self,
        index: int,
        *,
        width: int = 0,
        height: int = 0,
        warmup_s: float = 1.0,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Не найден OpenCV. Установите opencv-contrib-python и numpy."
            ) from exc

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError("Не удалось открыть камеру с индексом {}".format(index))

        if width > 0:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if warmup_s > 0:
            sleep(warmup_s)

    def read(self) -> Any:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("Камера не вернула кадр")
        return frame

    def close(self) -> None:
        self._capture.release()


class WS281xBackend:
    """LEDBackend для ленты станции через библиотеку rpi_ws281x."""

    def __init__(
        self,
        led_count: int,
        pin: int,
        *,
        brightness: int = 128,
        frequency_hz: int = 800_000,
        dma: int = 10,
        invert: bool = False,
        channel: int = 0,
    ) -> None:
        if led_count < 2:
            raise ValueError("led_count должен быть не меньше 2")
        if not 0 <= brightness <= 255:
            raise ValueError("brightness должен лежать в диапазоне 0..255")

        try:
            from rpi_ws281x import Color, PixelStrip
        except ImportError as exc:
            raise RuntimeError(
                "Для --led-backend ws281x установите библиотеку rpi-ws281x"
            ) from exc

        self._led_count = int(led_count)
        self._color_factory = Color
        self._strip = PixelStrip(
            self._led_count,
            pin,
            frequency_hz,
            dma,
            invert,
            brightness,
            channel,
        )
        self._strip.begin()

    @property
    def led_count(self) -> int:
        return self._led_count

    def set_effect(self, effect: str, color: RGB) -> None:
        # Станции по текущему station_protocol нужны только fill/off.
        if effect != "fill":
            raise RuntimeError(
                "WS281xBackend станции поддерживает только solid/fill"
            )
        self.set_pixels((color,) * self._led_count)

    def set_pixels(self, colors: Sequence[RGB]) -> None:
        if len(colors) != self._led_count:
            raise ValueError(
                "Ожидалось {} цветов, получено {}".format(
                    self._led_count, len(colors)
                )
            )
        for index, (red, green, blue) in enumerate(colors):
            self._strip.setPixelColor(
                index, self._color_factory(red, green, blue)
            )
        self._strip.show()

    def close(self) -> None:
        self.set_pixels(((0, 0, 0),) * self._led_count)


def _load_repo_modules(repo_root: Path) -> Tuple[Any, Any]:
    """Загрузить неизменённые модули Energoestafetta из ``repo_root/lib``."""
    lib_dir = repo_root.resolve() / "lib"
    required = (
        lib_dir / "led_interface.py",
        lib_dir / "station_protocol.py",
        lib_dir / "energy_relay_vision.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "В --repo-root не найдены файлы Energoestafetta: "
            + ", ".join(missing)
        )

    lib_value = str(lib_dir)
    if lib_value not in sys.path:
        sys.path.insert(0, lib_value)

    led = importlib.import_module("led_interface")
    protocol = importlib.import_module("station_protocol")
    return led, protocol


def _load_vision_module() -> Any:
    try:
        return importlib.import_module("energy_relay_vision")
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from exc


def select_drone(
    result: Any,
    config: StationConfig,
) -> Optional[DroneDetection]:
    """Выбрать наиболее уверенное допустимое цветовое пятно."""
    candidates = []
    for item in result.stations:
        if item.color not in config.expected_colors:
            continue
        if item.confidence < config.min_confidence:
            continue
        if not config.min_frame_ratio <= item.frame_ratio <= config.max_frame_ratio:
            continue
        candidates.append(item)

    if not candidates:
        return None

    best = max(candidates, key=lambda item: (item.confidence, item.frame_ratio))
    return DroneDetection(
        color=best.color,
        confidence=float(best.confidence),
        frame_ratio=float(best.frame_ratio),
        bbox=tuple(int(value) for value in best.bbox),
    )


def _validate_config(config: StationConfig) -> None:
    if not config.expected_colors:
        raise ValueError("Нужно указать хотя бы один ожидаемый цвет БВС")
    unknown = sorted(set(config.expected_colors) - set(DRONE_COLOR_RANGES))
    if unknown:
        raise ValueError("Неизвестные цвета БВС: " + ", ".join(unknown))
    if config.required_frames <= 0 or config.reset_frames <= 0:
        raise ValueError("required_frames и reset_frames должны быть больше нуля")
    if not 0.0 <= config.min_confidence <= 1.0:
        raise ValueError("min_confidence должен лежать в диапазоне 0..1")
    if config.min_frame_ratio <= 0:
        raise ValueError("min_frame_ratio должен быть больше нуля")
    if config.max_frame_ratio <= config.min_frame_ratio:
        raise ValueError("max_frame_ratio должен быть больше min_frame_ratio")
    if config.poll_interval_s < 0 or config.grant_hold_s < 0:
        raise ValueError("Интервалы времени не могут быть отрицательными")


def run_station(
    *,
    frame_reader: Callable[[], Any],
    detector: Any,
    grant_sender: Callable[[DroneDetection], None],
    turn_off: Callable[[], None],
    config: StationConfig,
    event_writer: Callable[[str, Dict[str, Any]], None],
    should_stop: Callable[[], bool] = lambda: False,
    sleep_fn: Callable[[float], None] = sleep,
) -> int:
    """Запустить автономный цикл станции.

    Возвращает количество отправленных приглашений. Внешние зависимости
    переданы явно, поэтому конечный автомат проверяется без камеры/RPi.
    """
    _validate_config(config)
    streak_color: Optional[str] = None
    streak_count = 0
    grants = 0

    while not should_stop():
        frame = frame_reader()
        detection = select_drone(detector.detect(frame), config)

        if detection is None:
            streak_color = None
            streak_count = 0
        elif detection.color == streak_color:
            streak_count += 1
        else:
            streak_color = detection.color
            streak_count = 1

        if detection is not None and streak_count >= config.required_frames:
            grant_sender(detection)
            grants += 1
            event_writer(
                "landing_granted",
                {
                    "drone_color": detection.color,
                    "confidence": round(detection.confidence, 4),
                    "frame_ratio": round(detection.frame_ratio, 6),
                    "bbox": detection.bbox,
                },
            )

            if config.grant_hold_s > 0:
                sleep_fn(config.grant_hold_s)
            turn_off()

            if config.once:
                return grants

            # Защита от повторного приглашения тому же БВС: новый цикл
            # разрешается только после нескольких кадров без его цвета.
            absent_frames = 0
            while absent_frames < config.reset_frames and not should_stop():
                frame = frame_reader()
                current = select_drone(detector.detect(frame), config)
                absent_frames = absent_frames + 1 if current is None else 0
                if config.poll_interval_s > 0:
                    sleep_fn(config.poll_interval_s)

            streak_color = None
            streak_count = 0
            event_writer("station_ready", {"grants": grants})

        if config.poll_interval_s > 0:
            sleep_fn(config.poll_interval_s)

    return grants


def _json_event(name: str, payload: Dict[str, Any]) -> None:
    print(
        json.dumps(
            {"event": name, "time": time.time(), **payload},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _parse_colors(value: str) -> Tuple[str, ...]:
    colors = tuple(
        part.strip().lower()
        for part in value.split(",")
        if part.strip()
    )
    if not colors:
        raise argparse.ArgumentTypeError("Список цветов пуст")
    unknown = sorted(set(colors) - set(DRONE_COLOR_RANGES))
    if unknown:
        raise argparse.ArgumentTypeError(
            "Неизвестные цвета: {}; допустимы: {}".format(
                ", ".join(unknown), ", ".join(sorted(DRONE_COLOR_RANGES))
            )
        )
    return colors


def _self_test(led: Any, protocol: Any) -> int:
    """Проверить фильтрацию кадров и совместимость с протоколом GitHub."""

    @dataclass(frozen=True)
    class FakeStation:
        color: str
        confidence: float
        frame_ratio: float
        bbox: Tuple[int, int, int, int] = (10, 20, 30, 40)

    @dataclass(frozen=True)
    class FakeResult:
        stations: Tuple[FakeStation, ...]

    class FakeDetector:
        @staticmethod
        def detect(frame: FakeResult) -> FakeResult:
            return frame

    frames = iter(
        [
            FakeResult((FakeStation("red", 0.2, 0.02),)),  # низкая уверенность
            FakeResult((FakeStation("red", 0.9, 0.02),)),
            FakeResult(tuple()),  # разрыв серии сбрасывает счётчик
            FakeResult((FakeStation("red", 0.8, 0.02),)),
            FakeResult((FakeStation("red", 0.85, 0.02),)),
            FakeResult((FakeStation("red", 0.95, 0.02),)),
        ]
    )

    led_calls = []

    def effect_writer(effect: str, color: RGB) -> None:
        led_calls.append((effect, color))

    backend = led.CallbackBackend(
        12,
        effect_writer=effect_writer,
        pixels_writer=lambda _colors: None,
    )
    led.configure_backend(backend)

    detections = []

    def grant_sender(detection: DroneDetection) -> None:
        detections.append(detection)
        protocol.send_signal(
            protocol.Signal.LANDING_GRANTED,
            set_led_fn=led.set_led,
        )

    events = []
    grants = run_station(
        frame_reader=lambda: next(frames),
        detector=FakeDetector(),
        grant_sender=grant_sender,
        turn_off=led.turn_off,
        config=StationConfig(
            expected_colors=("red",),
            required_frames=3,
            reset_frames=2,
            min_confidence=0.45,
            grant_hold_s=0.0,
            poll_interval_s=0.0,
            once=True,
        ),
        event_writer=lambda name, payload: events.append((name, payload)),
        sleep_fn=lambda _duration: None,
    )

    assert grants == 1
    assert len(detections) == 1
    assert detections[0].color == "red"
    assert events[0][0] == "landing_granted"
    assert led_calls == [
        ("fill", (0, 255, 0)),  # LANDING_GRANTED из station_protocol
        ("fill", (0, 0, 0)),  # turn_off
    ]

    try:
        _validate_config(StationConfig(expected_colors=("purple",)))
    except ValueError:
        pass
    else:
        raise AssertionError("Неизвестный цвет должен отклоняться")

    print("SELF-TEST: OK")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Автономный контроллер зарядной станции Энергоэстафеты"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="корень Energoestafetta, содержащий lib/ (по умолчанию папка файла)",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--camera-warmup", type=float, default=1.0)
    parser.add_argument(
        "--drone-colors",
        type=_parse_colors,
        default=("red",),
        help="цвета ленты БВС через запятую: red,green,yellow,blue",
    )
    parser.add_argument("--required-frames", type=int, default=5)
    parser.add_argument("--reset-frames", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--min-area-ratio", type=float, default=0.003)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--grant-hold", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--dictionary",
        default="DICT_4X4_50",
        help="словарь cv2.aruco, совпадающий с настройкой поля",
    )
    parser.add_argument(
        "--led-backend",
        choices=("console", "ws281x"),
        default="console",
        help="console для проверки, ws281x для физической ленты Raspberry Pi",
    )
    parser.add_argument("--led-count", type=int, default=12)
    parser.add_argument("--led-pin", type=int, default=18)
    parser.add_argument("--brightness", type=int, default=128)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="проверить конечный автомат без камеры и Raspberry Pi",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    led, protocol = _load_repo_modules(args.repo_root)

    if args.self_test:
        return _self_test(led, protocol)

    config = StationConfig(
        expected_colors=args.drone_colors,
        required_frames=args.required_frames,
        reset_frames=args.reset_frames,
        min_confidence=args.min_confidence,
        min_frame_ratio=args.min_area_ratio,
        max_frame_ratio=args.max_area_ratio,
        poll_interval_s=args.poll_interval,
        grant_hold_s=args.grant_hold,
        once=args.once,
    )
    _validate_config(config)

    if args.led_backend == "console":
        led_backend = led.ConsoleBackend(
            led_count=args.led_count,
            verbose=True,
        )
    else:
        led_backend = WS281xBackend(
            args.led_count,
            args.led_pin,
            brightness=args.brightness,
        )
    led.configure_backend(led_backend)

    vision = _load_vision_module()
    selected_ranges = {
        color: DRONE_COLOR_RANGES[color]
        for color in config.expected_colors
    }
    detector = vision.StationVision(
        dictionary_name=args.dictionary,
        allowed_marker_ids=None,
        min_color_area_ratio=config.min_frame_ratio,
        color_ranges=selected_ranges,
    )
    camera = OpenCVCamera(
        args.camera,
        width=args.width,
        height=args.height,
        warmup_s=args.camera_warmup,
    )

    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def grant_sender(detection: DroneDetection) -> None:
        _json_event(
            "drone_confirmed",
            {
                "drone_color": detection.color,
                "confidence": round(detection.confidence, 4),
            },
        )
        protocol.send_signal(
            protocol.Signal.LANDING_GRANTED,
            set_led_fn=led.set_led,
        )

    _json_event(
        "station_started",
        {
            "camera": args.camera,
            "expected_colors": config.expected_colors,
            "led_backend": args.led_backend,
        },
    )
    try:
        run_station(
            frame_reader=camera.read,
            detector=detector,
            grant_sender=grant_sender,
            turn_off=led.turn_off,
            config=config,
            event_writer=_json_event,
            should_stop=stop_event.is_set,
        )
    finally:
        camera.close()
        try:
            led.turn_off()
        finally:
            close = getattr(led_backend, "close", None)
            if callable(close):
                close()

    _json_event("station_stopped", {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
