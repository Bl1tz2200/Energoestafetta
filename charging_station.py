#!/usr/bin/env python3
"""Автономный контроллер зарядной станции «Энергоэстафеты».

Файл предназначен для Raspberry Pi станции и совместим с интерфейсами
репозитория Bl1tz2200/Energoestafetta:

* ``energy_relay_vision.StationVision`` распознаёт цвет ленты БВС;
* ``station_protocol.Signal.LANDING_GRANTED`` задаёт сигнал разрешения;
* ``led_interface.set_led(pattern, color)`` управляет лентой станции;
* ``station_link.StationChannel`` — Wi-Fi-канал до бортов (см. ниже).

Алгоритм:

1. Камера ищет заданный экспертами цвет ленты БВС.
2. Цвет должен быть подтверждён несколько кадров подряд.
3. Станция включает зелёную ленту — приглашение на посадку/зарядку.
4. После заданного времени лента выключается.
5. Перед новым приглашением станция ждёт исчезновения БВС из кадра.

**Два канала до борта.** Цвет ленты — основной и обязательный: по регламенту
станция должна опознавать нужный БВС именно по нему (шаг 7 алгоритма, 5
баллов). Wi-Fi-канал (``lib/station_link.py``, флаги ``--drone ИМЯ=IP``) —
надстройка над ним: борт явно просит посадку и докладывает «сел»/«улетел»,
поэтому приглашение снимается по факту посадки, а не по таймеру, и станция
знает, какой именно борт под ней. Цвет при этом остаётся решающим: запрос по
сети от борта, чей назначенный цвет не совпал с увиденным, получает отказ.
Без ``--drone`` станция работает ровно как раньше — по одному цвету и
таймеру ``--grant-hold``, то есть отказ роутера не срывает попытку.

Файл следует положить в корень Energoestafetta рядом с ``lib/`` либо передать
путь к репозиторию через ``--repo-root``. GitHub/ROS станции не требуются.

Проверка без камеры и Raspberry Pi:

    python3 charging_station.py --repo-root . --self-test

Полевой запуск сначала с безопасным выводом команд в консоль:

    python3 charging_station.py --repo-root . --camera 0 \
        --drone-colors red --led-backend console --once

Запуск с лентой WS281x на GPIO18 и связью с двумя бортами:

    sudo python3 charging_station.py --repo-root . --camera 0 \
        --drone uav1=192.168.0.11 --drone-color uav1=red \
        --drone uav2=192.168.0.12 --drone-color uav2=blue \
        --led-backend ws281x --led-pin 18 --led-count 12 --brightness 128
"""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass, replace as dataclasses_replace
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
class GrantDecision:
    """Итог сверки увиденного цвета с тем, кто просит посадку по сети."""

    granted: bool
    drone: Optional[str] = None  # кому адресовать; None — приглашение только лентой
    reason: str = ""  # заполняется только при отказе


@dataclass(frozen=True)
class StationConfig:
    """Параметры фильтрации и конечного автомата станции."""

    expected_colors: Tuple[str, ...] = ("red",)
    # Имя борта -> цвет его ленты. Кортеж пар, а не словарь: конфиг frozen.
    drone_colors: Tuple[Tuple[str, str], ...] = ()
    required_frames: int = 5
    reset_frames: int = 10
    min_confidence: float = 0.45
    min_frame_ratio: float = 0.003
    max_frame_ratio: float = 0.35
    poll_interval_s: float = 0.05
    grant_hold_s: float = 15.0
    require_request: bool = False
    landed_timeout_s: float = 60.0
    departed_timeout_s: float = 180.0
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


class StationRadio:
    """Сторона станции в Wi-Fi-протоколе ``lib/station_link.py``.

    Тонкая обёртка над ``StationChannel``: ``run_station`` работает только с
    этими семью методами, поэтому в самотесте её место занимает заглушка, а
    сеть не поднимается. Своих потоков нет — входящие разбираются в ``pump()``
    из кадрового цикла.
    """

    def __init__(
        self,
        link_module: Any,
        *,
        node_id: str,
        drones: Dict[str, str],
        port: int,
        bind_host: str = "0.0.0.0",
        request_ttl: float = 3.0,
        peer_port: Optional[int] = None,
    ) -> None:
        if not drones:
            raise ValueError("Нужен хотя бы один борт: --drone ИМЯ=IP")
        # peer_port нужен только чтобы поднять станцию и борт на одной машине
        # (--link-test); в поле порт с обеих сторон один и тот же.
        self._link = link_module.Link(
            node_id,
            link_module.UdpTransport(port, bind_host=bind_host, peer_port=peer_port),
            peers=dict(drones),
        )
        self._channel = link_module.StationChannel(
            self._link, request_ttl=request_ttl
        )

    def pump(self) -> None:
        self._channel.pump()

    def pending(self) -> Optional[str]:
        return self._channel.pending()

    def grant(self, drone: str) -> None:
        self._channel.grant(drone)

    def deny(self, drone: str, reason: str) -> None:
        self._channel.deny(drone, reason)

    def has_landed(self, drone: str) -> bool:
        return self._channel.has_landed(drone)

    def has_departed(self, drone: str) -> bool:
        return self._channel.has_departed(drone)

    def forget(self, drone: str) -> None:
        self._channel.forget(drone)

    def take_rejections(self) -> Tuple[str, ...]:
        """Забрать причины отказов: чужой IP, подменённое имя, битый пакет."""
        rejected = tuple(self._link.rejected)
        del self._link.rejected[:]
        return rejected

    def close(self) -> None:
        self._link.close()


def _load_repo_modules(repo_root: Path) -> Tuple[Any, Any, Any]:
    """Загрузить неизменённые модули Energoestafetta из ``repo_root/lib``."""
    lib_dir = repo_root.resolve() / "lib"
    required = (
        lib_dir / "led_interface.py",
        lib_dir / "station_protocol.py",
        lib_dir / "station_link.py",
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
    link = importlib.import_module("station_link")
    return led, protocol, link


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


def decide_grant(
    detection: DroneDetection,
    radio: Optional[Any],
    config: StationConfig,
) -> GrantDecision:
    """Решить, кого приглашать, по увиденному цвету и запросам из эфира.

    Цвет уже подтверждён кадрами — здесь решается только «кто это». Сеть не
    может выдать приглашение в обход цвета: борт с чужим цветом получает
    отказ, даже если исправно шлёт запрос со знакомого IP.
    """
    if radio is None:
        return GrantDecision(True)  # сети нет — как раньше, только по цвету

    assigned = dict(config.drone_colors)
    pending = radio.pending()
    if pending is not None:
        expected = assigned.get(pending)
        if expected is not None and expected != detection.color:
            return GrantDecision(
                False,
                pending,
                "борт {} должен показывать {}, а в кадре {}".format(
                    pending, expected, detection.color
                ),
            )
        return GrantDecision(True, pending)

    if config.require_request:
        return GrantDecision(
            False, None, "цвет подтверждён, но запроса на посадку по сети нет"
        )

    # Запроса нет (борт ещё не поднял связь или Wi-Fi лёг) — приглашаем по
    # цвету. Адресат известен, только если этот цвет назначен ровно одному борту.
    matched = sorted(
        name for name, color in config.drone_colors if color == detection.color
    )
    return GrantDecision(True, matched[0] if len(matched) == 1 else None)


def _wait_for_state(
    condition: Callable[[], bool],
    *,
    drone: str,
    radio: Any,
    denied: set,
    timeout_s: float,
    config: StationConfig,
    should_stop: Callable[[], bool],
    sleep_fn: Callable[[float], None],
    time_fn: Callable[[], float],
) -> bool:
    """Крутить приём, пока не выполнится ``condition`` или не выйдет время.

    Пока станция занята одним бортом, второй получает ``landing_denied`` —
    по одному разу на борт, чтобы отказ не уходил каждым опросом.
    """
    deadline = time_fn() + timeout_s
    step = config.poll_interval_s if config.poll_interval_s > 0 else 0.05
    while not should_stop():
        radio.pump()
        if condition():
            return True
        other = radio.pending()
        if other is not None and other != drone and other not in denied:
            radio.deny(other, "станция занята бортом " + drone)
            denied.add(other)
        if time_fn() >= deadline:
            return False
        sleep_fn(step)
    return False


def _serve_landing(
    drone: str,
    *,
    radio: Any,
    config: StationConfig,
    turn_off: Callable[[], None],
    event_writer: Callable[[str, Dict[str, Any]], None],
    should_stop: Callable[[], bool],
    sleep_fn: Callable[[float], None],
    time_fn: Callable[[], float],
) -> None:
    """Довести приглашение до конца по докладам борта, а не по таймеру.

    Зелёная лента горит, пока борт не доложил «сел», затем гаснет, и станция
    остаётся занятой до доклада «улетел». Оба ожидания ограничены таймаутом:
    упавший или севший с пульта борт не должен занять станцию навсегда.
    """
    denied: set = set()
    granted_at = time_fn()
    landed = _wait_for_state(
        lambda: radio.has_landed(drone),
        drone=drone,
        radio=radio,
        denied=denied,
        timeout_s=config.landed_timeout_s,
        config=config,
        should_stop=should_stop,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )
    turn_off()
    if not landed:
        event_writer(
            "landing_timeout",
            {"drone": drone, "timeout_s": config.landed_timeout_s},
        )
        radio.forget(drone)
        return

    landed_at = time_fn()
    event_writer(
        "drone_landed",
        {"drone": drone, "waited_s": round(landed_at - granted_at, 2)},
    )
    departed = _wait_for_state(
        lambda: radio.has_departed(drone),
        drone=drone,
        radio=radio,
        denied=denied,
        timeout_s=config.departed_timeout_s,
        config=config,
        should_stop=should_stop,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )
    if departed:
        event_writer(
            "drone_departed",
            {"drone": drone, "occupied_s": round(time_fn() - landed_at, 2)},
        )
    else:
        event_writer(
            "departure_timeout",
            {"drone": drone, "timeout_s": config.departed_timeout_s},
        )
    radio.forget(drone)


def _report_link(
    radio: Any,
    announced: set,
    event_writer: Callable[[str, Dict[str, Any]], None],
) -> None:
    """Показать в логе новый запрос на посадку и отбракованные датаграммы."""
    pending = radio.pending()
    if pending is None:
        announced.clear()
    elif pending not in announced:
        announced.add(pending)
        event_writer("landing_request", {"drone": pending})
    for reason in radio.take_rejections():
        event_writer("link_rejected", {"reason": reason})


def _validate_config(config: StationConfig) -> None:
    if not config.expected_colors:
        raise ValueError("Нужно указать хотя бы один ожидаемый цвет БВС")
    unknown = sorted(set(config.expected_colors) - set(DRONE_COLOR_RANGES))
    if unknown:
        raise ValueError("Неизвестные цвета БВС: " + ", ".join(unknown))
    assigned = dict(config.drone_colors)
    if len(assigned) != len(config.drone_colors):
        raise ValueError("Для одного борта задано два разных цвета")
    stray = sorted(set(assigned.values()) - set(config.expected_colors))
    if stray:
        raise ValueError(
            "Цвета бортов не входят в список ожидаемых: " + ", ".join(stray)
        )
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
    if config.landed_timeout_s <= 0 or config.departed_timeout_s <= 0:
        raise ValueError("Таймауты посадки и отлёта должны быть больше нуля")


def run_station(
    *,
    frame_reader: Callable[[], Any],
    detector: Any,
    grant_sender: Callable[[DroneDetection], None],
    turn_off: Callable[[], None],
    config: StationConfig,
    event_writer: Callable[[str, Dict[str, Any]], None],
    radio: Optional[Any] = None,
    should_stop: Callable[[], bool] = lambda: False,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
) -> int:
    """Запустить автономный цикл станции.

    Возвращает количество отправленных приглашений. Внешние зависимости
    переданы явно, поэтому конечный автомат проверяется без камеры/RPi/сети.
    ``radio`` — канал до бортов (``StationRadio``); ``None`` оставляет прежнее
    поведение «только цвет и таймер».
    """
    _validate_config(config)
    streak_color: Optional[str] = None
    streak_count = 0
    grants = 0
    announced: set = set()

    while not should_stop():
        if radio is not None:
            radio.pump()
            _report_link(radio, announced, event_writer)

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
            decision = decide_grant(detection, radio, config)
            if not decision.granted:
                event_writer(
                    "landing_refused",
                    {
                        "drone": decision.drone,
                        "drone_color": detection.color,
                        "reason": decision.reason,
                    },
                )
                if radio is not None and decision.drone is not None:
                    radio.deny(decision.drone, decision.reason)
                # Серия сбрасывается, иначе отказ уходил бы каждым кадром.
                streak_color = None
                streak_count = 0
                if config.poll_interval_s > 0:
                    sleep_fn(config.poll_interval_s)
                continue

            grant_sender(detection)
            if radio is not None and decision.drone is not None:
                radio.grant(decision.drone)
            grants += 1
            event_writer(
                "landing_granted",
                {
                    "drone": decision.drone,
                    "drone_color": detection.color,
                    "confidence": round(detection.confidence, 4),
                    "frame_ratio": round(detection.frame_ratio, 6),
                    "bbox": detection.bbox,
                },
            )

            # Борт на связи — ведём его по докладам; иначе снимаем приглашение
            # по таймеру, как без сети.
            by_radio = radio is not None and decision.drone is not None
            if by_radio:
                _serve_landing(
                    decision.drone,
                    radio=radio,
                    config=config,
                    turn_off=turn_off,
                    event_writer=event_writer,
                    should_stop=should_stop,
                    sleep_fn=sleep_fn,
                    time_fn=time_fn,
                )
            else:
                if config.grant_hold_s > 0:
                    sleep_fn(config.grant_hold_s)
                turn_off()

            if config.once:
                return grants

            if not by_radio:
                # Защита от повторного приглашения тому же БВС: новый цикл
                # разрешается только после нескольких кадров без его цвета.
                # При связи по сети эту роль играет доклад «улетел».
                absent_frames = 0
                while absent_frames < config.reset_frames and not should_stop():
                    frame = frame_reader()
                    current = select_drone(detector.detect(frame), config)
                    absent_frames = absent_frames + 1 if current is None else 0
                    if config.poll_interval_s > 0:
                        sleep_fn(config.poll_interval_s)

            streak_color = None
            streak_count = 0
            announced.clear()
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


def _parse_assignment(value: str) -> Tuple[str, str]:
    """``имя=значение`` -> пара. Формат общий для ``--drone`` и ``--drone-color``."""
    name, separator, assigned = value.partition("=")
    name, assigned = name.strip(), assigned.strip()
    if not separator or not name or not assigned:
        raise argparse.ArgumentTypeError(
            "Ожидается имя=значение, например uav1=192.168.0.11: " + value
        )
    return name, assigned


def _collect_pairs(
    values: Optional[Sequence[Tuple[str, str]]], what: str
) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for name, assigned in values or ():
        if name in pairs and pairs[name] != assigned:
            raise SystemExit(
                "Для борта {} задано два разных значения ({}): {} и {}".format(
                    name, what, pairs[name], assigned
                )
            )
        pairs[name] = assigned
    return pairs


@dataclass(frozen=True)
class _FakeStation:
    color: str
    confidence: float
    frame_ratio: float
    bbox: Tuple[int, int, int, int] = (10, 20, 30, 40)


@dataclass(frozen=True)
class _FakeResult:
    stations: Tuple[_FakeStation, ...]


class _FakeDetector:
    """Кадр в самотесте — это уже готовый результат детектора."""

    @staticmethod
    def detect(frame: _FakeResult) -> _FakeResult:
        return frame


class _FakeRadio:
    """Заглушка ``StationRadio``: сценарий бортов задаётся списком шагов.

    ``script`` — что происходит в эфире на каждом вызове ``pump()``:
    ``("request", "uav1")`` — борт просит посадку, ``("landed", "uav1")`` /
    ``("departed", "uav1")`` — доклады, ``None`` — тишина.
    """

    def __init__(self, script: Sequence[Optional[Tuple[str, str]]]) -> None:
        self._script = list(script)
        self._requested: Optional[str] = None
        self._landed: set = set()
        self._departed: set = set()
        self.grants: list = []
        self.denials: list = []

    def pump(self) -> None:
        if not self._script:
            return
        step = self._script.pop(0)
        if step is None:
            return
        kind, drone = step
        if kind == "request":
            self._requested = drone
        elif kind == "landed":
            self._landed.add(drone)
        elif kind == "departed":
            self._departed.add(drone)
            self._requested = None
        else:
            raise AssertionError("неизвестный шаг сценария: " + kind)

    def pending(self) -> Optional[str]:
        return self._requested

    def grant(self, drone: str) -> None:
        self.grants.append(drone)

    def deny(self, drone: str, reason: str) -> None:
        self.denials.append((drone, reason))

    def has_landed(self, drone: str) -> bool:
        return drone in self._landed

    def has_departed(self, drone: str) -> bool:
        return drone in self._departed

    def forget(self, drone: str) -> None:
        self._requested = None
        self._landed.discard(drone)
        self._departed.discard(drone)

    def take_rejections(self) -> Tuple[str, ...]:
        return ()


def _run_case(
    led: Any,
    protocol: Any,
    frames: Sequence[_FakeResult],
    config: StationConfig,
    radio: Optional[Any] = None,
) -> Tuple[int, list, list, list]:
    """Прогнать один сценарий станции на заглушках и вернуть его следы."""
    led_calls: list = []
    backend = led.CallbackBackend(
        12,
        effect_writer=lambda effect, color: led_calls.append((effect, color)),
        pixels_writer=lambda _colors: None,
    )
    led.configure_backend(backend)

    detections: list = []

    def grant_sender(detection: DroneDetection) -> None:
        detections.append(detection)
        protocol.send_signal(
            protocol.Signal.LANDING_GRANTED,
            set_led_fn=led.set_led,
        )

    events: list = []
    clock = [0.0]
    stream = list(frames)
    taken = [0]

    def frame_reader() -> _FakeResult:
        frame = stream[taken[0]]
        taken[0] += 1
        return frame

    grants = run_station(
        frame_reader=frame_reader,
        detector=_FakeDetector(),
        grant_sender=grant_sender,
        turn_off=led.turn_off,
        config=config,
        event_writer=lambda name, payload: events.append((name, payload)),
        radio=radio,
        # Кадры кончились — сценарий отработан; так цикл завершается сам, без
        # исключения из иссякшего итератора.
        should_stop=lambda: taken[0] >= len(stream),
        sleep_fn=lambda duration: clock.__setitem__(0, clock[0] + duration),
        time_fn=lambda: clock[0],
    )
    return grants, detections, events, led_calls


def _self_test(led: Any, protocol: Any, link: Any) -> int:
    """Проверить фильтрацию кадров, связь с бортом и протокол репозитория."""

    red = _FakeResult((_FakeStation("red", 0.9, 0.02),))
    weak_red = _FakeResult((_FakeStation("red", 0.2, 0.02),))
    empty = _FakeResult(tuple())
    base = StationConfig(
        expected_colors=("red",),
        required_frames=3,
        reset_frames=2,
        min_confidence=0.45,
        grant_hold_s=0.0,
        poll_interval_s=0.0,
        once=True,
    )

    # 1. Без сети: цвет подтверждён тремя кадрами подряд, разрыв серии сбрасывает
    # счётчик, приглашение — зелёная лента из station_protocol.
    grants, detections, events, led_calls = _run_case(
        led,
        protocol,
        [weak_red, red, empty, red, red, red],
        base,
    )
    assert grants == 1
    assert len(detections) == 1 and detections[0].color == "red"
    assert events[0][0] == "landing_granted"
    assert events[0][1]["drone"] is None  # адресата нет: сеть не настроена
    assert led_calls == [
        ("fill", (0, 255, 0)),  # LANDING_GRANTED из station_protocol
        ("fill", (0, 0, 0)),  # turn_off
    ]

    # 2. Со связью: запрос борта -> приглашение ему адресно -> лента гаснет по
    # докладу «сел», станция освобождается по докладу «улетел», а не по таймеру.
    linked = dataclasses_replace(
        base,
        drone_colors=(("uav1", "red"),),
        once=False,
        grant_hold_s=99.0,  # не должен использоваться: борт на связи
    )
    radio = _FakeRadio([("request", "uav1"), None, None, ("landed", "uav1"),
                        ("departed", "uav1")])
    grants, _, events, led_calls = _run_case(
        led, protocol, [red, red, red, empty], linked, radio
    )
    assert grants == 1
    assert radio.grants == ["uav1"], radio.grants
    names = [name for name, _payload in events]
    assert names == [
        "landing_request",
        "landing_granted",
        "drone_landed",
        "drone_departed",
        "station_ready",
    ], names
    assert led_calls == [("fill", (0, 255, 0)), ("fill", (0, 0, 0))]

    # 3. Цвет решает: борт со знакомого IP, но с чужим цветом получает отказ,
    # а не приглашение (шаг 7 регламента — опознание именно по ленте).
    wrong = dataclasses_replace(
        base, expected_colors=("red", "blue"),
        drone_colors=(("uav1", "blue"),), once=False,
    )
    radio = _FakeRadio([("request", "uav1")])
    grants, _, events, led_calls = _run_case(
        led, protocol, [red, red, red, empty], wrong, radio
    )
    assert grants == 0, "чужой цвет не должен приглашаться"
    assert radio.grants == []
    assert radio.denials and radio.denials[0][0] == "uav1"
    assert [name for name, _payload in events][-1] == "landing_refused"
    assert led_calls == [], "лента не должна загораться при отказе"

    # 4. Занятая станция отказывает второму борту — по одному разу, а не каждым
    # опросом эфира.
    busy = dataclasses_replace(
        base, drone_colors=(("uav1", "red"), ("uav2", "red")), once=False,
    )
    radio = _FakeRadio(
        [("request", "uav1"), None, None, ("request", "uav2"), None, None,
         ("landed", "uav1"), ("departed", "uav1")]
    )
    grants, _, events, _ = _run_case(
        led, protocol, [red, red, red, empty], busy, radio
    )
    assert grants == 1 and radio.grants == ["uav1"]
    assert radio.denials == [("uav2", "станция занята бортом uav1")], radio.denials

    # 5. Замолчавший борт не занимает станцию навсегда: оба ожидания выходят
    # по таймауту, лента гаснет.
    silent = dataclasses_replace(
        base, drone_colors=(("uav1", "red"),), once=False,
        poll_interval_s=0.1, landed_timeout_s=1.0,
    )
    radio = _FakeRadio([("request", "uav1")])
    grants, _, events, led_calls = _run_case(
        led, protocol, [red, red, red, empty], silent, radio
    )
    assert grants == 1
    assert [name for name, _payload in events][2] == "landing_timeout"
    assert led_calls[-1] == ("fill", (0, 0, 0))

    # 6. --require-request: без запроса по сети одного цвета мало.
    strict = dataclasses_replace(
        base, drone_colors=(("uav1", "red"),), once=False, require_request=True,
    )
    radio = _FakeRadio([])
    grants, _, events, _ = _run_case(
        led, protocol, [red, red, red, empty], strict, radio
    )
    assert grants == 0 and radio.grants == []
    assert events[-1][0] == "landing_refused"

    # 7. Решение о приглашении отделено от цикла и проверяется напрямую.
    detection = DroneDetection("red", 0.9, 0.02, (0, 0, 10, 10))
    assert decide_grant(detection, None, base) == GrantDecision(True)
    lone = dataclasses_replace(base, drone_colors=(("uav1", "red"),))
    # Связь ещё не поднята, но цвет назначен ровно одному борту — адресат ясен.
    assert decide_grant(detection, _FakeRadio([]), lone).drone == "uav1"

    # 8. Сигналы протокола станции совпадают с таблицей station_protocol.
    assert protocol.SIGNAL_COLORS[protocol.Signal.LANDING_GRANTED] == "green"
    assert link.Kind.LANDING_GRANTED.value == "landing_granted"
    assert link.DEFAULT_PORT > 0

    # 9. Неверный конфиг отклоняется до выхода на площадку.
    for broken in (
        StationConfig(expected_colors=("purple",)),
        StationConfig(expected_colors=("red",), drone_colors=(("uav1", "blue"),)),
        StationConfig(expected_colors=("red",), landed_timeout_s=0.0),
    ):
        try:
            _validate_config(broken)
        except ValueError:
            continue
        raise AssertionError("Неверный конфиг должен отклоняться: {}".format(broken))

    print("SELF-TEST: OK")
    return 0


def _link_test(led: Any, protocol: Any, link: Any, port: int) -> int:
    """Прогнать полный цикл станции против живого борта на 127.0.0.1.

    Самотест проверяет автомат на заглушке радио, а здесь между станцией и
    бортом настоящие UDP-сокеты, настоящий ``StationRadio`` и настоящий
    ``station_link.DroneChannel`` — то есть ровно та связка, которая поедет на
    площадку. Камера и лента не нужны: кадры подставные, лента консольная.
    """
    host = "127.0.0.1"
    radio = StationRadio(
        link,
        node_id="station1",
        drones={"uav1": host},
        port=port,
        bind_host=host,
        peer_port=port + 1,
    )
    drone_link = link.Link(
        "uav1",
        link.UdpTransport(port + 1, bind_host=host, peer_port=port),
        peers={"station1": host},
    )
    drone = link.DroneChannel(drone_link, station="station1")

    led.configure_backend(led.ConsoleBackend(led_count=12, verbose=False))
    red = _FakeResult((_FakeStation("red", 0.9, 0.02),))
    config = StationConfig(
        expected_colors=("red",),
        drone_colors=(("uav1", "red"),),
        required_frames=2,
        poll_interval_s=0.02,
        landed_timeout_s=5.0,
        departed_timeout_s=5.0,
        once=True,
    )
    events: list = []
    frames = [0]

    def frame_reader() -> _FakeResult:
        # Борт «летит» по своему расписанию в соседнем потоке; станция просто
        # видит красную ленту в каждом кадре.
        frames[0] += 1
        return red

    def fly() -> None:
        answer = drone.request_landing(timeout=8.0, interval=0.2)
        if answer is None or answer.kind != link.Kind.LANDING_GRANTED:
            return
        drone.report_landed()
        time.sleep(0.3)  # «зарядка»
        drone.report_departed()

    pilot = threading.Thread(target=fly, daemon=True)
    pilot.start()
    try:
        grants = run_station(
            frame_reader=frame_reader,
            detector=_FakeDetector(),
            grant_sender=lambda detection: protocol.send_signal(
                protocol.Signal.LANDING_GRANTED, set_led_fn=led.set_led
            ),
            turn_off=led.turn_off,
            config=config,
            event_writer=lambda name, payload: events.append((name, payload)),
            radio=radio,
            should_stop=lambda: frames[0] > 2000,
        )
    finally:
        pilot.join(timeout=2.0)
        radio.close()
        drone_link.close()

    names = [name for name, _payload in events]
    expected = ["landing_request", "landing_granted", "drone_landed", "drone_departed"]
    if grants != 1 or names != expected:
        print("ОШИБКА: ожидалось {}, получено {} (приглашений {})".format(
            expected, names, grants))
        return 1
    granted_to = events[1][1]["drone"]
    if granted_to != "uav1":
        print("ОШИБКА: приглашение адресовано {}, а не uav1".format(granted_to))
        return 1
    print("LINK-TEST: OK (станция <-> борт по UDP {} <-> {})".format(port, port + 1))
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
    parser.add_argument(
        "--drone",
        action="append",
        type=_parse_assignment,
        metavar="ИМЯ=IP",
        default=None,
        help="борт и его адрес в Wi-Fi для связи по station_link "
             "(флаг повторяемый; без него станция работает только по цвету)",
    )
    parser.add_argument(
        "--drone-color",
        action="append",
        type=_parse_assignment,
        metavar="ИМЯ=ЦВЕТ",
        default=None,
        help="цвет ленты, назначенный борту: по нему станция его и опознаёт",
    )
    parser.add_argument(
        "--station-id",
        default="station1",
        help="имя этой станции в протоколе связи",
    )
    parser.add_argument(
        "--link-port",
        type=int,
        default=5801,
        help="UDP-порт связи, одинаковый на станции и на бортах",
    )
    parser.add_argument("--link-bind", default="0.0.0.0",
                        help="локальный адрес приёма датаграмм")
    parser.add_argument(
        "--request-ttl",
        type=float,
        default=3.0,
        help="через сколько секунд молчания запрос борта считается протухшим",
    )
    parser.add_argument(
        "--require-request",
        action="store_true",
        help="приглашать только при запросе по сети (по умолчанию хватает цвета)",
    )
    parser.add_argument(
        "--landed-timeout",
        type=float,
        default=60.0,
        help="сколько ждать доклада «сел» после приглашения, с",
    )
    parser.add_argument(
        "--departed-timeout",
        type=float,
        default=180.0,
        help="сколько ждать доклада «улетел», прежде чем освободить станцию, с",
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
    parser.add_argument(
        "--link-test",
        action="store_true",
        help="прогнать цикл станции против живого борта на 127.0.0.1 (без камеры)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    led, protocol, link = _load_repo_modules(args.repo_root)

    if args.self_test:
        return _self_test(led, protocol, link)
    if args.link_test:
        return _link_test(led, protocol, link, args.link_port)

    drones = _collect_pairs(args.drone, "адрес")
    assigned_colors = _collect_pairs(args.drone_color, "цвет")
    unknown = sorted(set(assigned_colors.values()) - set(DRONE_COLOR_RANGES))
    if unknown:
        raise SystemExit("Неизвестные цвета бортов: " + ", ".join(unknown))
    # Цвет без адреса — это молча не работающая связь: борт опознан не будет.
    strangers = sorted(set(assigned_colors) - set(drones))
    if strangers:
        raise SystemExit(
            "Цвет задан бортам без адреса (--drone ИМЯ=IP): " + ", ".join(strangers)
        )
    if args.require_request and not drones:
        raise SystemExit("--require-request без --drone ИМЯ=IP невозможен")

    # Назначенный борту цвет обязан попасть в фильтр детектора, иначе станция
    # его просто не увидит; поэтому список ожидаемых цветов расширяется сам.
    expected_colors = tuple(
        dict.fromkeys(tuple(args.drone_colors) + tuple(assigned_colors.values()))
    )

    config = StationConfig(
        expected_colors=expected_colors,
        drone_colors=tuple(sorted(assigned_colors.items())),
        required_frames=args.required_frames,
        reset_frames=args.reset_frames,
        min_confidence=args.min_confidence,
        min_frame_ratio=args.min_area_ratio,
        max_frame_ratio=args.max_area_ratio,
        poll_interval_s=args.poll_interval,
        grant_hold_s=args.grant_hold,
        require_request=args.require_request,
        landed_timeout_s=args.landed_timeout,
        departed_timeout_s=args.departed_timeout,
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

    radio: Optional[StationRadio] = None
    if drones:
        radio = StationRadio(
            link,
            node_id=args.station_id,
            drones=drones,
            port=args.link_port,
            bind_host=args.link_bind,
            request_ttl=args.request_ttl,
        )
        _json_event(
            "link_started",
            {
                "station_id": args.station_id,
                "port": args.link_port,
                "drones": drones,
                "drone_colors": dict(config.drone_colors),
                "require_request": config.require_request,
            },
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
            radio=radio,
            should_stop=stop_event.is_set,
        )
    finally:
        camera.close()
        if radio is not None:
            radio.close()
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
