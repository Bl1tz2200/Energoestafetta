#!/usr/bin/env python3
"""Единый интерфейс управления светодиодной лентой для «Энергоэстафеты».

Зафиксированная сигнатура из шага 0 PLAN.md:

    set_led(pattern, color)

Допустимые вызовы:

    set_led("solid", "red")
    set_led("blink", "yellow")
    set_led("rainbow", None)
    set_led("half-half", ("red", "blue"))

Паттерны:

* ``solid`` — вся лента горит одним цветом;
* ``blink`` — вся лента мигает одним цветом;
* ``rainbow`` — эффект радуги, цвет не передаётся;
* ``half-half`` — первая и вторая половины ленты имеют разные цвета.

По умолчанию установлен безопасный ConsoleBackend: команды проверяются и
печатаются, но физическая лента не меняется. Дрон соревнования — Skyris
Technic 6S (PX4 + Orange Pi 5 Pro), его штатный API — это ROS 1 (rospy),
пакеты ``technic`` и ``led_msgs`` (см. docs.skyris.ru/technic6S/LedStrip.html):

    rospy.init_node('flight')  # один раз в полётном скрипте
    use_technic_ros1_backend()

Для другого API используйте ``CallbackBackend``.
"""

from __future__ import annotations

import argparse
import re
import threading
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)


RGB = Tuple[int, int, int]
ColorValue = Union[str, RGB]
HalfColors = Tuple[ColorValue, ColorValue]
PatternColor = Union[ColorValue, HalfColors, None]


COLORS: Dict[str, RGB] = {
    "off": (0, 0, 0),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "white": (255, 255, 255),
}

PATTERNS = frozenset({"solid", "blink", "rainbow", "half-half"})
_HEX_COLOR = re.compile(r"^#?([0-9a-fA-F]{6})$")


@dataclass(frozen=True)
class LEDCommand:
    """Нормализованная команда, которую получил backend."""

    pattern: str
    colors: Tuple[RGB, ...]


class LEDBackend(Protocol):
    """Минимальный контракт аппаратного или тестового драйвера."""

    @property
    def led_count(self) -> int:
        ...

    def set_effect(self, effect: str, color: RGB) -> None:
        ...

    def set_pixels(self, colors: Sequence[RGB]) -> None:
        ...


class ConsoleBackend:
    """Безопасный backend для разработки без ROS и физической ленты."""

    def __init__(self, led_count: int = 12, verbose: bool = True) -> None:
        if led_count < 2:
            raise ValueError("В ленте должно быть минимум два светодиода")
        self._led_count = int(led_count)
        self.verbose = bool(verbose)
        self.last_effect: Optional[Tuple[str, RGB]] = None
        self.last_pixels: Optional[Tuple[RGB, ...]] = None

    @property
    def led_count(self) -> int:
        return self._led_count

    def set_effect(self, effect: str, color: RGB) -> None:
        self.last_effect = (effect, color)
        self.last_pixels = None
        if self.verbose:
            print("LED effect={} color={}".format(effect, color), flush=True)

    def set_pixels(self, colors: Sequence[RGB]) -> None:
        pixels = tuple(colors)
        if len(pixels) != self._led_count:
            raise ValueError(
                "Ожидалось {} цветов, получено {}".format(
                    self._led_count, len(pixels)
                )
            )
        self.last_pixels = pixels
        self.last_effect = None
        if self.verbose:
            split = _summarize_halves(pixels)
            print("LED pixels={} {}".format(len(pixels), split), flush=True)


class CallbackBackend:
    """Адаптер для любого API, которого ещё нет в этом файле.

    ``effect_writer`` получает эффект ``fill``, ``blink`` или ``rainbow`` и
    RGB-кортеж. ``pixels_writer`` получает список RGB для всей ленты.
    """

    def __init__(
        self,
        led_count: int,
        effect_writer: Callable[[str, RGB], None],
        pixels_writer: Callable[[Sequence[RGB]], None],
    ) -> None:
        if led_count < 2:
            raise ValueError("В ленте должно быть минимум два светодиода")
        if not callable(effect_writer) or not callable(pixels_writer):
            raise TypeError("Оба writer должны быть вызываемыми функциями")
        self._led_count = int(led_count)
        self._effect_writer = effect_writer
        self._pixels_writer = pixels_writer

    @property
    def led_count(self) -> int:
        return self._led_count

    def set_effect(self, effect: str, color: RGB) -> None:
        self._effect_writer(effect, color)

    def set_pixels(self, colors: Sequence[RGB]) -> None:
        self._pixels_writer(tuple(colors))


class TechnicROS1Backend:
    """Backend ROS 1 (rospy) для штатного LED API Skyris Technic 6S.

    Реальные сервисы платформы (docs.skyris.ru/technic6S/LedStrip.html):

    * ``led/set_effect`` (``technic.srv.SetLEDEffect``) — параметры
      ``r, g, b`` и опциональный ``effect`` (``fill`` по умолчанию, также
      ``fade``, ``flash``, ``blink``, ``rainbow``);
    * ``led/set_leds`` (``led_msgs.srv.SetLEDs``) — список
      ``led_msgs.msg.LEDState(index, r, g, b)``;
    * ``led/state`` (``led_msgs.msg.LEDStateArray``) — текущее число диодов.

    Узел ROS (``rospy.init_node(...)``) должен быть поднят полётным
    сценарием заранее — backend его не создаёт и не закрывает.

    Прокси сервисов можно передать напрямую (``set_effect_proxy``,
    ``set_leds_proxy``, ``state_reader``) — тогда backend не импортирует
    ``rospy``, что удобно для тестов интерфейса без установленного ROS 1.
    """

    def __init__(
        self,
        led_count: Optional[int] = None,
        state_timeout: float = 5.0,
        set_effect_proxy: Optional[Callable[..., Any]] = None,
        set_leds_proxy: Optional[Callable[[Sequence[Any]], Any]] = None,
        state_reader: Optional[Callable[[float], int]] = None,
        led_state_factory: Optional[Callable[[int, int, int, int], Any]] = None,
    ) -> None:
        if state_timeout <= 0:
            raise ValueError("state_timeout должен быть больше нуля")
        if led_count is not None and led_count < 2:
            raise ValueError("В ленте должно быть минимум два светодиода")

        need_ros = (
            set_effect_proxy is None
            or set_leds_proxy is None
            or (led_count is None and state_reader is None)
        )
        if need_ros:
            try:
                import rospy
                from technic.srv import SetLEDEffect
                from led_msgs.srv import SetLEDs
                from led_msgs.msg import LEDState, LEDStateArray
            except ImportError as exc:
                raise RuntimeError(
                    "Нужны rospy, technic и led_msgs. Выполните source "
                    "/opt/ros/noetic/setup.bash и source своего catkin_ws"
                ) from exc
            if set_effect_proxy is None:
                set_effect_proxy = rospy.ServiceProxy(
                    "led/set_effect", SetLEDEffect
                )
            if set_leds_proxy is None:
                set_leds_proxy = rospy.ServiceProxy("led/set_leds", SetLEDs)
            if led_state_factory is None:
                led_state_factory = LEDState
            if state_reader is None:
                def state_reader(timeout: float) -> int:
                    message = rospy.wait_for_message(
                        "led/state", LEDStateArray, timeout=timeout
                    )
                    return len(message.leds)

        if led_state_factory is None:
            led_state_factory = lambda index, r, g, b: (index, r, g, b)

        self._set_effect_proxy = set_effect_proxy
        self._set_leds_proxy = set_leds_proxy
        self._state_reader = state_reader
        self._led_state_factory = led_state_factory
        self._state_timeout = float(state_timeout)
        # Число диодов запрашивается лениво: solid/blink/rainbow не должны
        # зависеть от доступности топика состояния. Оно нужно только half-half.
        self._led_count = int(led_count) if led_count is not None else None

    @property
    def led_count(self) -> int:
        if self._led_count is None:
            count = self._state_reader(self._state_timeout)
            if count < 2:
                raise RuntimeError(
                    "Некорректное число светодиодов: {}".format(count)
                )
            self._led_count = count
        return self._led_count

    def set_effect(self, effect: str, color: RGB) -> None:
        if effect == "rainbow":
            self._set_effect_proxy(effect="rainbow")
            return
        red, green, blue = color
        kwargs: Dict[str, Any] = {"r": red, "g": green, "b": blue}
        if effect != "fill":
            kwargs["effect"] = effect
        self._set_effect_proxy(**kwargs)

    def set_pixels(self, colors: Sequence[RGB]) -> None:
        count = self.led_count
        if len(colors) != count:
            raise ValueError(
                "Ожидалось {} цветов, получено {}".format(
                    count, len(colors)
                )
            )
        states = [
            self._led_state_factory(index, red, green, blue)
            for index, (red, green, blue) in enumerate(colors)
        ]
        self._set_leds_proxy(states)


class LEDController:
    """Проверяет публичный интерфейс и переводит его в команды backend."""

    def __init__(self, backend: LEDBackend) -> None:
        self._backend = backend
        self._lock = threading.RLock()
        self._last_command: Optional[LEDCommand] = None

    @property
    def backend(self) -> LEDBackend:
        return self._backend

    @property
    def last_command(self) -> Optional[LEDCommand]:
        with self._lock:
            return self._last_command

    def set_led(self, pattern: str, color: PatternColor) -> LEDCommand:
        normalized_pattern = _normalize_pattern(pattern)
        normalized_colors = _normalize_pattern_color(
            normalized_pattern, color
        )
        command = LEDCommand(normalized_pattern, normalized_colors)

        with self._lock:
            if normalized_pattern == "solid":
                self._backend.set_effect("fill", normalized_colors[0])
            elif normalized_pattern == "blink":
                self._backend.set_effect("blink", normalized_colors[0])
            elif normalized_pattern == "rainbow":
                self._backend.set_effect("rainbow", (0, 0, 0))
            elif normalized_pattern == "half-half":
                first, second = normalized_colors
                count = self._backend.led_count
                split = count // 2
                # При нечётном количестве центральный светодиод относится ко
                # второй половине. Разница между половинами будет ровно один.
                pixels = (first,) * split + (second,) * (count - split)
                self._backend.set_pixels(pixels)
            else:  # защита на случай изменения PATTERNS без реализации
                raise AssertionError(
                    "Паттерн не реализован: " + normalized_pattern
                )
            self._last_command = command
        return command


_controller = LEDController(ConsoleBackend())


def set_led(pattern: str, color: PatternColor) -> LEDCommand:
    """Установить индикацию ленты.

    Это единственная функция, которую должны вызывать полётные сценарии.

    Args:
        pattern: ``solid``, ``blink``, ``rainbow`` или ``half-half``.
        color:
            * имя цвета, ``#RRGGBB`` или RGB-кортеж для solid/blink;
            * ``None`` для rainbow;
            * пара цветов для half-half.

    Returns:
        Нормализованная команда; удобна для логирования и тестов.

    Raises:
        ValueError/TypeError: команда имеет неверный формат.
        RuntimeError: аппаратный backend не принял команду.
    """
    return _controller.set_led(pattern, color)


def configure_backend(backend: LEDBackend) -> LEDController:
    """Заменить backend и вернуть новый контроллер."""
    global _controller
    if backend is None:
        raise TypeError("backend не может быть None")
    _controller = LEDController(backend)
    return _controller


def use_technic_ros1_backend(
    led_count: Optional[int] = None,
    state_timeout: float = 5.0,
) -> LEDController:
    """Подключить физическую ленту через штатный ROS 1 API Skyris Technic 6S.

    Полётный скрипт должен поднять узел ROS до вызова::

        rospy.init_node('flight')
        use_technic_ros1_backend()
    """
    return configure_backend(
        TechnicROS1Backend(led_count=led_count, state_timeout=state_timeout)
    )


def current_command() -> Optional[LEDCommand]:
    """Последняя успешно применённая команда или None."""
    return _controller.last_command


def turn_off() -> LEDCommand:
    """Погасить всю ленту."""
    return set_led("solid", "off")


def close_led_backend() -> None:
    """Освободить ресурсы текущего backend, если он это поддерживает."""
    close = getattr(_controller.backend, "close", None)
    if callable(close):
        close()


def _normalize_pattern(pattern: str) -> str:
    if not isinstance(pattern, str):
        raise TypeError("pattern должен быть строкой")
    normalized = pattern.strip().lower().replace("_", "-")
    if normalized not in PATTERNS:
        raise ValueError(
            "Неизвестный pattern {!r}; допустимы: {}".format(
                pattern, ", ".join(sorted(PATTERNS))
            )
        )
    return normalized


def _normalize_pattern_color(
    pattern: str, color: PatternColor
) -> Tuple[RGB, ...]:
    if pattern == "rainbow":
        if color is not None:
            raise ValueError("Для rainbow второй аргумент должен быть None")
        return ()

    if pattern == "half-half":
        if not _is_pair_of_colors(color):
            raise ValueError(
                "Для half-half нужна пара цветов, например ('red', 'blue')"
            )
        assert color is not None
        return (_parse_color(color[0]), _parse_color(color[1]))

    if color is None:
        raise ValueError("Для {} необходимо указать цвет".format(pattern))
    if _is_pair_of_colors(color):
        raise ValueError(
            "Для {} нужен один цвет, а не пара".format(pattern)
        )
    return (_parse_color(color),)


def _is_pair_of_colors(value: object) -> bool:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return False
    # RGB-кортеж имеет длину 3, поэтому пара RGB/имён здесь однозначна.
    return all(_looks_like_color(item) for item in value)


def _looks_like_color(value: object) -> bool:
    if isinstance(value, str):
        return True
    return (
        isinstance(value, (tuple, list))
        and len(value) == 3
        and all(isinstance(channel, int) and not isinstance(channel, bool)
                for channel in value)
    )


def _parse_color(value: ColorValue) -> RGB:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in COLORS:
            return COLORS[normalized]
        match = _HEX_COLOR.fullmatch(normalized)
        if match:
            raw = match.group(1)
            return tuple(
                int(raw[index : index + 2], 16) for index in (0, 2, 4)
            )  # type: ignore[return-value]
        raise ValueError(
            "Неизвестный цвет {!r}; используйте имя, #RRGGBB или RGB".format(
                value
            )
        )

    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise TypeError("RGB-цвет должен содержать ровно три канала")
    channels = []
    for channel in value:
        if not isinstance(channel, int) or isinstance(channel, bool):
            raise TypeError("Каналы RGB должны быть целыми числами")
        if not 0 <= channel <= 255:
            raise ValueError("Каналы RGB должны лежать в диапазоне 0..255")
        channels.append(channel)
    return channels[0], channels[1], channels[2]


def _summarize_halves(colors: Iterable[RGB]) -> str:
    pixels = tuple(colors)
    if not pixels:
        return "empty"
    split = len(pixels) // 2
    return "first={} second={}".format(
        pixels[0], pixels[split]
    )


def _self_test() -> int:
    backend = ConsoleBackend(led_count=11, verbose=False)
    configure_backend(backend)

    assert set_led("solid", "red") == LEDCommand("solid", ((255, 0, 0),))
    assert backend.last_effect == ("fill", (255, 0, 0))

    assert set_led("blink", "#FFFF00") == LEDCommand(
        "blink", ((255, 255, 0),)
    )
    assert backend.last_effect == ("blink", (255, 255, 0))

    assert set_led("rainbow", None) == LEDCommand("rainbow", ())
    assert backend.last_effect == ("rainbow", (0, 0, 0))

    assert set_led("half_half", ("red", "blue")) == LEDCommand(
        "half-half", ((255, 0, 0), (0, 0, 255))
    )
    assert backend.last_pixels is not None
    assert backend.last_pixels[:5] == ((255, 0, 0),) * 5
    assert backend.last_pixels[5:] == ((0, 0, 255),) * 6

    assert turn_off() == LEDCommand("solid", ((0, 0, 0),))
    assert current_command() == LEDCommand("solid", ((0, 0, 0),))

    # Проверяем TechnicROS1Backend с подставными rospy-прокси, без установки
    # ROS 1: set_effect_proxy/set_leds_proxy/state_reader эмулируют реальные
    # сервисы led/set_effect, led/set_leds и топик led/state.
    effect_calls: list = []
    pixel_calls: list = []
    state_calls = [0]

    def fake_set_effect(**kwargs: object) -> None:
        effect_calls.append(kwargs)

    def fake_set_leds(states: list) -> None:
        pixel_calls.append(states)

    def fake_state_reader(timeout: float) -> int:
        state_calls[0] += 1
        return 10

    technic_backend = TechnicROS1Backend(
        state_timeout=2.0,
        set_effect_proxy=fake_set_effect,
        set_leds_proxy=fake_set_leds,
        state_reader=fake_state_reader,
    )
    configure_backend(technic_backend)

    set_led("solid", "green")
    assert effect_calls[-1] == {"r": 0, "g": 255, "b": 0}

    set_led("blink", "yellow")
    assert effect_calls[-1] == {
        "r": 255,
        "g": 255,
        "b": 0,
        "effect": "blink",
    }

    set_led("rainbow", None)
    assert effect_calls[-1] == {"effect": "rainbow"}

    set_led("half-half", ("red", "blue"))
    sent_leds = pixel_calls[-1]
    assert sent_leds[:5] == [(index, 255, 0, 0) for index in range(5)]
    assert sent_leds[5:] == [(index, 0, 0, 255) for index in range(5, 10)]
    assert state_calls[0] == 1

    invalid_commands = (
        ("rainbow", "red"),
        ("half-half", "red"),
        ("solid", ("red", "blue")),
        ("blink", None),
        ("unknown", "red"),
    )
    for pattern, color in invalid_commands:
        try:
            set_led(pattern, color)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                "Некорректная команда была принята: {!r}, {!r}".format(
                    pattern, color
                )
            )

    print("SELF-TEST: OK")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Единый интерфейс управления LED-лентой"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="проверить все паттерны без физической ленты",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="вывести примеры команд через безопасный ConsoleBackend",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.self_test:
        return _self_test()
    if args.demo:
        configure_backend(ConsoleBackend(led_count=12, verbose=True))
        set_led("solid", "red")
        set_led("blink", "yellow")
        set_led("rainbow", None)
        set_led("half-half", ("red", "blue"))
        turn_off()
        return 0
    _build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
