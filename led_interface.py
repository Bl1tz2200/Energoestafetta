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
печатаются, но физическая лента не меняется. Для ROS 2 на платформе Sverk
передайте уже созданный объект drone в ``use_sverk_ros2_backend(drone)``.
Для автономной проверки ленты backend может создать ROS-узел сам:
``use_sverk_ros2_backend()``. Для другого API используйте ``CallbackBackend``.
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


class SverkROS2Backend:
    """Backend ROS 2 для официального Python API ``sverk_interfaces``.

    Если ``drone`` уже создан полётным сценарием через
    ``sverk_interfaces.init(...)``, передайте его сюда. Если ``drone=None``,
    backend создаст собственный ROS-узел и будет отвечать за его закрытие.

    Импорт ``sverk_interfaces`` выполняется только при создании собственного
    узла, поэтому тесты интерфейса работают и на компьютере без ROS 2.
    """

    def __init__(
        self,
        drone: Optional[Any] = None,
        led_count: Optional[int] = None,
        state_timeout: float = 5.0,
        node_name: str = "energy_relay_led",
    ) -> None:
        if state_timeout <= 0:
            raise ValueError("state_timeout должен быть больше нуля")
        if led_count is not None and led_count < 2:
            raise ValueError("В ленте должно быть минимум два светодиода")

        self._owns_drone = drone is None
        if drone is None:
            try:
                import sverk_interfaces
            except ImportError as exc:
                raise RuntimeError(
                    "sverk_interfaces недоступен. Выполните source "
                    "/opt/ros/*/setup.bash и source ~/sverk_ws/install/setup.bash"
                ) from exc
            drone = sverk_interfaces.init(Nodename=node_name)

        self._drone = drone
        self._led = getattr(drone, "led", None)
        if self._led is None:
            if self._owns_drone:
                self.close()
            raise RuntimeError(
                "LED API недоступен: drone.led is None. Проверьте пакеты "
                "led_interfaces и узел /led_control."
            )

        self._state_timeout = float(state_timeout)
        # Число диодов запрашивается лениво: solid/blink/rainbow не должны
        # зависеть от доступности топика состояния. Оно нужно только half-half.
        self._led_count = int(led_count) if led_count is not None else None

    @property
    def led_count(self) -> int:
        if self._led_count is None:
            state = self._led.get_state(timeout=self._state_timeout)
            leds = getattr(state, "leds", None) if state is not None else None
            if leds is None:
                raise RuntimeError(
                    "Не получено состояние /led_control/state за {:.1f} с. "
                    "Передайте led_count явно или проверьте LED-узел.".format(
                        self._state_timeout
                    )
                )
            count = len(leds)
            if count < 2:
                raise RuntimeError(
                    "Некорректное число светодиодов: {}".format(count)
                )
            self._led_count = count
        return self._led_count

    def set_effect(self, effect: str, color: RGB) -> None:
        red, green, blue = color
        result = self._led.set_effect(
            effect,
            r=red,
            g=green,
            b=blue,
            timeout=self._state_timeout,
        )
        _ensure_sverk_success(result, "/led_control/set_effect")

    def set_pixels(self, colors: Sequence[RGB]) -> None:
        count = self.led_count
        if len(colors) != count:
            raise ValueError(
                "Ожидалось {} цветов, получено {}".format(
                    count, len(colors)
                )
            )
        states = [
            (index, red, green, blue)
            for index, (red, green, blue) in enumerate(colors)
        ]
        result = self._led.set_leds(
            states,
            timeout=self._state_timeout,
        )
        _ensure_sverk_success(result, "/led_control/set_leds")

    def close(self) -> None:
        """Закрыть только тот drone/ROS-узел, который backend создал сам."""
        if self._owns_drone and self._drone is not None:
            close = getattr(self._drone, "close", None)
            if callable(close):
                close()
            self._drone = None


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


def use_sverk_ros2_backend(
    drone: Optional[Any] = None,
    led_count: Optional[int] = None,
    state_timeout: float = 5.0,
    node_name: str = "energy_relay_led",
) -> LEDController:
    """Подключить физическую ленту через ROS 2 ``sverk_interfaces``.

    В полётном сценарии предпочтительно передать уже созданный ``drone``:

        drone = sverk_interfaces.init(Nodename="fly_head")
        use_sverk_ros2_backend(drone)

    Для отдельного теста ленты ``drone`` можно не передавать — тогда будет
    создан собственный узел с именем ``node_name``.
    """
    return configure_backend(
        SverkROS2Backend(
            drone=drone,
            led_count=led_count,
            state_timeout=state_timeout,
            node_name=node_name,
        )
    )


def current_command() -> Optional[LEDCommand]:
    """Последняя успешно применённая команда или None."""
    return _controller.last_command


def turn_off() -> LEDCommand:
    """Погасить всю ленту."""
    return set_led("solid", "off")


def close_led_backend() -> None:
    """Освободить собственный ROS 2 узел backend, если он был создан модулем."""
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


def _ensure_sverk_success(result: object, service_name: str) -> None:
    # Методы возвращают ответ ROS-сервиса; в некоторых сборках wrapper может
    # вернуть bool или None. Отсутствие success считаем успешным возвратом.
    success = result is not False and getattr(result, "success", True)
    if not success:
        message = getattr(result, "message", "команда отклонена")
        raise RuntimeError("{}: {}".format(service_name, message))


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

    class FakeState:
        leds = [object()] * 10

    class FakeLedAPI:
        def __init__(self) -> None:
            self.effects = []
            self.pixel_calls = []
            self.state_calls = 0

        def set_effect(self, effect: str, **kwargs: object) -> bool:
            self.effects.append((effect, kwargs))
            return True

        def set_leds(self, leds: list, **kwargs: object) -> bool:
            self.pixel_calls.append((leds, kwargs))
            return True

        def get_state(self, **kwargs: object) -> FakeState:
            self.state_calls += 1
            return FakeState()

    class FakeDrone:
        def __init__(self) -> None:
            self.led = FakeLedAPI()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake_drone = FakeDrone()
    use_sverk_ros2_backend(fake_drone, state_timeout=2.0)
    set_led("solid", "green")
    assert fake_drone.led.effects[-1] == (
        "fill",
        {"r": 0, "g": 255, "b": 0, "timeout": 2.0},
    )
    set_led("half-half", ("red", "blue"))
    sent_leds, call_options = fake_drone.led.pixel_calls[-1]
    assert sent_leds[:5] == [
        (index, 255, 0, 0) for index in range(5)
    ]
    assert sent_leds[5:] == [
        (index, 0, 0, 255) for index in range(5, 10)
    ]
    assert call_options == {"timeout": 2.0}
    assert fake_drone.led.state_calls == 1
    close_led_backend()
    assert not fake_drone.closed

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
