"""Унифицированный интерфейс управления захватом груза.

Основной API полётного сценария:

    gripper_open()
    gripper_close()

Дрон соревнования — Skyris Technic 6S (PX4 + Orange Pi 5 Pro). На этой
платформе сервопривод НЕ управляется через ROS — используется системная
утилита ``gpio`` (WiringPi) на самом Orange Pi
(см. docs.skyris.ru/technic6S/GPIO.html)::

    use_technic_gpio_gripper(
        pin=17,
        open_pulse=1000,
        close_pulse=2000,
    )

``open_pulse``/``close_pulse`` — это не углы, а «сырые» значения PWM
(аналог ширины импульса сервопривода), их нужно подобрать под конкретный
захват по факту на площадке. Для другого механизма по-прежнему можно
использовать ``configure_gripper()``.

Проверка модуля без реального Orange Pi/``gpio`` (у ``TechnicGPIOBackend``
есть аргумент ``runner``, который подставляется заглушкой вместо реального
процесса ``gpio``)::

    python3 gripper_control.py --self-test
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from enum import Enum
from threading import Lock
from time import sleep
from typing import Any


class GripperState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


_open_action: Callable[[], None] | None = None
_close_action: Callable[[], None] | None = None
_shutdown_action: Callable[[], None] | None = None
_state = GripperState.UNKNOWN
_lock = Lock()


def configure_gripper(
    *,
    open_action: Callable[[], None],
    close_action: Callable[[], None],
    shutdown_action: Callable[[], None] | None = None,
) -> None:
    """Подключить аппаратные функции управления захватом.

    Состояние сбрасывается в UNKNOWN, чтобы первая команда всегда дошла до
    заново подключённого оборудования.
    """
    if not callable(open_action) or not callable(close_action):
        raise TypeError("open_action и close_action должны быть вызываемыми функциями")
    if shutdown_action is not None and not callable(shutdown_action):
        raise TypeError("shutdown_action должен быть вызываемой функцией или None")

    global _open_action, _close_action, _shutdown_action, _state
    with _lock:
        _open_action = open_action
        _close_action = close_action
        _shutdown_action = shutdown_action
        _state = GripperState.UNKNOWN


class TechnicGPIOBackend:
    """Управление сервоприводом Skyris Technic 6S через утилиту ``gpio``.

    На этой платформе сервопривод НЕ является ROS-сервисом: скрипт вызывает
    системную утилиту ``gpio`` (WiringPi) на Orange Pi напрямую, как в
    примерах docs.skyris.ru/technic6S/GPIO.html::

        gpio mode <pin> pwm
        gpio pwmr <pin> <pwm_range>
        gpio pwmc <pin> <pwm_clock>
        gpio pwm <pin> <value>

    ``open_pulse``/``close_pulse`` — это не углы 0..180, а «сырые» значения
    PWM (в примерах документации использовались 1000/1500/2000 при
    ``pwm_range=20000, pwm_clock=24``). Подбираются по факту под конкретный
    сервопривод/захват на площадке.

    ``runner`` можно подменить в тестах, чтобы не вызывать реальный
    процесс ``gpio``.
    """

    def __init__(
        self,
        pin: int,
        *,
        open_pulse: int,
        close_pulse: int,
        pwm_range: int = 20000,
        pwm_clock: int = 24,
        startup_delay: float = 0.3,
        runner: Callable[..., None] | None = None,
    ) -> None:
        if not isinstance(pin, int) or isinstance(pin, bool) or pin < 0:
            raise ValueError("pin должен быть целым числом >= 0")
        for name, value in (("open_pulse", open_pulse), ("close_pulse", close_pulse)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(name + " должен быть положительным целым числом")
        if pwm_range <= 0 or pwm_clock <= 0:
            raise ValueError("pwm_range и pwm_clock должны быть больше нуля")
        if startup_delay < 0:
            raise ValueError("startup_delay не может быть отрицательным")

        self.pin = pin
        self.open_pulse = open_pulse
        self.close_pulse = close_pulse
        self._closed = False
        self._run = runner or _run_gpio_cli

        self._run("mode", pin, "pwm")
        self._run("pwmr", pin, pwm_range)
        self._run("pwmc", pin, pwm_clock)
        if startup_delay:
            sleep(startup_delay)

    def open(self) -> None:
        self._ensure_open()
        self._run("pwm", self.pin, self.open_pulse)

    def close(self) -> None:
        self._ensure_open()
        self._run("pwm", self.pin, self.close_pulse)

    def shutdown(self) -> None:
        """Обнулить PWM на пине. Повторный вызов безопасен."""
        if self._closed:
            return
        self._closed = True
        self._run("pwm", self.pin, 0)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Backend захвата уже остановлен")


def _run_gpio_cli(*args: Any) -> None:
    import subprocess

    subprocess.run(["gpio", *map(str, args)], check=True)


def use_technic_gpio_gripper(
    pin: int,
    *,
    open_pulse: int,
    close_pulse: int,
    pwm_range: int = 20000,
    pwm_clock: int = 24,
    startup_delay: float = 0.3,
) -> TechnicGPIOBackend:
    """Настроить захват через ``gpio`` CLI на Skyris Technic 6S."""
    backend = TechnicGPIOBackend(
        pin,
        open_pulse=open_pulse,
        close_pulse=close_pulse,
        pwm_range=pwm_range,
        pwm_clock=pwm_clock,
        startup_delay=startup_delay,
    )
    configure_gripper(
        open_action=backend.open,
        close_action=backend.close,
        shutdown_action=backend.shutdown,
    )
    return backend


def gripper_close(*, delay: float = 0.0) -> None:
    """Закрыть захват.

    Args:
        delay: пауза после команды в секундах.
    """
    if delay < 0:
        raise ValueError("delay не может быть отрицательным")

    global _state
    with _lock:
        if _close_action is None:
            raise RuntimeError("Захват не настроен: вызови configure_gripper()")

        if _state is GripperState.CLOSED:
            return

        _close_action()
        _state = GripperState.CLOSED

    if delay:
        sleep(delay)


def gripper_open(*, delay: float = 0.0) -> None:
    """Открыть захват.

    Args:
        delay: пауза после команды в секундах.
    """
    if delay < 0:
        raise ValueError("delay не может быть отрицательным")

    global _state
    with _lock:
        if _open_action is None:
            raise RuntimeError("Захват не настроен: вызови configure_gripper()")

        if _state is GripperState.OPEN:
            return

        _open_action()
        _state = GripperState.OPEN

    if delay:
        sleep(delay)


def get_gripper_state() -> GripperState:
    """Вернуть последнее известное состояние захвата."""
    with _lock:
        return _state


def shutdown_gripper() -> None:
    """Отключить настроенный backend и сбросить известное состояние."""
    global _open_action, _close_action, _shutdown_action, _state
    with _lock:
        action = _shutdown_action
        _open_action = None
        _close_action = None
        _shutdown_action = None
        _state = GripperState.UNKNOWN
    if action is not None:
        action()


def _self_test() -> int:
    # Гарантируем чистое состояние независимо от порядка запуска тестов.
    shutdown_gripper()

    # Захват не настроен -> gripper_open()/gripper_close() кидают RuntimeError
    try:
        gripper_open()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Ожидался RuntimeError для ненастроенного захвата")

    calls: list = []

    def fake_runner(*args: Any) -> None:
        calls.append(args)

    # startup_delay=0, чтобы самотест не спал реально
    backend = TechnicGPIOBackend(
        17, open_pulse=1000, close_pulse=2000, startup_delay=0, runner=fake_runner
    )
    assert calls == [("mode", 17, "pwm"), ("pwmr", 17, 20000), ("pwmc", 17, 24)]

    configure_gripper(
        open_action=backend.open,
        close_action=backend.close,
        shutdown_action=backend.shutdown,
    )
    assert get_gripper_state() == GripperState.UNKNOWN

    gripper_close()
    assert calls[-1] == ("pwm", 17, 2000)
    assert get_gripper_state() == GripperState.CLOSED

    calls_before_repeat = len(calls)
    gripper_close()  # повторная команда в том же состоянии — no-op
    assert len(calls) == calls_before_repeat

    gripper_open()
    assert calls[-1] == ("pwm", 17, 1000)
    assert get_gripper_state() == GripperState.OPEN

    shutdown_gripper()
    assert calls[-1] == ("pwm", 17, 0)
    assert get_gripper_state() == GripperState.UNKNOWN

    # После shutdown backend больше не принимает команды
    try:
        backend.open()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Ожидался RuntimeError после shutdown()")

    # Валидация параметров конструктора
    invalid_kwargs = (
        {"pin": -1, "open_pulse": 1000, "close_pulse": 2000},
        {"pin": 17, "open_pulse": 0, "close_pulse": 2000},
        {"pin": 17, "open_pulse": 1000, "close_pulse": 2000, "pwm_range": 0},
    )
    for kwargs in invalid_kwargs:
        try:
            TechnicGPIOBackend(startup_delay=0, runner=fake_runner, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Ожидался ValueError для {!r}".format(kwargs))

    try:
        configure_gripper(open_action="not callable", close_action=backend.close)
    except TypeError:
        pass
    else:
        raise AssertionError("Ожидался TypeError для нефункционального open_action")

    try:
        gripper_open(delay=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Ожидался ValueError для отрицательного delay")

    shutdown_gripper()
    print("SELF-TEST: OK")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Управление захватом груза (Энергоэстафета)"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="проверить логику захвата без реального Orange Pi/gpio",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.self_test:
        return _self_test()
    _build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
