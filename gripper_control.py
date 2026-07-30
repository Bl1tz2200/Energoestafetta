"""Унифицированный интерфейс управления захватом груза.

Аппаратные функции открытия и закрытия передаются через configure_gripper().
Это позволяет использовать один API с сервоприводом, реле или другим механизмом.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from threading import Lock
from time import sleep


class GripperState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


_open_action: Callable[[], None] | None = None
_close_action: Callable[[], None] | None = None
_state = GripperState.UNKNOWN
_lock = Lock()


def configure_gripper(
    *,
    open_action: Callable[[], None],
    close_action: Callable[[], None],
) -> None:
    """Подключить аппаратные функции управления захватом."""
    if not callable(open_action) or not callable(close_action):
        raise TypeError("open_action и close_action должны быть вызываемыми функциями")

    global _open_action, _close_action
    _open_action = open_action
    _close_action = close_action


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
    return _state
