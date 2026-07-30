"""Унифицированный интерфейс управления захватом груза.

Основной API полётного сценария:

    gripper_open()
    gripper_close()

Для ROS 2 на платформе Sverk передайте уже созданный объект ``drone``:

    drone = sverk_interfaces.init(Nodename="fly_head")
    use_sverk_ros2_gripper(
        drone,
        open_angle=20,
        close_angle=95,
    )

Перед запуском программы должна работать нода сервопривода:

    ros2 launch servo_control servo.launch.py

Для другого механизма по-прежнему можно использовать ``configure_gripper()``.
"""

from __future__ import annotations

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


class SverkROS2GripperBackend:
    """Адаптер официального ROS 2 API ``drone.gpio``.

    Если ``drone`` не передан, backend создаёт собственный узел
    ``sverk_interfaces`` и закрывает его в ``shutdown()``. В полётном сценарии
    предпочтительно передавать общий объект drone.
    """

    def __init__(
        self,
        drone: Any | None = None,
        *,
        open_angle: float,
        close_angle: float,
        channel: int = 0,
        enable_timeout: float = 5.0,
        startup_delay: float = 0.3,
        node_name: str = "energy_relay_gripper",
    ) -> None:
        self.open_angle = _validate_angle(open_angle, "open_angle")
        self.close_angle = _validate_angle(close_angle, "close_angle")
        if not isinstance(channel, int) or isinstance(channel, bool) or channel < 0:
            raise ValueError("channel должен быть целым числом >= 0")
        if enable_timeout <= 0:
            raise ValueError("enable_timeout должен быть больше нуля")
        if startup_delay < 0:
            raise ValueError("startup_delay не может быть отрицательным")

        self.channel = channel
        self.enable_timeout = float(enable_timeout)
        self._owns_drone = drone is None
        self._closed = False

        if drone is None:
            try:
                import sverk_interfaces
            except ImportError as exc:
                raise RuntimeError(
                    "sverk_interfaces недоступен. Выполните source "
                    "/opt/ros/*/setup.bash и source ~/sverk_ws/install/setup.bash"
                ) from exc
            drone = sverk_interfaces.init(Nodename=node_name)

        self.drone = drone
        self.gpio = getattr(drone, "gpio", None)
        if self.gpio is None:
            self._close_owned_drone()
            raise RuntimeError("У объекта drone отсутствует API drone.gpio")

        try:
            self.gpio.servo_select_channel(channel)
            response = self.gpio.servo_enable(timeout=self.enable_timeout)
            _ensure_ros_success(response, "/servo_control/enable")
            if startup_delay:
                sleep(startup_delay)
        except Exception:
            self._close_owned_drone()
            raise

    def open(self) -> None:
        self._ensure_open()
        self.gpio.servo_set_angle(self.open_angle)

    def close(self) -> None:
        self._ensure_open()
        self.gpio.servo_set_angle(self.close_angle)

    def shutdown(self) -> None:
        """Отключить PWM и закрыть только созданный этим backend ROS-узел."""
        if self._closed:
            return
        self._closed = True
        try:
            response = self.gpio.servo_disable(timeout=self.enable_timeout)
            _ensure_ros_success(response, "/servo_control/enable")
        finally:
            self._close_owned_drone()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Backend захвата уже остановлен")

    def _close_owned_drone(self) -> None:
        if self._owns_drone and self.drone is not None:
            close = getattr(self.drone, "close", None)
            if callable(close):
                close()
            self.drone = None


def use_sverk_ros2_gripper(
    drone: Any | None = None,
    *,
    open_angle: float,
    close_angle: float,
    channel: int = 0,
    enable_timeout: float = 5.0,
    startup_delay: float = 0.3,
    node_name: str = "energy_relay_gripper",
) -> SverkROS2GripperBackend:
    """Настроить захват через ROS 2 ``sverk_interfaces`` и включить PWM."""
    backend = SverkROS2GripperBackend(
        drone,
        open_angle=open_angle,
        close_angle=close_angle,
        channel=channel,
        enable_timeout=enable_timeout,
        startup_delay=startup_delay,
        node_name=node_name,
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


def _validate_angle(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(name + " должен быть числом")
    angle = float(value)
    if not 0.0 <= angle <= 180.0:
        raise ValueError(name + " должен лежать в диапазоне 0..180")
    return angle


def _ensure_ros_success(response: object, operation: str) -> None:
    success = response is not False and getattr(response, "success", True)
    if not success:
        message = getattr(response, "message", "команда отклонена")
        raise RuntimeError("{}: {}".format(operation, message))
