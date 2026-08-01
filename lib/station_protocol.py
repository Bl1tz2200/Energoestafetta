#!/usr/bin/env python3
"""Протокол связи БВС <-> зарядная станция (PLAN.md, Приоритет 4, Задача 4.1).

Отдельный физический канал связи (сеть/радио) между бортовым компьютером
БВС и Raspberry Pi станции в этой раскладке не предусмотрен (см. Шаг 0
PLAN.md) — обмен сигналами идёт через то, что видят камеры друг друга:

* **Запрос посадки** (БВС -> станция): БВС уже показывает красный при поиске
  станции (Табл.1, через ``led_interface.set_led``; в полётный сценарий
  индикация ещё не подключена) — станция видит этот цвет своей камерой
  (``StationVision`` из ``energy_relay_vision.py``) и по нему же определяет,
  какой именно борт подлетел (самый дорогой критерий регламента, 5 баллов —
  это отдельно, задача 2.1).
* **Разрешение на посадку** (станция -> БВС): станция отвечает своим цветом
  (``green``) на собственной ленте станции; БВС считывает его своей камерой.
* **«Зарядка завершена»**: по TASK.md (алгоритм, шаг 5) БВС-1 взлетает после
  зарядки **и команды с клавиатуры** — то есть этот сигнал в регламенте
  явно завязан на оператора, а не на автоматическое зрение, поэтому здесь
  он реализован как явный ввод с клавиатуры, а не через цвет.

Модуль не завязан на rospy/OpenCV напрямую: цвет читается через переданную
функцию-читатель (``ColorReader``), а не через ``StationVision`` напрямую —
поэтому протокол проверяется самотестом без реального ROS 1/камеры/дрона.
В боевом коде ``ColorReader`` — это обёртка над кадром с камеры и
``StationVision.detect()``/``VisionResult``, которая возвращает цвет самого
уверенного распознанного пятна (или ``None``, если в кадре ничего не
распознано).

Пример использования на стороне БВС::

    import station_protocol as proto

    proto.send_signal(proto.Signal.LANDING_REQUEST)  # красный, уже часть Табл.1
    proto.wait_for_signal(proto.Signal.LANDING_GRANTED, read_station_color,
                           timeout=30.0)
    # ... посадка, имитация зарядки ...
    proto.wait_for_takeoff_command()  # блокируется до команды оператора

Пример на стороне станции (задача 2.1, ещё не реализована отдельным
скриптом)::

    proto.wait_for_signal(proto.Signal.LANDING_REQUEST, read_drone_color,
                           timeout=120.0)
    proto.send_signal(proto.Signal.LANDING_GRANTED)  # зелёный

Проверка модуля без ROS 1/камеры/дрона::

    python3 station_protocol.py --self-test
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import Enum
from time import sleep
from typing import Callable, Optional, Tuple

import led_interface as led


ColorReader = Callable[[], Optional[str]]
"""Возвращает цвет, распознанный в текущем кадре камеры ('red', 'green', ...)
или ``None``, если ничего уверенно не распознано."""


class Signal(str, Enum):
    """Сигналы протокола БВС<->станция (см. TASK.md, алгоритм, шаги 3, 7)."""

    LANDING_REQUEST = "landing_request"
    LANDING_GRANTED = "landing_granted"


SIGNAL_COLORS: dict = {
    Signal.LANDING_REQUEST: "red",
    Signal.LANDING_GRANTED: "green",
}


@dataclass(frozen=True)
class SignalTimeout(TimeoutError):
    """Сигнал не был получен за отведённое время."""

    signal: Signal
    timeout_s: float

    def __str__(self) -> str:  # noqa: D105 - сообщение важнее докстринга
        return "Сигнал {} не получен за {}с".format(self.signal.value, self.timeout_s)


def send_signal(
    signal: Signal,
    *,
    pattern: str = "solid",
    set_led_fn: Callable[[str, str], None] = led.set_led,
) -> None:
    """Отправить сигнал протокола, выставив соответствующий цвет на ленте.

    Вызывающая сторона сама решает, чья это лента (БВС или станции) — модуль
    лишь фиксирует таблицу цвет<->сигнал в одном месте.
    """
    set_led_fn(pattern, SIGNAL_COLORS[signal])


def wait_for_signal(
    signal: Signal,
    color_reader: ColorReader,
    *,
    timeout: float,
    poll_interval: float = 0.2,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
) -> None:
    """Блокироваться, пока камера не увидит цвет, соответствующий сигналу.

    Не использует жёсткий ``sleep`` на весь тайм-аут — опрашивает
    ``color_reader`` с шагом ``poll_interval`` и выходит сразу по получении
    сигнала, поэтому задержка ожидания не выше времени одного опроса.
    """
    if timeout <= 0:
        raise ValueError("timeout должен быть больше нуля")
    if poll_interval <= 0:
        raise ValueError("poll_interval должен быть больше нуля")

    expected = SIGNAL_COLORS[signal]
    deadline = time_fn() + timeout
    while True:
        if color_reader() == expected:
            return
        if time_fn() >= deadline:
            raise SignalTimeout(signal=signal, timeout_s=timeout)
        sleep_fn(poll_interval)


def wait_for_takeoff_command(
    prompt: str = "Зарядка завершена. Введите команду для взлёта (Enter/'go'): ",
    *,
    input_fn: Callable[[str], str] = input,
    accepted: Tuple[str, ...] = ("", "go", "го", "взлёт"),
) -> str:
    """Дождаться команды оператора с клавиатуры (TASK.md, алгоритм, шаг 5:
    БВС-1 взлетает «после зарядки и команды с клавиатуры»).

    Любой ввод, не входящий в ``accepted`` (без учёта регистра и пробелов по
    краям), считается «ещё не команда» — приглашение повторяется, а не
    прерывает миссию по опечатке.
    """
    normalized_accepted = {value.strip().lower() for value in accepted}
    while True:
        response = input_fn(prompt)
        if response.strip().lower() in normalized_accepted:
            return response


def _self_test() -> int:
    # send_signal выставляет цвет из таблицы протокола на переданную ленту
    sent = []

    def fake_set_led(pattern: str, color: str) -> None:
        sent.append((pattern, color))

    send_signal(Signal.LANDING_REQUEST, set_led_fn=fake_set_led)
    send_signal(Signal.LANDING_GRANTED, pattern="blink", set_led_fn=fake_set_led)
    assert sent == [("solid", "red"), ("blink", "green")]

    # wait_for_signal: не проходит сигнал, пока цвет не совпадёт
    clock = [0.0]

    def fake_time() -> float:
        return clock[0]

    def fake_sleep(duration: float) -> None:
        clock[0] += duration

    colors = iter([None, "red", "green"])

    def fake_reader() -> Optional[str]:
        return next(colors)

    polls = []
    original_colors = colors

    def counting_reader() -> Optional[str]:
        value = next(original_colors)
        polls.append(value)
        return value

    wait_for_signal(
        Signal.LANDING_GRANTED,
        counting_reader,
        timeout=5.0,
        poll_interval=0.5,
        sleep_fn=fake_sleep,
        time_fn=fake_time,
    )
    assert polls == [None, "red", "green"]

    # wait_for_signal: TimeoutError/SignalTimeout, если цвет так и не пришёл
    clock[0] = 0.0

    def never_reader() -> Optional[str]:
        return None

    try:
        wait_for_signal(
            Signal.LANDING_REQUEST,
            never_reader,
            timeout=1.0,
            poll_interval=0.3,
            sleep_fn=fake_sleep,
            time_fn=fake_time,
        )
    except SignalTimeout as exc:
        assert exc.signal is Signal.LANDING_REQUEST
        assert str(exc)  # сообщение формируется без ошибок
    else:
        raise AssertionError("Ожидался SignalTimeout")

    # wait_for_takeoff_command: повторяет запрос на нераспознанный ввод,
    # принимает команду по списку accepted без учёта регистра/пробелов
    inputs = iter(["asdf", "  Go  "])

    def fake_input(_prompt: str) -> str:
        return next(inputs)

    result = wait_for_takeoff_command(input_fn=fake_input)
    assert result.strip().lower() == "go"

    # пустой Enter тоже принимается
    empty_inputs = iter([""])

    def fake_input_empty(_prompt: str) -> str:
        return next(empty_inputs)

    wait_for_takeoff_command(input_fn=fake_input_empty)

    print("SELF-TEST: OK")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Протокол связи БВС<->станция (Энергоэстафета)"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="проверить логику протокола без ROS 1/камеры/дрона",
    )
    parser.add_argument(
        "--wait-takeoff-command",
        action="store_true",
        help="полевая отладка: заблокироваться на реальном вводе с клавиатуры",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.self_test:
        return _self_test()
    if args.wait_takeoff_command:
        wait_for_takeoff_command()
        print("Команда на взлёт получена")
        return 0
    _build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
