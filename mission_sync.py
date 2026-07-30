#!/usr/bin/env python3
"""Синхронный взлёт и асинхронное выполнение задач (PLAN.md, Приоритет 4,
Задача 4.2).

Два пункта регламента (TASK.md, алгоритм, шаги 1-2):

1. **Синхронный взлёт двух БВС** — БВС-1 и БВС-2 запускаются как отдельные
   скрипты на отдельных бортовых компьютерах (Orange Pi у каждого свой), т.е.
   это два разных процесса, возможно на разных машинах одной сети (в
   комплекте есть роутер — TASK.md, «Требования к железу»). Синхронизировать
   момент взлёта между независимыми процессами через общий Python-объект
   (``threading.Event`` и т.п.) нельзя — нужен канал поверх сети.
   ``TakeoffBarrier`` — координатор на основе TCP-рандеву: и БВС-1, и БВС-2
   подключаются к общему координатору (например, к ноутбуку оператора) и
   блокируются в ``wait_at_barrier()``; как только подключились оба, всем
   участникам одновременно уходит сигнал «взлёт».
2. **Асинхронное выполнение задач через события/таймеры, не через жёсткий
   ``sleep``** — чтобы попытка не разваливалась, если один из шагов
   задержался. ``run_concurrently()`` запускает несколько задач каждую в
   своём потоке и ждёт завершения всех (или таймаута), не блокируя одну
   задачу ожиданием другой; полезно для наземной оркестрации/самотеста двух
   миссий сразу (например, в Gazebo с двумя симулированными бортами) и как
   более надёжная замена ручному ``time.sleep`` на фиксированный интервал,
   когда на самом деле нужно ждать конкретное условие.

Ни одна функция здесь не блокирует остальные модули репозитория — это
самостоятельный слой синхронизации, который полётные скрипты подключают по
желанию (см. ``bvs1_flight.py``/``bvs2_flight.py``: интеграция — отдельный
шаг Приоритета 5, «Полная сборка»).

Проверка модуля без реального ROS 1/дрона/сети (используется loopback)::

    python3 mission_sync.py --self-test

Полевая отладка координатора на реальной сети — до подключения к полётным
скриптам, чтобы сначала проверить сам барьер отдельно::

    # на ноутбуке оператора (координатор):
    python3 mission_sync.py --serve --port 5757 --parties 2

    # на каждом БВС (клиент):
    python3 mission_sync.py --wait --host <ip-координатора> --port 5757
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from dataclasses import dataclass
from time import sleep
from typing import Callable, Dict, List, Optional

_GO_MESSAGE = b"GO"


@dataclass(frozen=True)
class BarrierConfig:
    """Параметры TCP-координатора синхронного взлёта."""

    host: str = "0.0.0.0"
    port: int = 5757
    party_count: int = 2
    accept_timeout_s: float = 120.0


def serve_takeoff_barrier(
    config: BarrierConfig,
    *,
    ready_callback: Optional[Callable[[int], None]] = None,
) -> None:
    """Координатор: дождаться подключения ``party_count`` участников и
    отпустить их всех одновременно.

    Вызывается один раз с одной стороны (обычно — с ноутбука оператора,
    который виден по сети обоим БВС). Не отпускает никого, пока не
    подключились все участники — это и есть точка синхронизации взлёта.
    """
    if config.party_count <= 0:
        raise ValueError("party_count должен быть больше нуля")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((config.host, config.port))
        server.listen(config.party_count)
        server.settimeout(config.accept_timeout_s)

        connections: List[socket.socket] = []
        try:
            while len(connections) < config.party_count:
                conn, _addr = server.accept()
                connections.append(conn)
                if ready_callback is not None:
                    ready_callback(len(connections))

            for conn in connections:
                conn.sendall(_GO_MESSAGE)
        finally:
            for conn in connections:
                conn.close()
    finally:
        server.close()


def wait_at_takeoff_barrier(
    host: str,
    port: int,
    *,
    connect_timeout_s: float = 60.0,
    retry_interval_s: float = 1.0,
    recv_timeout_s: float = 120.0,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
) -> None:
    """Участник: подключиться к координатору и заблокироваться до общего
    сигнала «взлёт». Подключение переповторяется, пока координатор ещё не
    поднят (полезно, если скрипты БВС стартуют чуть раньше координатора)."""
    if connect_timeout_s <= 0:
        raise ValueError("connect_timeout_s должен быть больше нуля")

    deadline = time_fn() + connect_timeout_s
    sock: Optional[socket.socket] = None
    last_error: Optional[OSError] = None
    while sock is None:
        try:
            sock = socket.create_connection((host, port), timeout=5.0)
        except OSError as exc:
            last_error = exc
            if time_fn() >= deadline:
                raise TimeoutError(
                    "Не удалось подключиться к координатору взлёта "
                    "{}:{} за {}с".format(host, port, connect_timeout_s)
                ) from last_error
            sleep_fn(retry_interval_s)

    try:
        sock.settimeout(recv_timeout_s)
        data = sock.recv(len(_GO_MESSAGE))
        if data != _GO_MESSAGE:
            raise RuntimeError(
                "Координатор взлёта прислал неожиданный ответ: {!r}".format(data)
            )
    finally:
        sock.close()


@dataclass
class TaskResult:
    """Итог одной задачи, запущенной через ``run_concurrently``."""

    name: str
    error: Optional[BaseException]
    duration_s: float


def run_concurrently(
    tasks: Dict[str, Callable[[], None]],
    *,
    timeout: Optional[float] = None,
) -> List[TaskResult]:
    """Запустить несколько задач параллельно (каждую в своём потоке) и
    собрать результат каждой, не давая задержке одной задачи заблокировать
    ожидание другой.

    Ошибка внутри задачи не прерывает остальные — она попадает в
    соответствующий ``TaskResult.error``, вызывающий код сам решает, что с
    этим делать (в отличие от жёсткого ``sleep`` между шагами, где сбой
    одного шага мог бы тихо провалить всю попытку).
    """
    if not tasks:
        return []

    results: Dict[str, TaskResult] = {}
    lock = threading.Lock()

    def _run(name: str, fn: Callable[[], None]) -> None:
        start = time.monotonic()
        error: Optional[BaseException] = None
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - репортим любую ошибку задачи
            error = exc
        duration = time.monotonic() - start
        with lock:
            results[name] = TaskResult(name=name, error=error, duration_s=duration)

    threads = [
        threading.Thread(target=_run, args=(name, fn), name=name, daemon=True)
        for name, fn in tasks.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout)

    ordered: List[TaskResult] = []
    for name in tasks:
        with lock:
            result = results.get(name)
        if result is None:
            result = TaskResult(
                name=name,
                error=TimeoutError(
                    "Задача {!r} не завершилась за {}с".format(name, timeout)
                ),
                duration_s=float(timeout or 0.0),
            )
        ordered.append(result)
    return ordered


def _self_test() -> int:
    # run_concurrently: обе задачи выполняются параллельно, а не одна за
    # другой — суммарное время близко к самой долгой задаче, а не к сумме.
    events_order: List[str] = []
    order_lock = threading.Lock()

    def slow_task() -> None:
        time.sleep(0.2)
        with order_lock:
            events_order.append("slow")

    def fast_task() -> None:
        with order_lock:
            events_order.append("fast")

    start = time.monotonic()
    results = run_concurrently({"slow": slow_task, "fast": fast_task})
    elapsed = time.monotonic() - start
    assert elapsed < 0.4, "задачи не должны выполняться последовательно"
    assert {r.name for r in results} == {"slow", "fast"}
    assert all(r.error is None for r in results)
    assert events_order[0] == "fast"  # быстрая задача не ждёт медленную

    # run_concurrently: ошибка одной задачи не мешает другой отработать и
    # попадает в TaskResult вызывающей стороне
    def failing_task() -> None:
        raise ValueError("boom")

    def ok_task() -> None:
        return None

    results = run_concurrently({"fail": failing_task, "ok": ok_task})
    by_name = {r.name: r for r in results}
    assert isinstance(by_name["fail"].error, ValueError)
    assert by_name["ok"].error is None

    # run_concurrently: зависшая задача не блокирует получение результатов
    # остальных дольше timeout
    def hanging_task() -> None:
        time.sleep(5.0)

    start = time.monotonic()
    results = run_concurrently({"hang": hanging_task, "ok": ok_task}, timeout=0.2)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0
    by_name = {r.name: r for r in results}
    assert isinstance(by_name["hang"].error, TimeoutError)
    assert by_name["ok"].error is None

    # TakeoffBarrier: оба участника отпускаются только после того, как
    # подключились оба - и делают это почти одновременно (не по очереди)
    config = BarrierConfig(host="127.0.0.1", port=0, party_count=2, accept_timeout_s=5.0)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((config.host, 0))
    real_port = server.getsockname()[1]
    server.close()
    config = BarrierConfig(
        host="127.0.0.1", port=real_port, party_count=2, accept_timeout_s=5.0
    )

    release_times: Dict[str, float] = {}
    release_lock = threading.Lock()

    def client(name: str, delay_before_connect: float) -> None:
        time.sleep(delay_before_connect)
        wait_at_takeoff_barrier(
            config.host, config.port, connect_timeout_s=5.0, retry_interval_s=0.1
        )
        with release_lock:
            release_times[name] = time.monotonic()

    server_thread = threading.Thread(
        target=serve_takeoff_barrier, args=(config,), daemon=True
    )
    server_thread.start()

    client_threads = [
        threading.Thread(target=client, args=("bvs1", 0.0), daemon=True),
        threading.Thread(target=client, args=("bvs2", 0.3), daemon=True),
    ]
    for thread in client_threads:
        thread.start()
    for thread in client_threads:
        thread.join(timeout=5.0)
    server_thread.join(timeout=5.0)

    assert set(release_times) == {"bvs1", "bvs2"}
    # bvs1 подключился первым, но не улетел раньше времени в одиночку —
    # он должен был дождаться bvs2 (иначе разница была бы >= 0.3с)
    assert abs(release_times["bvs1"] - release_times["bvs2"]) < 0.15

    # TakeoffBarrier: если подключился только один участник, координатор не
    # отпускает никого и завершается таймаутом accept(); одинокий клиент не
    # виснет молча - соединение закрывается без GO, и это заметно как ошибка
    lonely_config = BarrierConfig(
        host="127.0.0.1", port=real_port + 1, party_count=2, accept_timeout_s=0.5
    )
    lone_client_error: List[Optional[BaseException]] = [None]

    def lone_client() -> None:
        try:
            wait_at_takeoff_barrier(
                lonely_config.host,
                lonely_config.port,
                connect_timeout_s=0.3,
                retry_interval_s=0.05,
            )
        except BaseException as exc:  # noqa: BLE001 - хотим увидеть саму ошибку
            lone_client_error[0] = exc

    lone_thread = threading.Thread(target=lone_client, daemon=True)
    lone_thread.start()
    try:
        serve_takeoff_barrier(lonely_config)
    except OSError:
        pass  # accept() таймаутится - второй участник так и не пришёл
    else:
        raise AssertionError(
            "Координатор не должен был отпустить единственного участника"
        )
    finally:
        lone_thread.join(timeout=2.0)
    assert isinstance(lone_client_error[0], (RuntimeError, OSError))

    print("SELF-TEST: OK")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Синхронный взлёт и асинхронное выполнение задач (Энергоэстафета)"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="проверить логику синхронизации без реальной сети/ROS 1/дрона",
    )
    parser.add_argument("--serve", action="store_true", help="запустить координатора взлёта")
    parser.add_argument("--wait", action="store_true", help="подключиться к координатору как БВС")
    parser.add_argument("--host", default="0.0.0.0", help="адрес координатора")
    parser.add_argument("--port", type=int, default=5757)
    parser.add_argument("--parties", type=int, default=2, help="сколько участников ждать (--serve)")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.self_test:
        return _self_test()
    if args.serve:
        config = BarrierConfig(host=args.host, port=args.port, party_count=args.parties)
        print("Ожидаю {} участников на {}:{}...".format(args.parties, args.host, args.port))
        serve_takeoff_barrier(
            config, ready_callback=lambda n: print("Подключился участник {}/{}".format(n, args.parties))
        )
        print("Сигнал взлёта отправлен всем участникам")
        return 0
    if args.wait:
        print("Подключаюсь к координатору {}:{}...".format(args.host, args.port))
        wait_at_takeoff_barrier(args.host, args.port)
        print("Получен сигнал взлёта")
        return 0
    _build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
