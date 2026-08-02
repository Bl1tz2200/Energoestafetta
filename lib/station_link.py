#!/usr/bin/env python3
"""Связь БВС <-> зарядная станция по Wi-Fi: одноранговая, без сервера.

Центральной машины-координатора в комплекте нет, поэтому и в протоколе её
нет: обе стороны открывают **один и тот же UDP-порт** и шлют датаграммы
напрямую на IP собеседника, заданные флагами. Никто никого не «принимает»,
порядок включения не важен, выключенный собеседник не вешает цикл — у UDP
нет установки соединения, отправка не блокируется.

Почему не TCP: там одна из сторон обязана слушать (то есть быть сервером), а
неподнятый собеседник даёт зависание на ``connect`` и таймауты прямо внутри
кадрового цикла станции.

Надёжность вместо гарантий TCP:

* каждое сообщение уходит ``repeat`` раз подряд и несёт номер ``seq`` плюс
  идентификатор запуска процесса ``session`` — повторы на приёме
  отбрасываются (``_Dedup``), перезапуск борта не путается со старыми
  номерами;
* все сообщения идемпотентны: повторный ``landing_granted`` ничего не ломает;
* важные состояния переспрашиваются самим сценарием (борт повторяет запрос,
  пока висит над станцией), поэтому единичная потеря пакета не срывает
  попытку.

**Опознание борта — по IP.** Список задаётся парами ``имя=адрес``; датаграмма
с адреса вне списка отбрасывается, как и датаграмма, где имя отправителя не
совпадает с зарегистрированным для этого IP. Это одновременно и ответ на
вопрос «какой именно борт подлетел», и защита от чужого процесса в той же
сети роутера.

**Потоков нет.** Приём — явный вызов ``Link.pump()`` из основного цикла
(кадрового у станции, полётного у борта): одна точка входа, предсказуемый
порядок, самотест без сети и без гонок.

Стороны протокола::

    БВС-1 завис над кубом
      БВС-1   -> станция   landing_request   (повтор, пока не ответят)
      станция: IP в списке + камера видит нужный цвет
      станция -> БВС-1     landing_granted   + зелёная лента
      БВС-1   -> станция   landed            (сел, моторы выключены)
      станция: гасит ленту, встаёт в «занято»
      БВС-1   -> станция   departed          (взлетел) -> станция свободна

Проверка без сети и железа::

    python3 station_link.py --self-test

Полевая отладка (порт один и тот же с обеих сторон)::

    # на станции: смотреть эфир и отвечать на hello
    python3 station_link.py --id station1 --peer uav1=192.168.0.11 --listen

    # на борту: проверить связь и сымитировать запрос посадки
    python3 station_link.py --id uav1 --peer station1=192.168.0.21 --hello
    python3 station_link.py --id uav1 --peer station1=192.168.0.21 --request
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import select
import socket
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


PROTOCOL_VERSION = 1
DEFAULT_PORT = 5801
MAX_DATAGRAM = 4096  # наши сообщения — сотни байт; всё крупнее не наше

Peers = Dict[str, str]  # имя узла -> IP
StopFn = Callable[[], bool]


class Kind(str, Enum):
    """Сообщения протокола. Значение — то, что уходит в JSON."""

    LANDING_REQUEST = "landing_request"  # БВС: прошу посадку, вот кто я
    LANDING_GRANTED = "landing_granted"  # станция: садись
    LANDING_DENIED = "landing_denied"  # станция: занята/не тот борт (с причиной)
    LANDED = "landed"  # БВС: сел, моторы выключены
    DEPARTED = "departed"  # БВС: улетел, станция свободна
    HELLO = "hello"  # проверка связи, состояние не меняет


@dataclass(frozen=True)
class Message:
    """Одно сообщение протокола = одна UDP-датаграмма."""

    kind: str
    sender: str
    target: str = ""  # пусто — всем адресатам из списка
    seq: int = 0
    session: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    source_ip: str = ""  # заполняет приёмник, в датаграмму не входит

    def to_bytes(self) -> bytes:
        return json.dumps(
            {
                "v": PROTOCOL_VERSION,
                "kind": self.kind,
                "from": self.sender,
                "to": self.target,
                "seq": self.seq,
                "sid": self.session,
                "data": self.data,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    def describe(self) -> str:
        tail = " " + json.dumps(self.data, ensure_ascii=False) if self.data else ""
        return "{} {} -> {} (seq {}, {}){}".format(
            self.kind,
            self.sender,
            self.target or "всем",
            self.seq,
            self.source_ip or "?",
            tail,
        )


def decode_message(payload: bytes, source_ip: str) -> Optional[Message]:
    """Разобрать датаграмму. ``None`` — мусор или чужая версия протокола.

    Приём чужого трафика на общем порту — норма, поэтому ничего не бросаем:
    непонятая датаграмма просто игнорируется.
    """
    if not payload or len(payload) > MAX_DATAGRAM:
        return None
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("v") != PROTOCOL_VERSION:
        return None

    kind = raw.get("kind")
    sender = raw.get("from")
    if not isinstance(kind, str) or not isinstance(sender, str) or not sender:
        return None
    data = raw.get("data")
    target = raw.get("to")
    session = raw.get("sid")
    seq = raw.get("seq")
    return Message(
        kind=kind,
        sender=sender,
        target=target if isinstance(target, str) else "",
        seq=int(seq) if isinstance(seq, int) else 0,
        session=session if isinstance(session, str) else "",
        data=data if isinstance(data, dict) else {},
        source_ip=source_ip,
    )


# ═══════════════════════════════════════════════════════════════════════
#  ТРАНСПОРТ
# ═══════════════════════════════════════════════════════════════════════


class Transport(ABC):
    """Как байты уходят и приходят. Подменяется в самотесте на память."""

    @abstractmethod
    def send(self, payload: bytes, ip: str) -> bool:
        """Отправить датаграмму. ``False`` — не ушла (собеседник недоступен)."""

    @abstractmethod
    def receive(self, timeout: float) -> Optional[Tuple[bytes, str]]:
        """Забрать одну датаграмму или ``None``, если за ``timeout`` ничего нет."""

    def close(self) -> None:
        """Освободить ресурсы. По умолчанию освобождать нечего."""


class UdpTransport(Transport):
    """UDP-сокет на фиксированном порту: и приём, и отправка на явные IP."""

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        *,
        bind_host: str = "0.0.0.0",
        peer_port: Optional[int] = None,
    ) -> None:
        self.port = int(port)
        # В бою порт с обеих сторон один и тот же; разные нужны только чтобы
        # поднять оба узла на одной машине (проверка сокетов, --loopback-test).
        self.peer_port = int(peer_port) if peer_port else self.port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._socket.bind((bind_host, self.port))
        except OSError as exc:
            self._socket.close()
            raise RuntimeError(
                "Не удалось занять UDP-порт {}: {}".format(self.port, exc)
            ) from exc

    def send(self, payload: bytes, ip: str) -> bool:
        try:
            self._socket.sendto(payload, (ip, self.peer_port))
            return True
        except OSError:
            # Недоступный борт не должен ронять станцию: следующая попытка
            # уйдёт со следующим повтором сообщения.
            return False

    def receive(self, timeout: float) -> Optional[Tuple[bytes, str]]:
        ready, _, _ = select.select([self._socket], [], [], max(0.0, timeout))
        if not ready:
            return None
        try:
            payload, address = self._socket.recvfrom(MAX_DATAGRAM)
        except OSError:
            return None
        return payload, address[0]

    def close(self) -> None:
        self._socket.close()


class LoopbackNetwork:
    """Сеть в памяти для самотеста: доставка по IP, без сокетов.

    ``drop`` вызывается на каждой отправляемой датаграмме и, вернув ``True``,
    её теряет — так проверяется, что повторы делают своё дело.
    """

    def __init__(self, *, drop: Optional[Callable[[int], bool]] = None) -> None:
        self._inboxes: Dict[str, List[Tuple[bytes, str]]] = {}
        self._drop = drop
        self.sent = 0
        self.lost = 0

    def transport(self, ip: str) -> "LoopbackTransport":
        self._inboxes.setdefault(ip, [])
        return LoopbackTransport(self, ip)

    def deliver(self, payload: bytes, destination: str, source: str) -> bool:
        self.sent += 1
        if self._drop is not None and self._drop(self.sent):
            self.lost += 1
            return True  # для отправителя потеря в сети неотличима от успеха
        inbox = self._inboxes.get(destination)
        if inbox is None:
            return False  # никто не слушает этот адрес
        inbox.append((payload, source))
        return True

    def take(self, ip: str) -> Optional[Tuple[bytes, str]]:
        inbox = self._inboxes.get(ip)
        if not inbox:
            return None
        return inbox.pop(0)


class LoopbackTransport(Transport):
    """Сторона ``LoopbackNetwork``, притворяющаяся сокетом одного узла."""

    def __init__(self, network: LoopbackNetwork, ip: str) -> None:
        self.network = network
        self.ip = ip

    def send(self, payload: bytes, ip: str) -> bool:
        return self.network.deliver(payload, ip, self.ip)

    def receive(self, timeout: float) -> Optional[Tuple[bytes, str]]:
        return self.network.take(self.ip)


# ═══════════════════════════════════════════════════════════════════════
#  УЗЕЛ
# ═══════════════════════════════════════════════════════════════════════


class _Dedup:
    """Повторы одного сообщения принимаем один раз.

    Ключ — (отправитель, запуск процесса, номер). Запуск в ключе нужен, чтобы
    перезапущенный борт, у которого номера снова пошли с единицы, не оказался
    навсегда «уже слышанным».
    """

    def __init__(self, capacity: int = 512) -> None:
        self._seen: "OrderedDict[Tuple[str, str, int], None]" = OrderedDict()
        self._capacity = int(capacity)

    def is_new(self, message: Message) -> bool:
        if message.seq <= 0:
            return True  # без номера дедуп невозможен — пропускаем как есть
        key = (message.sender, message.session, message.seq)
        if key in self._seen:
            return False
        self._seen[key] = None
        while len(self._seen) > self._capacity:
            self._seen.popitem(last=False)
        return True


class Link:
    """Узел протокола: знает свой идентификатор и адреса собеседников."""

    def __init__(
        self,
        node_id: str,
        transport: Transport,
        *,
        peers: Optional[Peers] = None,
        strict: bool = True,
        session: Optional[str] = None,
        repeat: int = 3,
    ) -> None:
        if not node_id:
            raise ValueError("Узлу нужен идентификатор (--id)")
        if repeat < 1:
            raise ValueError("repeat должен быть не меньше единицы")

        self.node_id = node_id
        self.transport = transport
        self.peers: Peers = dict(peers or {})
        self.strict = bool(strict)
        self.repeat = int(repeat)
        self.session = session or os.urandom(3).hex()

        self._by_ip = {ip: name for name, ip in self.peers.items()}
        self._dedup = _Dedup()
        self._seq = 0
        self.sent = 0
        self.received = 0
        self.rejected: List[str] = []  # причины отказов — для отладки в поле

    # ------------------------------------------------------------ отправка
    def send(
        self,
        kind: str,
        *,
        target: str = "",
        data: Optional[Dict[str, Any]] = None,
        repeat: Optional[int] = None,
    ) -> int:
        """Отправить сообщение. Возвращает число ушедших датаграмм.

        ``target`` — имя собеседника из списка; пусто — всем сразу. Повтор
        уходит подряд, без пауз: цикл вызывающей стороны не должен вставать
        из-за протокола, а от единичной потери спасает уже сам повтор.
        """
        if target and target not in self.peers:
            raise KeyError(
                "Адресат {} не задан флагом --peer/--drone".format(target)
            )
        destinations = [self.peers[target]] if target else list(self.peers.values())
        if not destinations:
            return 0

        self._seq += 1
        message = Message(
            kind=str(getattr(kind, "value", kind)),
            sender=self.node_id,
            target=target,
            seq=self._seq,
            session=self.session,
            data=dict(data or {}),
        )
        payload = message.to_bytes()

        delivered = 0
        for _ in range(repeat if repeat is not None else self.repeat):
            for ip in destinations:
                if self.transport.send(payload, ip):
                    delivered += 1
        self.sent += delivered
        return delivered

    # -------------------------------------------------------------- приём
    def pump(self, timeout: float = 0.0) -> List[Message]:
        """Разобрать всё, что пришло, и вернуть принятые сообщения.

        Вызывается из основного цикла вызывающей стороны — своих потоков у
        протокола нет. ``timeout`` больше нуля ждёт первую датаграмму,
        остальные разбираются без ожидания.
        """
        messages: List[Message] = []
        wait = max(0.0, timeout)
        while True:
            item = self.transport.receive(wait)
            wait = 0.0
            if item is None:
                return messages
            payload, source_ip = item
            message = decode_message(payload, source_ip)
            if message is None:
                self.rejected.append("не наш формат от {}".format(source_ip))
                continue
            if message.sender == self.node_id:
                continue  # собственная датаграмма, вернувшаяся к нам
            if not self._accept(message):
                continue
            if not self._dedup.is_new(message):
                continue
            if message.target and message.target != self.node_id:
                continue  # адресовано другому узлу
            self.received += 1
            messages.append(message)

    def _accept(self, message: Message) -> bool:
        """Проверка «кто это» по IP: собственно опознание борта."""
        if not self.strict:
            return True
        known = self._by_ip.get(message.source_ip)
        if known is None:
            self.rejected.append(
                "{} не в списке адресов (сообщение {})".format(
                    message.source_ip, message.kind
                )
            )
            return False
        if known != message.sender:
            self.rejected.append(
                "с адреса {} ({}) пришло чужое имя {}".format(
                    message.source_ip, known, message.sender
                )
            )
            return False
        return True

    def close(self) -> None:
        self.transport.close()


# ═══════════════════════════════════════════════════════════════════════
#  СТОРОНЫ: СТАНЦИЯ И БОРТ
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class DroneState:
    """Что станция знает про один борт."""

    requested_at: Optional[float] = None
    granted_at: Optional[float] = None
    landed_at: Optional[float] = None
    departed_at: Optional[float] = None


class StationChannel:
    """Сторона станции: свежие запросы бортов и выдача разрешений."""

    def __init__(
        self,
        link: Link,
        *,
        request_ttl: float = 3.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if request_ttl <= 0:
            raise ValueError("request_ttl должен быть больше нуля")
        self.link = link
        self.request_ttl = float(request_ttl)
        self.time_fn = time_fn
        self.states: Dict[str, DroneState] = {}
        self.log: List[Message] = []

    def _state(self, drone: str) -> DroneState:
        return self.states.setdefault(drone, DroneState())

    def pump(self, timeout: float = 0.0) -> List[Message]:
        """Разобрать входящие и обновить состояние бортов."""
        messages = self.link.pump(timeout)
        now = self.time_fn()
        for message in messages:
            state = self._state(message.sender)
            if message.kind == Kind.LANDING_REQUEST:
                state.requested_at = now
            elif message.kind == Kind.LANDED:
                state.landed_at = now
            elif message.kind == Kind.DEPARTED:
                state.departed_at = now
                state.requested_at = None
            elif message.kind == Kind.HELLO and message.sender in self.link.peers:
                # Ответ на проверку связи: борт увидит, что станция жива.
                # Незнакомое имя оставляем без ответа: кадровый цикл станции
                # не должен падать из-за постороннего трафика на порту.
                self.link.send(Kind.HELLO, target=message.sender, repeat=1)
            self.log.append(message)
        return messages

    def pending(self) -> Optional[str]:
        """Борт, который просит посадку прямо сейчас, или ``None``.

        Запрос протухает через ``request_ttl``: борт повторяет его, пока
        висит над станцией, поэтому замолчавший борт (улетел, упал, сел с
        пульта) не должен вечно держать станцию занятой.
        """
        now = self.time_fn()
        fresh = [
            (state.requested_at, drone)
            for drone, state in self.states.items()
            if state.requested_at is not None
            and now - state.requested_at <= self.request_ttl
        ]
        if not fresh:
            return None
        return max(fresh)[1]  # самый свежий запрос

    def grant(self, drone: str) -> None:
        self._state(drone).granted_at = self.time_fn()
        self.link.send(Kind.LANDING_GRANTED, target=drone)

    def deny(self, drone: str, reason: str) -> None:
        self.link.send(Kind.LANDING_DENIED, target=drone, data={"reason": reason})

    def has_landed(self, drone: str) -> bool:
        state = self.states.get(drone)
        return bool(state and state.landed_at is not None)

    def has_departed(self, drone: str) -> bool:
        state = self.states.get(drone)
        return bool(state and state.departed_at is not None)

    def forget(self, drone: str) -> None:
        """Забыть борт: следующая попытка посадки начинается с чистого листа."""
        self.states.pop(drone, None)


class DroneChannel:
    """Сторона борта: запросить посадку и доложить о посадке/взлёте."""

    def __init__(
        self,
        link: Link,
        *,
        station: str,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.link = link
        self.station = station
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn

    def request_landing(
        self,
        *,
        timeout: float = 30.0,
        interval: float = 0.5,
        should_stop: StopFn = lambda: False,
    ) -> Optional[Message]:
        """Просить посадку, пока не разрешат. Возвращает ответ станции.

        ``None`` — станция не ответила за ``timeout`` (или остановили извне).
        Отказ (``landing_denied``) возвращается тем же способом: решение, что
        с ним делать, принимает полётный сценарий, а не протокол.
        """
        if timeout <= 0 or interval <= 0:
            raise ValueError("timeout и interval должны быть больше нуля")
        deadline = self.time_fn() + timeout
        while not should_stop():
            self.link.send(Kind.LANDING_REQUEST, target=self.station)
            waited = 0.0
            while waited < interval:
                for message in self.link.pump(interval / 4.0):
                    if message.kind in (Kind.LANDING_GRANTED, Kind.LANDING_DENIED):
                        return message
                waited += interval / 4.0
                self.sleep_fn(0.0)
            if self.time_fn() >= deadline:
                return None
        return None

    def report_landed(self) -> None:
        self.link.send(Kind.LANDED, target=self.station)

    def report_departed(self) -> None:
        self.link.send(Kind.DEPARTED, target=self.station)

    def hello(self, *, timeout: float = 3.0) -> bool:
        """Проверка связи: станция жива и трафик между устройствами ходит."""
        self.link.send(Kind.HELLO, target=self.station)
        deadline = self.time_fn() + timeout
        while self.time_fn() < deadline:
            for message in self.link.pump(0.2):
                if message.kind == Kind.HELLO:
                    return True
        return False


# ═══════════════════════════════════════════════════════════════════════
#  РАЗБОР АРГУМЕНТОВ
# ═══════════════════════════════════════════════════════════════════════


def parse_peer(value: str) -> Tuple[str, str]:
    """``имя=адрес`` -> (``имя``, ``адрес``). Адрес — IP либо имя хоста."""
    name, separator, address = value.partition("=")
    name, address = name.strip(), address.strip()
    if not separator or not name or not address:
        raise argparse.ArgumentTypeError(
            "Адресат задаётся как имя=адрес, например uav1=192.168.0.11: " + value
        )
    try:
        ipaddress.ip_address(address)
    except ValueError:
        if any(character.isspace() for character in address):
            raise argparse.ArgumentTypeError(
                "Адрес не похож ни на IP, ни на имя хоста: " + address
            )
    return name, address


def peers_from_values(values: Optional[Sequence[Tuple[str, str]]]) -> Peers:
    peers: Peers = {}
    for name, address in values or ():
        if name in peers and peers[name] != address:
            raise ValueError(
                "Для {} задано два разных адреса: {} и {}".format(
                    name, peers[name], address
                )
            )
        peers[name] = address
    return peers


# ═══════════════════════════════════════════════════════════════════════
#  САМОТЕСТ
# ═══════════════════════════════════════════════════════════════════════


def _self_test() -> int:
    station_ip, drone_ip, stranger_ip = "10.0.0.1", "10.0.0.2", "10.0.0.9"

    def build(drop: Optional[Callable[[int], bool]] = None):
        network = LoopbackNetwork(drop=drop)
        station = Link(
            "station1",
            network.transport(station_ip),
            peers={"uav1": drone_ip},
        )
        drone = Link(
            "uav1",
            network.transport(drone_ip),
            peers={"station1": station_ip},
        )
        return network, station, drone

    # 1. Полный обмен: запрос -> грант -> посадка -> отлёт.
    clock = [100.0]
    network, station_link, drone_link = build()
    station = StationChannel(station_link, time_fn=lambda: clock[0])
    drone = DroneChannel(
        drone_link, station="station1", time_fn=lambda: clock[0],
        sleep_fn=lambda _seconds: None,
    )

    drone_link.send(Kind.LANDING_REQUEST, target="station1")
    station.pump()
    assert station.pending() == "uav1", "станция не увидела запрос борта"

    station.grant("uav1")
    answers = [m.kind for m in drone_link.pump()]
    assert answers == [Kind.LANDING_GRANTED], answers

    drone.report_landed()
    station.pump()
    assert station.has_landed("uav1")
    assert not station.has_departed("uav1")

    drone.report_departed()
    station.pump()
    assert station.has_departed("uav1")
    assert station.pending() is None, "после отлёта запрос должен быть снят"

    # 2. Повторы одного сообщения принимаются один раз.
    drone_link.send(Kind.HELLO, target="station1", repeat=5)
    hellos = [m for m in station_link.pump() if m.kind == Kind.HELLO]
    assert len(hellos) == 1, "дедуп не сработал: {}".format(len(hellos))

    # 3. Потеря части датаграмм: сообщение всё равно доходит.
    network, station_link, drone_link = build(drop=lambda index: index % 3 != 0)
    drone_link.send(Kind.LANDING_REQUEST, target="station1", repeat=3)
    assert len(station_link.pump()) == 1, "повторы не спасли от потери"
    assert network.lost > 0

    # 4. Чужой адрес и подмена имени отбрасываются.
    network, station_link, _ = build()
    stranger = Link("uav1", network.transport(stranger_ip), peers={"station1": station_ip})
    stranger.send(Kind.LANDING_REQUEST, target="station1")
    assert station_link.pump() == [], "принято сообщение с постороннего адреса"

    liar = Link("uav2", network.transport(drone_ip), peers={"station1": station_ip})
    liar.send(Kind.LANDING_REQUEST, target="station1")
    assert station_link.pump() == [], "принято чужое имя со знакомого адреса"
    assert station_link.rejected, "отказы должны попадать в журнал"

    # 4a. Нестрогий режим: hello от незнакомца не должен ронять цикл станции.
    network = LoopbackNetwork()
    open_link = Link("station1", network.transport(station_ip), peers={}, strict=False)
    outsider = Link("ghost", network.transport(stranger_ip), peers={"station1": station_ip})
    outsider.send(Kind.HELLO, target="station1")
    assert len(StationChannel(open_link).pump()) == 1

    # 5. Запрос протухает: замолчавший борт не держит станцию занятой.
    clock = [0.0]
    network, station_link, drone_link = build()
    station = StationChannel(station_link, request_ttl=3.0, time_fn=lambda: clock[0])
    drone_link.send(Kind.LANDING_REQUEST, target="station1")
    station.pump()
    assert station.pending() == "uav1"
    clock[0] += 5.0
    assert station.pending() is None, "устаревший запрос должен сниматься"

    # 6. Молчащая станция: борт уходит по тайм-ауту, а не висит вечно.
    clock = [0.0]
    network = LoopbackNetwork()
    lonely = Link("uav1", network.transport(drone_ip), peers={"station1": station_ip})
    silent = DroneChannel(
        lonely, station="station1", time_fn=lambda: clock[0],
        sleep_fn=lambda _seconds: clock.__setitem__(0, clock[0] + 0.2),
    )
    assert silent.request_landing(timeout=1.0, interval=0.4) is None

    # 7. Недоступный собеседник не роняет процесс.
    class DeadTransport(Transport):
        def send(self, payload: bytes, ip: str) -> bool:
            return False

        def receive(self, timeout: float) -> Optional[Tuple[bytes, str]]:
            return None

    dead = Link("station1", DeadTransport(), peers={"uav1": drone_ip})
    assert dead.send(Kind.LANDING_GRANTED, target="uav1") == 0

    # 8. Разбор аргументов.
    assert parse_peer("uav1=192.168.0.11") == ("uav1", "192.168.0.11")
    for bad in ("uav1", "=1.2.3.4", "uav1="):
        try:
            parse_peer(bad)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError("должно отвергаться: " + repr(bad))

    print("SELF-TEST: OK")
    return 0


def _loopback_test(port: int = DEFAULT_PORT) -> int:
    """Тот же обмен, но через настоящие UDP-сокеты на 127.0.0.1.

    Самотест выше проверяет протокол, а этот — что сокеты вообще открываются,
    датаграммы ходят и приём не блокируется. Оба узла на одной машине, поэтому
    порты соседние; в бою порт с обеих сторон один и тот же.
    """
    host = "127.0.0.1"
    station_link = Link(
        "station1",
        UdpTransport(port, bind_host=host, peer_port=port + 1),
        peers={"uav1": host},
    )
    drone_link = Link(
        "uav1",
        UdpTransport(port + 1, bind_host=host, peer_port=port),
        peers={"station1": host},
    )
    try:
        station = StationChannel(station_link)
        drone = DroneChannel(drone_link, station="station1")

        drone_link.send(Kind.LANDING_REQUEST, target="station1")
        station.pump(1.0)
        if station.pending() != "uav1":
            print("ОШИБКА: станция не получила запрос по сокету")
            return 1

        station.grant("uav1")
        answers = [message.kind for message in drone_link.pump(1.0)]
        if Kind.LANDING_GRANTED not in answers:
            print("ОШИБКА: борт не получил разрешение по сокету: {}".format(answers))
            return 1

        drone.report_departed()
        station.pump(1.0)
        if not station.has_departed("uav1"):
            print("ОШИБКА: станция не увидела отлёт")
            return 1
    finally:
        station_link.close()
        drone_link.close()

    print("LOOPBACK-TEST: OK (UDP {} <-> {})".format(port, port + 1))
    return 0


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Связь БВС<->зарядная станция по Wi-Fi (без сервера)"
    )
    parser.add_argument("--id", help="имя этого узла, например station1 или uav1")
    parser.add_argument(
        "--peer",
        action="append",
        type=parse_peer,
        metavar="ИМЯ=АДРЕС",
        help="собеседник и его IP (флаг повторяемый)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="UDP-порт, одинаковый с обеих сторон")
    parser.add_argument("--bind", default="0.0.0.0", help="локальный адрес приёма")
    parser.add_argument("--listen", action="store_true",
                        help="печатать входящие и отвечать на hello")
    parser.add_argument("--request", action="store_true",
                        help="сымитировать борт: просить посадку и ждать ответ")
    parser.add_argument("--grant", metavar="БОРТ",
                        help="сымитировать станцию: выдать разрешение борту")
    parser.add_argument("--hello", action="store_true",
                        help="проверить связь с собеседниками")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="сколько ждать ответа, с")
    parser.add_argument("--self-test", action="store_true",
                        help="проверить протокол без сети и железа")
    parser.add_argument("--loopback-test", action="store_true",
                        help="проверить настоящие UDP-сокеты на 127.0.0.1")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if args.loopback_test:
        return _loopback_test(args.port)
    if not args.id:
        parser.error("нужен --id (имя этого узла)")

    peers = peers_from_values(args.peer)
    if not peers:
        parser.error("нужен хотя бы один --peer ИМЯ=АДРЕС")

    link = Link(args.id, UdpTransport(args.port, bind_host=args.bind), peers=peers)
    print("узел {} на UDP:{}, адресаты: {}".format(
        args.id, args.port,
        ", ".join("{}={}".format(name, ip) for name, ip in sorted(peers.items()))))

    try:
        if args.hello:
            station = sorted(peers)[0]
            drone = DroneChannel(link, station=station)
            alive = drone.hello(timeout=min(args.timeout, 5.0))
            print("связь с {}: {}".format(station, "есть" if alive else "НЕТ ОТВЕТА"))
            if not alive:
                print("проверьте, что на роутере выключена изоляция клиентов "
                      "и что обе стороны используют порт {}".format(args.port))
            return 0 if alive else 1

        if args.grant:
            StationChannel(link).grant(args.grant)
            print("разрешение отправлено борту {}".format(args.grant))
            return 0

        if args.request:
            station = sorted(peers)[0]
            answer = DroneChannel(link, station=station).request_landing(
                timeout=args.timeout)
            if answer is None:
                print("станция {} не ответила за {:.0f}с".format(station, args.timeout))
                return 1
            print("ответ станции: " + answer.describe())
            return 0 if answer.kind == Kind.LANDING_GRANTED else 1

        if args.listen:
            channel = StationChannel(link)
            print("слушаем эфир, Ctrl+C — выход")
            while True:
                for message in channel.pump(0.5):
                    print(message.describe(), flush=True)
                while link.rejected:
                    print("отклонено: " + link.rejected.pop(0), flush=True)

        parser.print_help()
        return 0
    except KeyboardInterrupt:
        print("остановлено")
        return 0
    finally:
        link.close()


if __name__ == "__main__":
    raise SystemExit(main())
