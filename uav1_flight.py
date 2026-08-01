#!/usr/bin/env python3
"""БВС-1: взлёт, перелёт к своей зарядной станции по меткам и посадка на неё.

Заменяет прежнее решение, летавшее по глобальной локализации ``aruco_map``
(удалено; осталось в ветке ``main``):

* **навигация только по видимым ArUco-меткам** — телеметрия не используется
  вовсе, ни ``get_telemetry``, ни ``aruco_map``, ни TF. Место, высота и
  поворот считаются по кадру с ``main_camera/image_raw`` (см. докстринг
  ``lib/marker_nav.py``). Подход перенесён из проверенного в полёте
  ``fly_head.py`` (репозиторий Bl1tz2200/Snake);
* **все команды полёта — ``frame_id='body'`` и ровно одна ось за команду**
  (x, либо y, либо z). Единственная точка отправки — ``Pilot.hop``;
* **маршрут обходит деревья** высотой ~1 м, стоящие на стыках четырёх меток
  (``marker_nav.plan_route``);
* посадка на куб станции — пошаговый спуск с контролем
  ``/rangefinder/range`` и ручным дизармом: поведение штатного ``land()`` на
  приподнятой поверхности в документации Skyris не описано.

Курс намеренно **не удерживается**: камера смотрит вниз, а поворот поля в
кадре измеряется на каждом кадре заново, поэтому разворот дрона навигации не
мешает и лишних команд по yaw не нужно.

Запуск на дроне (подробности — в README, раздел про ветку UAV1)::

    python3 uav1_flight.py --probe                 # без моторов, проверка зрения
    python3 uav1_flight.py --map config/field_map.txt \
        --start-marker 48 --station-marker 5 --station-height 0.8
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import marker_nav as nav  # noqa: E402  (после sys.path выше)


class FlightAborted(RuntimeError):
    """Полёт прекращаем: команды не проходят или обстановка стала опасной."""


# ═══════════════════════════════════════════════════════════════════════
#  НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MissionConfig:
    """Всё, что подгоняется на площадке. Значения по умолчанию — из регламента
    и из ``fly_head.py``; дрон другой, поэтому калибровать заново."""

    map_path: str = "config/field_map.txt"
    start_marker_id: int = 48
    station_marker_id: int = 5

    # Рабочая высота, м. 2.0 при деревьях 1 м даёт метр запаса над кроной —
    # именно он держит дрон на маршруте, проходящем вплотную вдоль ствола
    # (см. tree_clearance). Потолок полётной зоны по регламенту — 4 м.
    alt: float = 2.0
    speed: float = 0.3  # горизонтальная скорость, м/с
    # Вертикаль. Здесь нет ни сетки, ни стен, поэтому осторожничать незачем:
    # ползучий взлёт только тратит время попытки и держит дрон в самой
    # неустойчивой зоне у земли (подсос от пола) дольше нужного.
    climb_speed: float = 0.5
    descent_speed: float = 0.2  # у куба, наоборот, спешить нельзя

    # ── Плавность хода ──────────────────────────────────────────────────
    # Резкость полёта задаётся не скоростью, а ДЛИНОЙ одной команды.
    # ``navigate(frame_id='body')`` мгновенно ставит цель на step_length
    # вперёд, и контроллер PX4 отрабатывает эту ошибку положения наклоном:
    # чем длиннее шаг, тем больше наклон и тем сильнее разгон. Поэтому
    # уменьшение ``speed`` само по себе рывки не убирает — нужно уменьшать шаг.
    step_length: float = 0.3  # максимум одного горизонтального шага, м
    climb_step: float = 1.0  # максимум одной вертикальной команды, м
    # Горизонтальный шаг отдаётся не одной командой, а ЛЕСЕНКОЙ микрокоманд по
    # micro_step через micro_pause. Смысл: цель всё время держится в нескольких
    # сантиметрах впереди дрона, поэтому ошибка положения — а значит и наклон,
    # которым её отрабатывает PX4, — остаётся малой на всём перелёте. Так
    # скорость задаётся отношением micro_step/micro_pause (здесь 0.2 м/с) и не
    # зависит от того, как прошивка понимает аргумент speed.
    micro_step: float = 0.06
    micro_pause: float = 0.3
    # После каждого шага цель ставится в текущую точку (navigate(0,0,0,body)):
    # это гасит остаточную скорость, и следующий кадр снимается с неподвижного
    # дрона. Без торможения команды складываются с инерцией — дрон
    # разгоняется от шага к шагу и проскакивает цель.
    brake_pause: float = 0.6  # пауза на гашение скорости после шага, с
    settle_after_hop: bool = True  # ждать спокойного кадра после каждого шага

    marker_size: float = 0.33  # сторона метки, м — по ней кадр переводится в метры
    aruco_dict: str = "DICT_4X4_50"
    gain: float = 0.5  # какую долю промаха отрабатывать за раз
    tol: float = 0.08  # «над меткой»: доля диагонали кадра
    tries: int = 16  # попыток довести дрон до одного узла (шаг короткий — их больше)
    # Запас к паузе перелёта, с. Меньше, чем был раньше, потому что после шага
    # теперь идёт явное торможение и ожидание спокойного кадра: если дрон не
    # успел дойти, торможение просто остановит его раньше, а недоделанное
    # доберёт следующий шаг — контур замкнут по меткам.
    hop_pad: float = 0.5
    hop_max_steps: float = 1.5  # длиннее этого (в шагах решётки) команд не бывает
    hop_dead_zone: float = 0.05  # мельче — не команда, а шум опознания метки
    alt_fix: float = 0.25  # предел поправки высоты за одну команду, м
    alt_dead_zone: float = 0.07

    look_up: float = 0.3  # подъём «осмотреться», если меток не видно, м
    climb_max: float = 1.0  # выше рабочей высоты вслепую не подниматься, м
    blind_frames: int = 2  # столько пустых кадров подряд — и поднимаемся
    settle_frames: int = 3  # столько кадров метка должна стоять на месте
    settle_drift: float = 0.15  # «стоит на месте»: сдвиг меньше этой доли метки
    settle_tries: int = 20
    frame_pause: float = 0.3  # пауза между кадрами в циклах ожидания, с

    blind_tolerance: float = 0.15  # доводка на закрытую метку по карте, м
    blind_confirm: int = 3  # столько кадров подряд остаток должен быть мал
    # Метки, физически закрытые объектами: площадка «Н» на старте, кубы
    # зарядных станций, груз. Камера их не увидит никогда, поэтому вставать
    # над ними можно только счётом по карте от соседей, а сверять положение —
    # по координатам, а не по «вижу нужную метку».
    blind_markers: Tuple[int, ...] = (0, 5, 37, 48)
    start_tolerance: float = 0.7  # допуск сверки места со стартовой меткой, м

    station_height: float = 0.8  # высота куба станции, м
    descent_step: float = 0.15
    touchdown_range: float = 0.15  # дальномер ближе этого — дрон стоит
    rangefinder_timeout: float = 1.0

    tree_groups: Tuple[Tuple[int, ...], ...] = nav.DEFAULT_TREE_GROUPS
    tree_height: float = 1.0
    # Зазор для ПЛАНИРОВАНИЯ маршрута. 0.5 м — ровно половина ячейки, то есть
    # проход вплотную вдоль ствола (ребро решётки, соседнее с деревом, отстоит
    # от него ровно на 0.5 м). Такой маршрут короче на два перегона; запас по
    # вертикали при этом обеспечивает alt - tree_height.
    tree_clearance: float = 0.5
    # Зазор для СТОРОЖА: ближе этого к стволу лететь нельзя ни при каких
    # обстоятельствах — команда отменяется, миссия прерывается. Он заметно
    # меньше планового не по небрежности: команда идёт по осям корпуса, а
    # маршрут проложен по осям поля, поэтому фактический перегон у
    # развёрнутого дрона всегда немного отходит от идеального ребра решётки.
    # Если сторож требует того же, что планировщик, эти законные отклонения
    # он принимает за опасность и сажает дрон посреди исправного полёта.
    tree_guard_clearance: float = 0.3
    tree_vertical_margin: float = 0.4
    geofence_margin: float = 0.5

    nav_fail_max: int = 5  # столько отказов navigate подряд — и полёт прекращаем
    frame_timeout: float = 2.0
    camera_topic: str = "main_camera/image_raw"
    rangefinder_topic: str = "/rangefinder/range"
    node_name: str = "uav1_flight"
    quiet: bool = False

    def hop_limit(self, grid_step: float) -> float:
        """Длина одной горизонтальной команды: короткий шаг ради плавности,
        но не длиннее аварийного предела в шагах решётки."""
        return min(self.step_length, self.hop_max_steps * grid_step)


# ═══════════════════════════════════════════════════════════════════════
#  ROS: единственное место с rospy/technic/mavros/cv_bridge
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RosLink:
    """Прокси и читалки, которыми сценарий разговаривает с дроном."""

    navigate: Callable[..., Any]
    land: Callable[[], Any]
    arming: Callable[[bool], Any]
    read_frame: Callable[[float], Any]
    read_range: Callable[[float], float]


def init_ros(config: MissionConfig, *, init_node: bool = True) -> RosLink:
    """Поднять ROS-узел и вернуть прокси. Импорт ROS спрятан здесь намеренно."""
    try:
        import rospy
        from technic import srv
        from std_srvs.srv import Trigger
        from mavros_msgs.srv import CommandBool
        from sensor_msgs.msg import Image, Range
        from cv_bridge import CvBridge
    except ImportError as exc:
        raise RuntimeError(
            "Нужны rospy, technic, std_srvs, mavros_msgs, sensor_msgs и cv_bridge. "
            "Выполните source /opt/ros/noetic/setup.bash и source своего catkin_ws"
        ) from exc

    if init_node:
        rospy.init_node(config.node_name)

    navigate = rospy.ServiceProxy("navigate", srv.Navigate)
    land = rospy.ServiceProxy("land", Trigger)
    arming_proxy = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
    bridge = CvBridge()

    def read_frame(timeout: float) -> Any:
        message = rospy.wait_for_message(config.camera_topic, Image, timeout=timeout)
        return bridge.imgmsg_to_cv2(message, "bgr8")

    def read_range(timeout: float) -> float:
        message = rospy.wait_for_message(config.rangefinder_topic, Range, timeout=timeout)
        return float(message.range)

    return RosLink(
        navigate=navigate,
        land=land,
        arming=lambda value: arming_proxy(value),
        read_frame=read_frame,
        read_range=read_range,
    )


# ═══════════════════════════════════════════════════════════════════════
#  ПИЛОТ
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Sight:
    """Что дрон видит на одном кадре и что из этого следует."""

    seen: Dict[int, nav.Marker]
    centre: Tuple[float, float]
    diagonal: float
    base: Optional[nav.Marker] = None
    frame_yaw: Optional[float] = None
    position: Optional[Tuple[float, float]] = None
    altitude: float = 0.0


class Pilot:
    """Сценарий полёта поверх кадра камеры. Команды — только через ``hop``."""

    def __init__(
        self,
        config: MissionConfig,
        ros: RosLink,
        field: Dict[int, Tuple[float, float]],
        detector: Any,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.ros = ros
        self.field = field
        self.detector = detector
        self.sleep = sleep_fn
        self.grid_step = nav.field_step(field)
        self.trees = nav.tree_positions(field, config.tree_groups)
        self.fence = nav.Geofence.from_field(field, margin=config.geofence_margin)

        self.side_ref = 0.0  # сторона метки на рабочей высоте (эталон высоты)
        # Последний ИЗМЕРЕННЫЙ поворот поля в кадре. None означает «ещё ни разу
        # не измеряли»: по одной метке в кадре поворот не определить, а
        # подставить вместо него ноль — значит развернуть все расчёты места на
        # неизвестный угол. Пока здесь None, места дрон не знает и не летит.
        self.yaw_ref: Optional[float] = None
        self.blind_up = 0.0  # набрано вслепую сверх рабочей высоты, м
        self.after_blind_climb = False  # см. hold_altitude: запрет сразу отдать высоту
        self.nav_fails = 0
        self.armed = False

    # ---------------------------------------------------------------- лог
    def say(self, text: str) -> None:
        if not self.config.quiet:
            print(text, flush=True)

    # ------------------------------------------------------------ команда
    def hop(self, axis: str, value: float, *, speed: Optional[float] = None,
            auto_arm: bool = False, brake: bool = True,
            ramp: Optional[bool] = None) -> None:
        """ЕДИНСТВЕННОЕ место, откуда уходит navigate(): ровно одна ось.

        Две другие координаты всегда нули, ``frame_id='body'``. ``yaw`` не
        передаётся вовсе: штатное значение сервиса (0) в кадре ``body``
        означает «не менять курс», а это ровно то, что нужно.

        Горизонтальный шаг отдаётся **лесенкой микрокоманд** (``ramp``): цель
        каждый раз переносится всего на ``micro_step`` вперёд от текущего
        места, поэтому ошибка положения, которую PX4 отрабатывает наклоном,
        всё время мала. Одна команда на весь шаг — это скачок ошибки на всю
        его длину, и именно из-за него дрон резко кренился и проскакивал цель.
        Скорость при этом задаётся не аргументом ``speed`` (прошивка
        экспериментальная, полагаться на него нельзя), а самим темпом
        ``micro_step / micro_pause``.

        По вертикали лесенка не нужна: там нет ни сетки, ни стен, а отрыв от
        земли, наоборот, должен быть решительным.
        """
        if axis not in ("x", "y", "z"):
            raise ValueError("Ось перелёта может быть только x, y или z: " + repr(axis))
        speed = speed if speed is not None else self.config.speed
        if ramp is None:
            ramp = axis in ("x", "y") and self.config.micro_step > 0.0

        remaining = float(value)
        first = True
        while abs(remaining) > 1e-6:
            piece = remaining
            if ramp:
                piece = math.copysign(min(self.config.micro_step, abs(remaining)), remaining)
            self._navigate_once(axis, piece, speed, auto_arm and first)
            first = False
            remaining -= piece
            if ramp and abs(remaining) > 1e-6:
                self.sleep(self.config.micro_pause)

        # Хвост: последней микрокоманде тоже нужно время долететь, а одиночной
        # команде — весь путь целиком.
        if ramp:
            self.sleep(self.config.micro_pause + self.config.hop_pad)
        else:
            self.sleep(abs(float(value)) / max(speed, 0.05) + self.config.hop_pad)

        # Торможение: цель переносится в текущую точку, остаточная скорость
        # гасится, и следующая команда начинается с неподвижного дрона. Без
        # этого шаги складываются с инерцией — дрон разгоняется от шага к шагу,
        # проскакивает цель и «плывёт» дальше, чем ему было велено.
        if brake:
            self._stop_moving()
            self.sleep(self.config.brake_pause)

    def _navigate_once(self, axis: str, value: float, speed: float,
                       auto_arm: bool) -> None:
        """Одна команда navigate() со счётчиком отказов."""
        request = {"x": 0.0, "y": 0.0, "z": 0.0, "speed": float(speed),
                   "frame_id": "body", "auto_arm": bool(auto_arm)}
        request[axis] = float(value)

        try:
            response = self.ros.navigate(**request)
            refused = response is not None and not getattr(response, "success", True)
            why = getattr(response, "message", "")
        except Exception as exc:  # noqa: BLE001 - для дрона отказ и исключение одно и то же
            refused, why = True, "исключение: {}".format(exc)

        if refused:
            self.nav_fails += 1
            self.say("          navigate отказал ({}/{}): {}".format(
                self.nav_fails, self.config.nav_fail_max, why))
            if self.nav_fails >= self.config.nav_fail_max:
                raise FlightAborted(
                    "navigate отказал {} раз подряд".format(self.nav_fails))
        else:
            self.nav_fails = 0
            if auto_arm:
                self.armed = True

    def _stop_moving(self, *, speed: Optional[float] = None) -> None:
        """``navigate(0, 0, 0, frame_id='body')`` — «цель там, где я сейчас»."""
        try:
            self.ros.navigate(x=0.0, y=0.0, z=0.0,
                              speed=float(speed if speed is not None else self.config.speed),
                              frame_id="body", auto_arm=False)
        except Exception as exc:  # noqa: BLE001 - гасить движение пробуем до конца
            self.say("          стоп не прошёл: {}".format(exc))

    def hold(self) -> None:
        """Аварийный стоп: «цель — там, где я сейчас». Все три координаты нули."""
        self.say("          стоп: navigate(0, 0, 0, frame_id='body')")
        self._stop_moving(speed=self.config.descent_speed)

    # -------------------------------------------------------------- зрение
    def look(self) -> Optional[Sight]:
        """Разобранный кадр или None, если камера ничего годного не отдала."""
        try:
            image = self.ros.read_frame(self.config.frame_timeout)
        except Exception as exc:  # noqa: BLE001 - следующий кадр может быть хорошим
            self.say("кадр не получен: {}".format(exc))
            return None
        if image is None or getattr(image, "ndim", 0) != 3 or getattr(image, "size", 0) == 0:
            return None

        height, width = image.shape[:2]
        centre = (width / 2.0, height / 2.0)
        seen = self.detector.detect(image)
        sight = Sight(seen=seen, centre=centre, diagonal=math.hypot(width, height))
        if not seen:
            return sight

        sight.base = nav.nearest_to_centre(seen, centre)
        # Поворот поля меряется по ПАРЕ меток; одной метки для этого мало.
        # Если пары в кадре нет, берём последний измеренный угол — курс дрон
        # нигде не меняет (команд по yaw в коде нет), поэтому он не «протухает».
        # А вот если не измеряли ещё ни разу, места мы не знаем — так и
        # оставляем, чтобы не полететь по развёрнутой на неизвестный угол карте.
        measured_yaw = nav.frame_yaw_from_pairs(seen, self.field, self.config.marker_size)
        if measured_yaw is not None:
            self.yaw_ref = measured_yaw
        sight.frame_yaw = self.yaw_ref
        if sight.frame_yaw is not None and sight.base is not None and sight.base.mid in self.field:
            sight.position = nav.estimate_position(
                sight.base, self.field, self.yaw_ref, centre, self.config.marker_size
            )
            sight.altitude = nav.alt_by_side(sight.base.side, self.side_ref, self.config.alt)
            # Пока метка видна, превышение над рабочей высотой — это измеренная
            # величина, а не счисленная: слепой подъём обнуляется честным замером.
            if sight.altitude:
                self.blind_up = max(0.0, sight.altitude - self.config.alt)
        return sight

    def settle(self, *, frames: Optional[int] = None, tries: Optional[int] = None,
               verbose: bool = True) -> float:
        """Дождаться спокойного висения; вернуть сторону метки на тихом кадре.

        Признак того, что дрон перестал качать, — метка под ним стоит в кадре:
        между соседними кадрами её центр и размер почти не меняются. Доводка
        по прыгающей метке раскачивает дрон сильнее, чем исправляет, поэтому
        команд до успокоения не отдаём вовсе.

        ``frames``/``tries`` позволяют коротко перевести дух после каждого
        шага (два спокойных кадра), не тратя полное ожидание, которое нужно
        только на калибровке и перед спуском на куб.
        """
        frames = self.config.settle_frames if frames is None else frames
        tries = self.config.settle_tries if tries is None else tries
        previous: Optional[Tuple[int, Tuple[float, float, float]]] = None
        calm = 0
        side = 0.0
        for _ in range(tries):
            sight = self.look()
            if sight is None or sight.base is None:
                previous, calm = None, 0
                self.sleep(self.config.frame_pause)
                continue

            base = sight.base
            side = base.side
            if previous is not None and previous[0] == base.mid:
                px, py, pside = previous[1]
                drift = math.hypot(base.x - px, base.y - py) / side
                zoom = abs(side - pside) / side
                if drift < self.config.settle_drift and zoom < self.config.settle_drift:
                    calm += 1
                    if calm >= frames:
                        if verbose:
                            self.say("          висим спокойно (метка {}, {:.0f} px)".format(
                                base.mid, side))
                        return side
                else:
                    calm = 0
            previous = (base.mid, (base.x, base.y, side))
            self.sleep(self.config.frame_pause)

        if verbose:
            self.say("          успокоиться не вышло — летим как есть")
        return side

    # -------------------------------------------------------------- сторожа
    def guard(self, sight: Sight) -> None:
        """Проверка обстановки по расчётному месту: геозона и деревья."""
        if sight.position is None:
            return
        x, y = sight.position
        if not self.fence.contains(x, y):
            raise FlightAborted(
                "дрон вне поля: ({:.2f}, {:.2f}), разрешено {}".format(
                    x, y, self.fence.describe()))
        tree = nav.tree_hit(x, y, self.trees, self.config.tree_guard_clearance)
        if tree is not None:
            raise FlightAborted(
                "дрон вплотную к дереву ({:.1f}, {:.1f}) — расходимся".format(*tree))

    def hop_danger(self, sight: Sight, hop: nav.Hop) -> Optional[str]:
        """Чем опасен этот перелёт, или None, если он безопасен.

        Проверяется весь отрезок, а не только его конец: команда идёт по осям
        корпуса, а маршрут проложен по осям поля, поэтому у развёрнутого дрона
        перегон получается косым и может пройти мимо дерева, «промахнувшись»
        концом.
        """
        if sight.position is None or sight.frame_yaw is None:
            return None
        forward = hop.value if hop.axis == "x" else 0.0
        left = hop.value if hop.axis == "y" else 0.0
        dx, dy = nav.body_to_field(forward, left, sight.frame_yaw)
        end = (sight.position[0] + dx, sight.position[1] + dy)
        if not self.fence.contains(*end):
            return "вывел бы за поле: ({:.2f}, {:.2f})".format(*end)
        if not nav.segment_is_clear(sight.position, end, self.trees,
                                    self.config.tree_guard_clearance):
            return "прошёл бы вплотную к дереву"
        return None

    def plan_hop(self, sight: Sight, forward: float, left: float,
                 limit: float) -> Optional[nav.Hop]:
        """Выбрать одну безопасную команду из вариантов. None — двигаться незачем."""
        variants = nav.hop_candidates(
            forward, left, limit=limit, dead_zone=self.config.hop_dead_zone)
        if not variants:
            return None
        reasons = []
        for hop in variants:
            danger = self.hop_danger(sight, hop)
            if danger is None:
                return hop
            reasons.append(danger)
        raise FlightAborted(
            "безопасного перелёта нет: " + "; ".join(sorted(set(reasons))))

    # ---------------------------------------------------------- поддержание
    def climb(self) -> None:
        """Подняться осмотреться, когда меток не видно, — но не выше предела.

        Без метки высоту проверить нечем, поэтому слепой подъём считаем сами и
        вовремя останавливаемся: иначе дрон, потерявший поле, набирал бы высоту
        до самого потолка зоны.
        """
        if self.blind_up + self.config.look_up > self.config.climb_max:
            self.say("          выше не поднимаемся: набрано {:.1f} м вслепую".format(
                self.blind_up))
            return
        self.blind_up += self.config.look_up
        self.after_blind_climb = True
        self.say("          меток не видно — поднимаемся осмотреться (+{:.1f} м)".format(
            self.blind_up))
        self.hop("z", self.config.look_up, speed=self.config.climb_speed)

    def hold_altitude(self, sight: Sight) -> bool:
        """Вернуть рабочую высоту отдельной командой. True — команда отдана.

        Нужна потому, что ``navigate(frame_id='body', z=0)`` держит не высоту,
        а «сколько было в момент команды»: на разгонах дрон подвсплывает, и
        ошибка копится только вверх. Метка в кадре — единственный доступный
        здесь измеритель высоты.

        Сразу после слепого подъёма высота не отдаётся: поднимались как раз
        затем, чтобы увидеть метки, и немедленный спуск снова их потерял бы —
        дрон завис бы в качелях «поднялся-опустился», не двигаясь к цели.
        Сначала перелёт, высота — следующим кадром.
        """
        if sight.base is None or self.side_ref <= 1.0 or self.after_blind_climb:
            return False
        correction = nav.hold_alt(
            sight.base.side, self.side_ref, self.config.alt,
            limit=self.config.alt_fix, dead_zone=self.config.alt_dead_zone,
        )
        if correction == 0.0:
            return False
        self.say("          высота {:.2f} м -> {:.2f} м ({:+.2f})".format(
            sight.altitude, self.config.alt, correction))
        self.hop("z", correction, speed=self.config.climb_speed)
        return True

    # ------------------------------------------------------------- перелёт
    def goto(self, target_id: int, *, blind: bool = False) -> bool:
        """Довести дрон до метки ``target_id``. True — встали над ней.

        Метка видна — наводимся по кадру. Не видна (ушла за край кадра или
        закрыта кубом станции) — считаем вектор по карте от той метки, которую
        видно, и добавляем её собственное смещение от центра кадра. ``blind``
        включает второй критерий успеха для закрытой метки: расчётный остаток
        мал несколько кадров подряд.
        """
        target_xy = nav.marker_xy(self.field, target_id)
        limit = self.config.hop_limit(self.grid_step)
        empty_frames = 0
        confirmations = 0
        moves = 0

        # Попыток считаем сами перелёты к цели: поправки высоты и кадры без
        # меток не должны съедать лимит. Общее число проходов всё равно
        # ограничено, чтобы цикл не крутился вечно на пустом кадре.
        for _ in range(self.config.tries * 4):
            if moves >= self.config.tries:
                break
            sight = self.look()
            if sight is None:
                self.sleep(self.config.frame_pause)
                continue
            # Меток нет вовсе или их слишком мало, чтобы знать своё место
            # (поворот поля меряется по паре меток) — лечится одним и тем же:
            # подняться и расширить обзор.
            if sight.base is None or sight.position is None:
                empty_frames += 1
                if empty_frames >= self.config.blind_frames:
                    self.say("          в кадре {} — места не определить".format(
                        "нет меток" if not sight.seen else "одна метка"))
                    self.climb()
                    empty_frames = 0
                else:
                    self.sleep(self.config.frame_pause)
                continue
            empty_frames = 0
            self.guard(sight)
            self.report(target_id, sight)

            if self.hold_altitude(sight):
                continue  # высота — отдельная команда, дальше по свежему кадру

            base = sight.base
            scale = nav.metres_per_pixel(base.side, self.config.marker_size)
            if target_id in sight.seen:
                marker = sight.seen[target_id]
                offset = math.hypot(marker.x - sight.centre[0], marker.y - sight.centre[1])
                if offset <= self.config.tol * sight.diagonal:
                    self.say("          над меткой {}".format(target_id))
                    return True
                forward, left = nav.frame_vector(marker, sight.centre, scale)
                forward, left = forward * self.config.gain, left * self.config.gain
            else:
                base_xy = nav.marker_xy(self.field, base.mid)
                by_map = nav.field_to_body(
                    target_xy[0] - base_xy[0], target_xy[1] - base_xy[1],
                    sight.frame_yaw or 0.0,
                )
                by_frame = nav.frame_vector(base, sight.centre, scale)
                forward, left = by_map[0] + by_frame[0], by_map[1] + by_frame[1]
                if blind and math.hypot(forward, left) <= self.config.blind_tolerance:
                    confirmations += 1
                    self.say("          над закрытой меткой {} (остаток {:.2f} м, {}/{})".format(
                        target_id, math.hypot(forward, left), confirmations,
                        self.config.blind_confirm))
                    if confirmations >= self.config.blind_confirm:
                        return True
                    self.sleep(self.config.frame_pause)
                    continue
                confirmations = 0
                forward, left = forward * self.config.gain, left * self.config.gain

            hop = self.plan_hop(sight, forward, left, limit)
            if hop is None:
                if target_id in sight.seen:
                    self.say("          над меткой {} (остаток в мёртвой зоне)".format(
                        target_id))
                    return True
                self.sleep(self.config.frame_pause)
                continue
            raw = forward if hop.axis == "x" else left
            if abs(raw) > abs(hop.value) + 1e-6:
                self.say("          перелёт {:.2f} м урезан до {:.2f} м по оси {}".format(
                    abs(raw), abs(hop.value), hop.axis))
            self.hop(hop.axis, hop.value)
            if self.config.settle_after_hop:
                # Следующий кадр должен сниматься с неподвижного дрона: иначе
                # поправка считается по смазанной картинке и уходит в раскачку.
                self.settle(frames=2, tries=6, verbose=False)
            self.after_blind_climb = False
            moves += 1

        self.say("          узел {} пропущен".format(target_id))
        return False

    def report(self, target_id: int, sight: Sight) -> None:
        ids = " ".join(str(mid) for mid in sorted(sight.seen)) if sight.seen else "не видно"
        where = ("({:.2f}, {:.2f})".format(*sight.position)
                 if sight.position is not None else "?")
        height = " | h≈{:.2f} м".format(sight.altitude) if sight.altitude else ""
        self.say("цель {:2d} | метки: {} | место {}{}".format(
            target_id, ids, where, height))

    # -------------------------------------------------------------- взлёт
    def takeoff(self) -> None:
        """Вертикальный взлёт на рабочую высоту и калибровка по спокойному кадру."""
        # Взлёт тоже порциями: команда «сразу на 2 м» — это скачок ошибки
        # положения на два метра, который PX4 отрабатывает рывком вверх.
        self.say("взлёт на {:.2f} м шагами по {:.2f} м (frame_id='body')".format(
            self.config.alt, self.config.climb_step))
        remaining = self.config.alt
        first = True
        while remaining > 1e-3:
            step = min(self.config.climb_step, remaining)
            self.hop("z", step, speed=self.config.climb_speed, auto_arm=first)
            self.armed = True
            first = False
            remaining -= step

        side = self.settle()
        if side <= 1.0:
            raise FlightAborted(
                "после взлёта не видно ни одной метки — лететь по карте не от чего")
        # Эталон высоты: «как выглядит поле с рабочей высоты». Дальше вся
        # высота считается относительно него.
        self.side_ref = side
        self.blind_up = 0.0
        self.say(">>> ВЫСОТА {:.2f} м = метка {:.0f} px в кадре (эталон)".format(
            self.config.alt, side))

        # Место определяется по паре меток: одной мало (по ней не измерить
        # поворот поля). В углу поля, где стартовая метка закрыта площадкой,
        # пары в кадре может не оказаться — тогда поднимаемся и расширяем обзор.
        sight = self.look()
        while sight is None or sight.position is None:
            self.say("          места не определить — нужны две метки в кадре")
            before = self.blind_up
            self.climb()
            if self.blind_up == before:  # выше подниматься уже нельзя
                raise FlightAborted(
                    "после взлёта в кадре меньше двух меток — поворот поля не "
                    "измерить, лететь по карте нельзя")
            sight = self.look()

        # Стартовую метку дрон НЕ увидит: её закрывает площадка «Н», как куб
        # закрывает метку станции. Поэтому сверяем не «что под дроном», а
        # расчётное место по окрестным меткам с координатами стартовой метки
        # по карте: это ловит и не ту раскладку, и не ту площадку, и грубую
        # ошибку опознания, но не требует видеть то, чего не видно.
        start_xy = nav.marker_xy(self.field, self.config.start_marker_id)
        error = math.hypot(sight.position[0] - start_xy[0], sight.position[1] - start_xy[1])
        self.say(
            ">>> ПОСЛЕ ВЗЛЁТА: место ({:.2f}, {:.2f}), старт {} в ({:.2f}, {:.2f}), "
            "расхождение {:.2f} м | видно: {}".format(
                sight.position[0], sight.position[1], self.config.start_marker_id,
                start_xy[0], start_xy[1], error,
                " ".join(str(mid) for mid in sorted(sight.seen))))
        if error > self.config.start_tolerance:
            raise FlightAborted(
                "после взлёта дрон в ({:.2f}, {:.2f}), а стартовая метка {} — "
                "в ({:.2f}, {:.2f}): расхождение {:.2f} м больше "
                "--start-tolerance {:.2f} м".format(
                    sight.position[0], sight.position[1], self.config.start_marker_id,
                    start_xy[0], start_xy[1], error, self.config.start_tolerance))

        measured = nav.measured_grid_step(sight.seen, self.field, self.config.marker_size)
        if measured:
            self.say(">>> ШАГ РЕШЁТКИ: {:.2f} м (по карте {:.2f} м)".format(
                measured, self.grid_step))
            if abs(measured - self.grid_step) > 0.3:
                self.say("!!! ПРОВЕРЬТЕ --marker-size: замер шага сильно расходится с картой")

    # ------------------------------------------------------------- посадка
    def descend_onto_station(self) -> bool:
        """Спуск на куб станции по дальномеру и дизарм. True — сели.

        Спуск строго вертикальный: боковое место уже зафиксировано, а над
        кубом метки закрыты и корректировать по ним нечего. Под дроном должен
        оказаться куб — это видно по дальномеру, который показывает высоту
        минус высоту куба.
        """
        floor_range = self.read_range()
        if floor_range is not None:
            expected = self.config.alt - self.config.station_height
            if floor_range <= expected + self.config.station_height * 0.4:
                self.say("          дальномер {:.2f} м при высоте {:.2f} м — под дроном куб".format(
                    floor_range, self.config.alt))
            else:
                self.say("!!! дальномер {:.2f} м — похоже, под дроном пол, а не куб "
                         "(ожидалось около {:.2f} м)".format(floor_range, expected))

        max_steps = int(self.config.alt / max(self.config.descent_step, 0.01)) + 4
        for step in range(1, max_steps + 1):
            self.hop("z", -self.config.descent_step, speed=self.config.descent_speed)
            distance = self.read_range()
            self.say("          спуск {}: дальномер {}".format(
                step, "{:.2f} м".format(distance) if distance is not None else "нет данных"))
            if distance is not None and distance <= self.config.touchdown_range:
                self.say("          касание — выключаем моторы")
                self.disarm()
                return True
        return False

    def read_range(self) -> Optional[float]:
        try:
            return self.ros.read_range(self.config.rangefinder_timeout)
        except Exception as exc:  # noqa: BLE001 - спуск продолжается и без одного замера
            self.say("          дальномер молчит: {}".format(exc))
            return None

    def disarm(self) -> None:
        """Выключить моторы. Если сервиса армирования на этой сборке нет —
        дожать штатной посадкой, а не оставлять дрон работающим на кубе."""
        try:
            self.ros.arming(False)
            self.armed = False
            return
        except Exception as exc:  # noqa: BLE001
            self.say("!!! дизарм не прошёл ({}) — дожимаем посадкой".format(exc))
        self.land()

    def land(self, *, tries: int = 3) -> None:
        """Обычная посадка с повтором: одиночный ``land()`` — слабая гарантия."""
        for attempt in range(1, tries + 1):
            try:
                response = self.ros.land()
            except Exception as exc:  # noqa: BLE001
                self.say("ПОСАДКА: попытка {} сорвалась — {}".format(attempt, exc))
                self.sleep(1.0)
                continue
            self.say("ПОСАДКА: попытка {} — {} {}".format(
                attempt, getattr(response, "success", "?"), getattr(response, "message", "")))
            if response is None or getattr(response, "success", True):
                self.sleep(5.0)
                return
            self.sleep(1.0)
        self.say("!" * 54)
        self.say("!!! ПОСАДКА НЕ ПОДТВЕРЖДЕНА — САЖАЙТЕ С ПУЛЬТА НЕМЕДЛЕННО !!!")
        self.say("!" * 54)


# ═══════════════════════════════════════════════════════════════════════
#  МИССИЯ
# ═══════════════════════════════════════════════════════════════════════


def run_mission(pilot: Pilot) -> bool:
    """Взлёт -> маршрут по меткам в обход деревьев -> посадка на куб станции."""
    config = pilot.config
    route = nav.plan_route(
        pilot.field,
        config.start_marker_id,
        config.station_marker_id,
        trees=pilot.trees,
        clearance=config.tree_clearance,
    )
    pilot.say("деревья: " + (", ".join(
        "({:.1f}, {:.1f})".format(x, y) for x, y in pilot.trees) or "нет"))
    pilot.say("маршрут {} -> {}: {}".format(
        config.start_marker_id, config.station_marker_id,
        " -> ".join(str(mid) for mid in route)))
    if config.alt < config.tree_height + config.tree_vertical_margin:
        raise FlightAborted(
            "рабочая высота {:.2f} м опасно близка к деревьям {:.2f} м — "
            "поднимите --alt".format(config.alt, config.tree_height))
    if config.tree_guard_clearance > config.tree_clearance:
        # Иначе сторож запретит ровно тот маршрут, который сам же планировщик
        # и построил, — дрон сядет посреди исправного полёта.
        raise FlightAborted(
            "--tree-guard-clearance ({:.2f} м) больше --tree-clearance ({:.2f} м): "
            "сторож запретит спланированный маршрут".format(
                config.tree_guard_clearance, config.tree_clearance))

    pilot.takeoff()

    for mid in route[:-1]:
        # Узел, закрытый объектом (площадка, куб, груз), камера не увидит —
        # над ним встаём счётом по карте от соседей, иначе дрон впустую
        # выработал бы все попытки, ожидая метку, которой в кадре не будет.
        pilot.goto(mid, blind=mid in config.blind_markers)

    station = route[-1] if route else config.station_marker_id
    pilot.say("подход к станции (метка {} закрыта кубом, наводимся по карте)".format(station))
    if not pilot.goto(station, blind=True):
        raise FlightAborted("над станцией встать не удалось — на куб не садимся")
    pilot.settle()

    if pilot.descend_onto_station():
        pilot.say(">>> БВС-1 НА ЗАРЯДНОЙ СТАНЦИИ, МОТОРЫ ВЫКЛЮЧЕНЫ")
        return True

    pilot.say("!!! касание не подтверждено — садимся штатно")
    pilot.hold()
    pilot.land()
    return False


def run_probe(pilot: Pilot, *, cycles: int = 0) -> None:
    """Проверка зрения без моторов: что видно, где мы и каким будет маршрут."""
    config = pilot.config
    route = nav.plan_route(
        pilot.field, config.start_marker_id, config.station_marker_id,
        trees=pilot.trees, clearance=config.tree_clearance,
    )
    print("деревья: " + ", ".join("({:.1f}, {:.1f})".format(x, y) for x, y in pilot.trees))
    print("геозона: " + pilot.fence.describe())
    print("маршрут {} -> {}: {}".format(
        config.start_marker_id, config.station_marker_id,
        " -> ".join(str(mid) for mid in route)))
    print("метки читаются словарём {}; моторы не трогаем".format(config.aruco_dict))

    count = 0
    while cycles <= 0 or count < cycles:
        count += 1
        sight = pilot.look()
        if sight is None:
            print("кадра нет")
        elif not sight.seen:
            print("меток в кадре нет")
        else:
            base = sight.base
            step = nav.measured_grid_step(sight.seen, pilot.field, config.marker_size)
            print(
                "метки: {} | под дроном {} ({:.0f} px) | поворот поля {:+.0f}° | "
                "место {} | шаг решётки {}".format(
                    " ".join(str(mid) for mid in sorted(sight.seen)),
                    base.mid if base else "?",
                    base.side if base else 0.0,
                    math.degrees(sight.frame_yaw or 0.0),
                    "({:.2f}, {:.2f})".format(*sight.position) if sight.position else "?",
                    "{:.2f} м".format(step) if step else "нет пары соседей",
                )
            )
        distance = pilot.read_range()
        if distance is not None:
            print("          дальномер: {:.2f} м".format(distance))
        time.sleep(1.0)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════


def _parse_tree_groups(values: Optional[Sequence[str]]) -> Tuple[Tuple[int, ...], ...]:
    if not values:
        return nav.DEFAULT_TREE_GROUPS
    if len(values) == 1 and values[0].strip().lower() == "none":
        return ()
    groups = []
    for value in values:
        ids = tuple(int(part) for part in value.replace(";", ",").split(",") if part.strip())
        if len(ids) < 2:
            raise ValueError("В группе меток дерева нужно минимум две метки: " + value)
        groups.append(ids)
    return tuple(groups)


def _parse_blind_markers(value: Optional[str]) -> Tuple[int, ...]:
    if value is None or value.strip().lower() in ("", "none"):
        return ()
    return tuple(int(part) for part in value.replace(";", ",").split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="БВС-1: перелёт к зарядной станции строго по ArUco-меткам")
    default = MissionConfig()
    parser.add_argument("--map", default=default.map_path, help="карта поля")
    parser.add_argument("--start-marker", type=int, default=default.start_marker_id)
    parser.add_argument("--station-marker", type=int, default=default.station_marker_id)
    parser.add_argument("--alt", type=float, default=default.alt, help="рабочая высота, м")
    parser.add_argument("--speed", type=float, default=default.speed)
    parser.add_argument("--climb-speed", type=float, default=default.climb_speed)
    parser.add_argument("--descent-speed", type=float, default=default.descent_speed)
    parser.add_argument("--marker-size", type=float, default=default.marker_size,
                        help="сторона метки, м — по ней кадр переводится в метры")
    parser.add_argument("--aruco-dict", default=default.aruco_dict)
    parser.add_argument("--gain", type=float, default=default.gain)
    parser.add_argument("--tol", type=float, default=default.tol)
    parser.add_argument("--tries", type=int, default=default.tries)
    parser.add_argument("--blind-markers",
                        default=",".join(str(mid) for mid in default.blind_markers),
                        help="метки, физически закрытые объектами (площадка, кубы "
                             "станций, груз) — их камера не видит никогда; 'none' — нет таких")
    parser.add_argument("--start-tolerance", type=float, default=default.start_tolerance,
                        help="допуск сверки места со стартовой меткой после взлёта, м")
    parser.add_argument("--blind-tolerance", type=float, default=default.blind_tolerance,
                        help="какой расчётный остаток считать «встали» над закрытой "
                             "меткой, м — поднимать, если дрон не может удержаться "
                             "точнее и миссия срывается у станции")
    parser.add_argument("--blind-confirm", type=int, default=default.blind_confirm,
                        help="столько кадров подряд остаток должен быть мал")
    parser.add_argument("--hop-pad", type=float, default=default.hop_pad)
    parser.add_argument("--hop-max-steps", type=float, default=default.hop_max_steps)
    parser.add_argument("--step-length", type=float, default=default.step_length,
                        help="максимум одной горизонтальной команды, м — главный "
                             "регулятор плавности: короче шаг, меньше наклон и рывок")
    parser.add_argument("--climb-step", type=float, default=default.climb_step,
                        help="максимум одной вертикальной команды, м")
    parser.add_argument("--brake-pause", type=float, default=default.brake_pause,
                        help="пауза на гашение скорости после каждого шага, с")
    parser.add_argument("--micro-step", type=float, default=default.micro_step,
                        help="длина микрокоманды внутри шага, м (0 — отдавать шаг "
                             "одной командой, резче)")
    parser.add_argument("--micro-pause", type=float, default=default.micro_pause,
                        help="пауза между микрокомандами, с; вместе с --micro-step "
                             "задаёт фактическую скорость перелёта")
    parser.add_argument("--no-settle-after-hop", dest="settle_after_hop",
                        action="store_false",
                        help="не ждать спокойного кадра после каждого шага (быстрее, "
                             "но поправки считаются по смазанной картинке)")
    parser.add_argument("--alt-fix", type=float, default=default.alt_fix)
    parser.add_argument("--station-height", type=float, default=default.station_height)
    parser.add_argument("--descent-step", type=float, default=default.descent_step)
    parser.add_argument("--touchdown-range", type=float, default=default.touchdown_range)
    parser.add_argument("--tree-markers", action="append", metavar="ID,ID,ID,ID",
                        help="метки, в общем углу которых стоит дерево (флаг повторяемый; "
                             "'none' — деревьев нет)")
    parser.add_argument("--tree-height", type=float, default=default.tree_height)
    parser.add_argument("--tree-clearance", type=float, default=default.tree_clearance,
                        help="зазор до ствола при построении маршрута, м")
    parser.add_argument("--tree-guard-clearance", type=float,
                        default=default.tree_guard_clearance,
                        help="ближе этого к стволу лететь нельзя: команда отменяется, "
                             "миссия прерывается (должен быть не больше --tree-clearance)")
    parser.add_argument("--tree-vertical-margin", type=float,
                        default=default.tree_vertical_margin)
    parser.add_argument("--geofence-margin", type=float, default=default.geofence_margin)
    parser.add_argument("--camera-topic", default=default.camera_topic)
    parser.add_argument("--rangefinder-topic", default=default.rangefinder_topic)
    parser.add_argument("--probe", action="store_true",
                        help="проверка зрения без моторов: что видно и каким будет маршрут")
    parser.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> MissionConfig:
    return MissionConfig(
        map_path=args.map,
        start_marker_id=args.start_marker,
        station_marker_id=args.station_marker,
        alt=args.alt,
        speed=args.speed,
        climb_speed=args.climb_speed,
        descent_speed=args.descent_speed,
        marker_size=args.marker_size,
        aruco_dict=args.aruco_dict,
        gain=args.gain,
        tol=args.tol,
        tries=args.tries,
        hop_pad=args.hop_pad,
        hop_max_steps=args.hop_max_steps,
        step_length=args.step_length,
        climb_step=args.climb_step,
        micro_step=args.micro_step,
        micro_pause=args.micro_pause,
        brake_pause=args.brake_pause,
        settle_after_hop=args.settle_after_hop,
        alt_fix=args.alt_fix,
        station_height=args.station_height,
        descent_step=args.descent_step,
        touchdown_range=args.touchdown_range,
        tree_groups=_parse_tree_groups(args.tree_markers),
        tree_height=args.tree_height,
        blind_markers=_parse_blind_markers(args.blind_markers),
        start_tolerance=args.start_tolerance,
        blind_tolerance=args.blind_tolerance,
        blind_confirm=args.blind_confirm,
        tree_clearance=args.tree_clearance,
        tree_guard_clearance=args.tree_guard_clearance,
        tree_vertical_margin=args.tree_vertical_margin,
        geofence_margin=args.geofence_margin,
        camera_topic=args.camera_topic,
        rangefinder_topic=args.rangefinder_topic,
        quiet=args.quiet,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    field = nav.read_field_map(config.map_path)
    detector = nav.MarkerDetector(config.aruco_dict, max_marker_id=max(field))
    ros = init_ros(config)
    pilot = Pilot(config, ros, field, detector)

    if args.probe:
        try:
            run_probe(pilot)
        except KeyboardInterrupt:
            print("проверка окончена")
        return 0

    try:
        return 0 if run_mission(pilot) else 1
    except KeyboardInterrupt:
        print("!!! прервано с клавиатуры — гасим движение и садимся", flush=True)
        pilot.hold()
        pilot.land()
        return 1
    except Exception as exc:  # noqa: BLE001 - последний рубеж: не падать молча в воздухе
        print("!!! МИССИЯ ПРЕРВАНА: {}".format(exc), flush=True)
        if pilot.armed:
            pilot.hold()
            pilot.land()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
