#!/usr/bin/env python3
"""Простой полёт БВС-1: взлёт, точки маршрута по карте, посадка на станцию.

Короткая версия ``uav1_flight.py`` — то же самое, но без всего, что можно
убрать. Смысл файла именно в размере: его целиком видно за один присест, и
ошибку в нём можно найти глазами, не запуская.

Что выкинуто по сравнению с ``uav1_flight.py``:

* полёт по камере (``--nav-mode markers``, ``cv2``, распознавание меток,
  поворот поля, высота по стороне метки) — здесь место всегда берётся у
  платформы из ``get_telemetry(frame_id='aruco_map')``;
* классы, конфиг-датакласс, слои — здесь простые функции и константы;
* поиск потерянной позиции крестом, сторож деревьев в полёте, геозона,
  удержание курса, дизарм через ``mavros/cmd/arming``, режим ``--probe``.

Что осталось, потому что без этого нельзя:

* маршрут строится **в обход деревьев** (``marker_nav.plan_route``) — деревья
  на поле есть, и лететь к станции по прямой нельзя;
* при любой беде дрон **садится**, а не продолжает вслепую;
* быстрый набор высоты: до высоты захвата карты PX4 держит точку вслепую, и
  чем короче эта фаза, тем меньше сносит (см. README, «Что такое VPE»).

Запуск на дроне::

    python3 simple_flight.py --map config/field_map.txt --station-marker 20

Расчётную часть (карта, деревья, маршрут) можно проверить и на десктопе, без
ROS и дрона — печать маршрута идёт до всякого подключения::

    python3 simple_flight.py --map config/field_map.txt --station-marker 20 --dry-run
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import marker_nav as nav  # noqa: E402  (после sys.path выше)


# ═══════════════════════════════════════════════════════════════════════
#  НАСТРОЙКИ. Всё, что можно покрутить, собрано здесь.
# ═══════════════════════════════════════════════════════════════════════

ALT = 2.0             # рабочая высота, м (деревья 1 м — метр запаса)
SPEED = 0.3           # горизонтальная скорость, м/с
CLIMB_SPEED = 1.5     # скорость набора высоты, м/с — намеренно больше SPEED
DESCENT_SPEED = 0.2   # скорость спуска на куб, м/с

TOLERANCE = 0.25      # «дошли до точки», м
PAD = 0.8             # запас к паузе перелёта на разгон и торможение, с
SETTLE = 3.0          # пауза на успокоение, с
TRIES = 8             # столько попыток на одну точку, дальше сдаёмся

TREE_CLEARANCE = 0.5  # зазор до ствола при построении маршрута, м

DESCENT_STEP = 0.15   # шаг спуска на куб, м
DESCENT_TRIES = 20    # столько шагов спуска, дальше сдаёмся
TOUCHDOWN = 0.15      # дальномер ближе этого — дрон стоит на кубе, м

LOST_TRIES = 10       # столько раз ждём пропавшую позицию по 0.5 с

# Прокси к дрону. Заполняются в connect(), до этого None.
navigate = None
get_telemetry = None
land = None
read_range = None


# ═══════════════════════════════════════════════════════════════════════
#  ROS. Единственное место, где импортируется rospy.
# ═══════════════════════════════════════════════════════════════════════

def connect():
    """Поднять ROS-узел и разложить сервисы дрона по глобальным переменным."""
    global navigate, get_telemetry, land, read_range

    import rospy
    from technic import srv
    from std_srvs.srv import Trigger
    from sensor_msgs.msg import Range

    rospy.init_node("simple_flight")

    navigate = rospy.ServiceProxy("navigate", srv.Navigate)
    get_telemetry = rospy.ServiceProxy("get_telemetry", srv.GetTelemetry)
    land = rospy.ServiceProxy("land", Trigger)

    def range_reader():
        message = rospy.wait_for_message("/rangefinder/range", Range, timeout=1.0)
        return float(message.range)

    read_range = range_reader


# ═══════════════════════════════════════════════════════════════════════
#  ГДЕ МЫ
# ═══════════════════════════════════════════════════════════════════════

def position():
    """Место дрона по карте: ``(x, y, z)`` или ``None``, если карта не видна.

    Платформа сообщает «места не знаю» через NaN — это штатно, метки уходят из
    кадра на кочке и при бликах.
    """
    try:
        telemetry = get_telemetry(frame_id="aruco_map")
    except Exception as error:
        print("  get_telemetry не ответил:", error)
        return None
    if math.isnan(telemetry.x) or math.isnan(telemetry.y):
        return None
    return telemetry.x, telemetry.y, telemetry.z


def wait_position():
    """Дождаться позиции. ``None`` — так и не появилась, дальше лететь нельзя."""
    for _ in range(LOST_TRIES):
        where = position()
        if where is not None:
            return where
        print("  карты не видно, ждём...")
        time.sleep(0.5)
    return None


# ═══════════════════════════════════════════════════════════════════════
#  ПОЛЁТ
# ═══════════════════════════════════════════════════════════════════════

def takeoff():
    """Вертикальный взлёт на рабочую высоту."""
    print("ВЗЛЁТ на {:.1f} м".format(ALT))
    navigate(x=0.0, y=0.0, z=ALT, yaw=0.0, speed=CLIMB_SPEED,
             frame_id="body", auto_arm=True)

    # Пока дрон поднимается, печатаем место раз в полсекунды: по этим строкам
    # видно, с какой высоты появляется карта и сносит ли дрон, пока её нет.
    for _ in range(int((ALT / CLIMB_SPEED + PAD + SETTLE) / 0.5)):
        time.sleep(0.5)
        where = position()
        print("  взлёт:", "({:+.2f}, {:+.2f}, {:.2f})".format(*where)
              if where is not None else "карты не видно")


def goto(x, y, name):
    """Лететь в точку карты, пока не окажемся ближе TOLERANCE. True — дошли."""
    for _ in range(TRIES):
        where = wait_position()
        if where is None:
            return False

        distance = math.hypot(x - where[0], y - where[1])
        print("  {} ({:.1f}, {:.1f}) | сейчас ({:.2f}, {:.2f}) | до цели {:.2f} м".format(
            name, x, y, where[0], where[1], distance))
        if distance <= TOLERANCE:
            return True

        navigate(x=x, y=y, z=ALT, yaw=0.0, speed=SPEED, frame_id="aruco_map")
        time.sleep(distance / SPEED + PAD)

    return False


def descend():
    """Спуск на куб станции шагами по дальномеру. True — коснулись куба."""
    print("СПУСК на станцию")
    for _ in range(DESCENT_TRIES):
        try:
            distance = read_range()
        except Exception as error:
            print("  дальномер молчит:", error)
            distance = None

        print("  спуск: дальномер",
              "{:.2f} м".format(distance) if distance is not None else "нет данных")
        if distance is not None and distance <= TOUCHDOWN:
            print("  касание")
            return True

        navigate(x=0.0, y=0.0, z=-DESCENT_STEP, yaw=0.0, speed=DESCENT_SPEED,
                 frame_id="body")
        time.sleep(DESCENT_STEP / DESCENT_SPEED + PAD)

    return False


def fly_mission(field, route, start_marker):
    """Весь полёт от взлёта до куба. True — сели на станцию."""
    takeoff()

    # Первая точка — своя же стартовая метка. Это и возврат (на взлёте дрон
    # сносит, пока он не видит карты), и проверка: дрон, который не может
    # прийти в заданную точку карты, тем более не пройдёт маршрут.
    start_x, start_y = nav.marker_xy(field, start_marker)
    if not goto(start_x, start_y, "старт"):
        print("!!! на стартовую точку встать не удалось")
        return False

    for marker in route:
        x, y = nav.marker_xy(field, marker)
        if not goto(x, y, "метка {}".format(marker)):
            print("!!! до метки {} не дошли".format(marker))
            return False

    time.sleep(SETTLE)
    return descend()


# ═══════════════════════════════════════════════════════════════════════
#  ЗАПУСК
# ═══════════════════════════════════════════════════════════════════════

def main():
    global ALT  # объявляем до первого упоминания ALT — иначе SyntaxError

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", default="config/field_map.txt")
    parser.add_argument("--start-marker", type=int, default=48,
                        help="метка, с которой взлетаем")
    parser.add_argument("--station-marker", type=int, default=5,
                        help="метка зарядной станции — цель полёта")
    parser.add_argument("--tree-markers", action="append", metavar="ID,ID,ID,ID",
                        help="метки, в общем углу которых стоит дерево "
                             "(флаг повторяемый; по умолчанию — четыре группы из "
                             "marker_nav)")
    parser.add_argument("--alt", type=float, default=ALT, help="рабочая высота, м")
    parser.add_argument("--dry-run", action="store_true",
                        help="только напечатать маршрут, к дрону не подключаться")
    args = parser.parse_args()
    ALT = args.alt

    # Расчётная часть: карта, деревья, маршрут. Здесь ещё нет ни ROS, ни дрона,
    # поэтому всё это проверяется на десктопе.
    field = nav.read_field_map(args.map)
    if not args.tree_markers:
        groups = nav.DEFAULT_TREE_GROUPS
    elif args.tree_markers == ["none"]:
        groups = ()  # деревьев на поле нет
    else:
        groups = tuple(tuple(int(part) for part in value.split(","))
                       for value in args.tree_markers)
    trees = nav.tree_positions(field, groups)
    route = nav.plan_route(field, args.start_marker, args.station_marker,
                           trees=trees, clearance=TREE_CLEARANCE)

    print("деревья: " + ", ".join("({:.1f}, {:.1f})".format(x, y) for x, y in trees))
    print("маршрут {} -> {}: {}".format(
        args.start_marker, args.station_marker,
        " -> ".join(str(marker) for marker in route)))

    if args.dry_run:
        print("--dry-run: к дрону не подключаемся")
        return

    connect()

    # Что бы ни случилось дальше — дрон должен сесть, а не зависнуть в воздухе
    # и не продолжать полёт вслепую.
    try:
        landed = fly_mission(field, route, args.start_marker)
        if landed:
            print(">>> БВС-1 НА ЗАРЯДНОЙ СТАНЦИИ")
        else:
            print("!!! до станции не добрались — садимся")
    except KeyboardInterrupt:
        print("!!! прервано с клавиатуры — садимся")
    except Exception as error:
        print("!!! ОШИБКА:", error)
    finally:
        # Сначала гасим движение («цель там, где я сейчас»), потом садимся.
        # land() вызывается и после удачного касания: моторы надо выключить.
        try:
            navigate(x=0.0, y=0.0, z=0.0, yaw=0.0, speed=DESCENT_SPEED,
                     frame_id="body")
            time.sleep(1.0)
            land()
            print("посадка")
        except Exception as error:
            print("!!! посадка не прошла:", error)


if __name__ == "__main__":
    main()
