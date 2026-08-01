#!/usr/bin/env python3
"""Простой полёт БВС-1 до метки по её НОМЕРУ: взлёт на 2 м, перелёт, посадка.

То же самое, что ``fly_to_station.py``, но цель задаётся не координатами, а
номером ArUco-метки: ``python3 fly_to_marker.py 5``. Координаты скрипт
находит сам в ``config/field_map.txt`` — при смене раскладки поля правится
только карта, а код трогать не надо.

Никаких проверок: ни обхода деревьев, ни геозоны, ни спуска по дальномеру.
Летим по прямой на 2 м (деревья на поле метровые, проходим сверху) и садимся
штатным ``land()``.

Почему не ``frame_id='aruco_5'``: документация Skyris разрешает лететь прямо
к метке (``navigate(frame_id='aruco_5', x=0, y=0, z=1)``), но фрейм ``aruco_N``
живёт, только пока метка видна в кадре. Метку зарядной станции закрывает куб —
она не появится в кадре никогда. Поэтому номер переводим в координаты по карте
и летим в ``aruco_map``.

ЗАПУСК НА ДРОНЕ
---------------
1. Скопировать на борт файл и карту поля::

       scp fly_to_marker.py orangepi@10.42.0.1:~
       scp -r config orangepi@10.42.0.1:~

2. Зайти на борт и подготовить окружение::

       ssh orangepi@10.42.0.1
       source /opt/ros/noetic/setup.bash

3. Поставить дрон на площадку «Н» (метка 48) и запустить::

       python3 fly_to_marker.py 5      # лететь к метке 5 (зарядка БВС-1)
       python3 fly_to_marker.py        # без аргументов — к метке 5 с метки 48
       python3 fly_to_marker.py 5 47   # взлетаем не с 48, а с метки 47

Второй аргумент — метка, с которой взлетаем. На сам полёт он НЕ влияет: взлёт
идёт в ``frame_id='body'`` («вверх оттуда, где стою»), а место дрон берёт у
ноды ``aruco_map``. Нужен он только для сводки перед взлётом и чтобы после
взлёта было видно, насколько дрон сошёл со стартовой метки.

Сводка «старт — цель — путь» печатается ДО взлёта. Если точки не те, жать
Ctrl+C, дрон ещё стоит на земле.

Пульт держать в руках: аварийный переход в ручной режим — только с него.
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import marker_nav as nav  # noqa: E402  (после sys.path выше)

import rospy  # noqa: E402
from technic import srv  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

# ─── настройки ────────────────────────────────────────────────────────────
ALT = 2.0                       # рабочая высота, м. Деревья метровые — 2 м их проходят
CLIMB_SPEED = 1.0               # скорость набора высоты, м/с
SPEED = 0.5                     # скорость перелёта, м/с
TOLERANCE = 0.2                 # «долетели», м
TIMEOUT = 40.0                  # дольше этого одну команду не ждём, с
MAP = "config/field_map.txt"    # карта поля: id size x y z rot_z rot_y rot_x
DEFAULT_MARKER = 5              # метка зарядной станции БВС-1
DEFAULT_START = 48              # метка площадки «Н», с которой взлетаем

# ─── откуда и куда ────────────────────────────────────────────────────────
# Считаем до подключения к дрону: опечатка в номере метки выясняется на земле,
# а не в воздухе.
target = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MARKER
start = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_START

field = nav.read_field_map(MAP)
target_x, target_y = nav.marker_xy(field, target)
start_x, start_y = nav.marker_xy(field, start)
route_length = math.hypot(target_x - start_x, target_y - start_y)

print("─" * 58)
print("старт  метка {:>2} -> ({:.1f}, {:.1f})".format(start, start_x, start_y))
print("цель   метка {:>2} -> ({:.1f}, {:.1f})".format(target, target_x, target_y))
print("путь   {:.2f} м по прямой, ~{:.0f} с на {:.1f} м/с".format(
    route_length, route_length / SPEED, SPEED))
print("высота {:.1f} м, взлёт ~{:.0f} с на {:.1f} м/с".format(
    ALT, ALT / CLIMB_SPEED, CLIMB_SPEED))
print("─" * 58)

# ─── подключение к дрону ──────────────────────────────────────────────────
rospy.init_node("fly_to_marker")

navigate = rospy.ServiceProxy("navigate", srv.Navigate)
get_telemetry = rospy.ServiceProxy("get_telemetry", srv.GetTelemetry)
land = rospy.ServiceProxy("land", Trigger)


def navigate_wait(x=0.0, y=0.0, z=0.0, speed=SPEED, frame_id="body", auto_arm=False):
    """Лететь в точку и ждать, пока долетим (пример из документации Skyris).

    Ждём по фрейму ``navigate_target`` — это «сколько осталось до цели»:
    сервис ``navigate`` возвращается сразу, дрон летит уже сам.
    """
    navigate(x=x, y=y, z=z, yaw=0.0, speed=speed, frame_id=frame_id, auto_arm=auto_arm)

    deadline = time.time() + TIMEOUT
    while not rospy.is_shutdown() and time.time() < deadline:
        left = get_telemetry(frame_id="navigate_target")
        if math.sqrt(left.x ** 2 + left.y ** 2 + left.z ** 2) < TOLERANCE:
            return
        rospy.sleep(0.2)


def report(what, mark, mark_x, mark_y):
    """Напечатать, где дрон сейчас и насколько это далеко от метки ``mark``.

    Ничего не проверяет и полёт не прерывает — только показывает цифру, по
    которой потом видно, куда дрон уехал и на каком этапе.
    """
    telemetry = get_telemetry(frame_id="aruco_map")
    if math.isnan(telemetry.x) or math.isnan(telemetry.y):
        print("{}: карты меток не видно, места не знаем".format(what))
        return
    away = math.hypot(telemetry.x - mark_x, telemetry.y - mark_y)
    print("{}: ({:.2f}, {:.2f}, {:.2f}) | до метки {} — {:.2f} м".format(
        what, telemetry.x, telemetry.y, telemetry.z, mark, away))


# ─── полёт ────────────────────────────────────────────────────────────────
print("взлёт на {:.1f} м".format(ALT))
navigate_wait(z=ALT, speed=CLIMB_SPEED, frame_id="body", auto_arm=True)
rospy.sleep(3.0)  # повисеть, дать нодам увидеть карту меток
report("НАЧАЛО", start, start_x, start_y)

print("летим к метке {}".format(target))
navigate_wait(x=target_x, y=target_y, z=ALT, frame_id="aruco_map")
rospy.sleep(3.0)
report("КОНЕЦ ", target, target_x, target_y)

print("посадка")
land()
