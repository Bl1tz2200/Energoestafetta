#!/usr/bin/env python3
"""Самый простой полёт БВС-1: взлёт на 2 м, перелёт к зарядке, посадка.

Никаких проверок, геозон, сторожей и режимов. Только то, что описано в
документации Skyris Technic 6S (docs.skyris.ru/technic6S): сервисы
``navigate``, ``get_telemetry``, ``land``.

Куда лететь, дрон понимает по ArUco-меткам поля: за него это делает нода
``aruco_map`` — она распознаёт метки в кадре и выдаёт положение дрона в
системе координат карты. Поэтому цель задаётся прямо координатами карты
(``frame_id='aruco_map'``), и никакого своего распознавания писать не нужно.

ЗАПУСК НА ДРОНЕ
---------------
1. Скопировать файл на борт::

       scp fly_to_station.py orangepi@10.42.0.1:~

2. Зайти на борт и подготовить окружение::

       ssh orangepi@10.42.0.1
       source /opt/ros/noetic/setup.bash

3. Поставить дрон на площадку «Н» (метка 48) и запустить::

       python3 fly_to_station.py

Пульт держать в руках: посадка по любой причине делается штатным ``land()``,
но аварийный переход в ручной режим — только с пульта.

ЕСЛИ СТАНЦИЯ В ДРУГОМ МЕСТЕ
---------------------------
Поменять STATION_X / STATION_Y ниже. Координаты берутся из
``config/field_map.txt``: строка ``5 0.33 6.00 1.00 ...`` означает, что метка
5 лежит в точке x=6.00, y=1.00.
"""

import math
import time

import rospy
from technic import srv
from std_srvs.srv import Trigger

# ─── настройки ────────────────────────────────────────────────────────────
ALT = 2.0            # рабочая высота, м. Деревья на поле метровые — 2 м их проходят
CLIMB_SPEED = 1.0    # скорость набора высоты, м/с
SPEED = 0.5          # скорость перелёта, м/с
STATION_X = 6.0      # зарядная станция БВС-1: метка 5 по config/field_map.txt
STATION_Y = 1.0
TOLERANCE = 0.2      # «долетели», м
TIMEOUT = 40.0       # дольше этого одну команду не ждём, с

# ─── подключение к дрону ──────────────────────────────────────────────────
rospy.init_node("fly_to_station")

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


# ─── полёт ────────────────────────────────────────────────────────────────
print("взлёт на {:.1f} м".format(ALT))
navigate_wait(z=ALT, speed=CLIMB_SPEED, frame_id="body", auto_arm=True)
rospy.sleep(3.0)  # повисеть, дать нодам увидеть карту меток

print("летим к станции ({:.1f}, {:.1f})".format(STATION_X, STATION_Y))
navigate_wait(x=STATION_X, y=STATION_Y, z=ALT, frame_id="aruco_map")
rospy.sleep(3.0)

print("посадка")
land()
