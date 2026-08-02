#!/usr/bin/env python3
"""Сценарий БВС-2: взлёт, груз, своя зарядка, возврат, посадка — без захвата.

Тот же приём, что в проверенном на тестах ``fly_to_marker.py``: цели задаются
НОМЕРАМИ ArUco-меток, координаты берутся из ``config/field_map.txt``, летим по
прямой на 2 м (деревья на поле метровые, проходим сверху). Проверок нет:
скрипт печатает, что происходит, но полёт ничем не прерывает.

ЧЕМ ОТЛИЧАЕТСЯ ОТ ПОЛНОГО РЕГЛАМЕНТА
------------------------------------
Сознательно выкинуты две вещи — обе требуют железа и связи, которых в этом
варианте нет:

* **захвата груза нет.** Сервопривод не трогаем. На метке груза дрон просто
  садится и ждёт ``CARGO_WAIT`` секунд — столько же, сколько заняло бы
  зависание с захватом. Дальше летит «как будто с грузом»;
* **запроса на посадку у зарядной станции нет.** Станция не спрашивается и
  не ждётся: прилетели — сели. Шаг 7 алгоритма (станция сама определяет БВС
  по цвету ленты и приглашает на посадку) в этом скрипте не участвует, лента
  всё равно горит по регламенту — станция, если она включена, увидит цвет.

Всё остальное — по алгоритму из docs/TASK.md, шаги 1, 2, 4, 5, 6, 8.

ЧТО ДЕЛАЕТ
----------
1. взлёт на 2 м с площадки «Н»            — лента ЖЁЛТАЯ МИГАЮЩАЯ
2. полёт в зону груза (метка 0)           — лента ЖЁЛТАЯ МИГАЮЩАЯ
3. посадка на метку груза, ожидание 15 с  — лента КРАСНАЯ
   (вместо захвата: сели и ждём)
4. взлёт, полёт к своей станции (метка 37) — лента КРАСНАЯ
5. посадка на станцию, зарядка 15 с       — лента КРАСНАЯ МИГАЮЩАЯ,
   за 5 с до взлёта                       — лента ЗЕЛЁНАЯ
6. взлёт, возврат на старт, посадка       — лента ЗЕЛЁНАЯ МИГАЮЩАЯ
7. лента гаснет

Станция БВС-2 — метка 37, это ОТДЕЛЬНЫЙ куб, не та станция на метке 5, куда
летает БВС-1: у каждого БВС своя. Метку станции закрывает тёмно-синий куб
80 см, метку груза закрывает сам груз — ни ту, ни другую камера не увидит
никогда. Поэтому ``frame_id='aruco_N'`` не годится (фрейм метки живёт, только
пока метка в кадре): номера переводим в координаты по карте и летим в
``aruco_map``, где место считается по любым другим видимым меткам вокруг.

Куб 80 см здесь сажается штатным ``land()`` — так же, как в ``fly_to_marker.py``,
который на тестах отработал. Управляемого спуска по дальномеру в этом
варианте нет: меньше кода — меньше мест, где сорвётся попытка.

ЗАПУСК НА ДРОНЕ
---------------
1. Скопировать на борт файл и карту поля::

       scp fly_uav2.py orangepi@10.42.0.1:~
       scp -r config orangepi@10.42.0.1:~

2. Зайти на борт и подготовить окружение::

       ssh orangepi@10.42.0.1
       source /opt/ros/noetic/setup.bash

3. Поставить дрон на его площадку «Н» и запустить::

       python3 fly_uav2.py 42          # старт с метки 42, груз 0, станция 37
       python3 fly_uav2.py 42 0 37     # то же самое явно
       python3 fly_uav2.py 42 3 41     # другая раскладка груза и станции

Четвёртый аргумент — своя карта поля, для проверки в Gazebo с другой
раскладкой::

       python3 fly_uav2.py 42 0 37 ~/catkin_ws/src/my_field.txt

Карту надо брать ИЗ ТОГО ЖЕ файла, который скормлен ноде ``aruco_map``
(проверить: ``rosparam get /aruco_map/map``). Если файлы разные, дрон считает
место по одной раскладке, а летит по другой — и уверенно улетает не туда.

Первый аргумент обязателен и умолчания не имеет: стартовая метка БВС-2 по
регламенту случайна и объявляется перед попыткой. Это ТОЧКА ВОЗВРАТА: именно
в неё дрон летит в конце. На взлёт она не влияет (взлёт идёт в
``frame_id='body'`` — «вверх оттуда, где стою»), и видеть её камерой не нужно:
место дрон берёт у ноды ``aruco_map`` по любым видимым меткам, а стартовую
всё равно закрывает площадка «Н».

Сводка «старт — груз — станция — путь» печатается ДО взлёта. Если точки не те,
жать Ctrl+C, дрон ещё стоит на земле.

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
LANDING = 6.0                   # пауза на посадку и остановку моторов, с
CARGO_WAIT = 15.0               # ждём на метке груза вместо захвата, с
CHARGE = 15.0                   # имитация зарядки, с (по регламенту ~15 с)
CHARGE_GREEN = 5.0              # из них последние — с зелёной лентой
MAP = "config/field_map.txt"    # карта поля: id size x y z rot_z rot_y rot_x
DEFAULT_CARGO = 0               # метка груза
DEFAULT_STATION = 37            # метка зарядной станции БВС-2 (своя, не 5)

# Цвета ленты по регламенту (docs/TASK.md, Табл.1)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
OFF = (0, 0, 0)

# ─── откуда и куда ────────────────────────────────────────────────────────
# Считаем до подключения к дрону: опечатка в номере метки выясняется на земле,
# а не в воздухе.
if len(sys.argv) < 2:
    print(__doc__.split("ЗАПУСК НА ДРОНЕ")[0].strip())
    print("\nНУЖЕН НОМЕР СТАРТОВОЙ МЕТКИ:")
    print("    python3 fly_uav2.py <старт> [груз] [станция] [карта]")
    print("    python3 fly_uav2.py 42")
    sys.exit(1)

start = int(sys.argv[1])
cargo = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CARGO
station = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_STATION
# Четвёртый аргумент — своя карта поля. Нужен в Gazebo: там раскладка меток
# другая, и брать её надо ИЗ ТОГО ЖЕ файла, что скормлен ноде aruco_map,
# иначе дрон считает место по одной карте, а летит по другой.
map_path = sys.argv[4] if len(sys.argv) > 4 else MAP

field = nav.read_field_map(map_path)
start_x, start_y = nav.marker_xy(field, start)
cargo_x, cargo_y = nav.marker_xy(field, cargo)
station_x, station_y = nav.marker_xy(field, station)

leg1 = math.hypot(cargo_x - start_x, cargo_y - start_y)
leg2 = math.hypot(station_x - cargo_x, station_y - cargo_y)
leg3 = math.hypot(start_x - station_x, start_y - station_y)
route_length = leg1 + leg2 + leg3

print("─" * 58)
print("карта    {}".format(map_path))
print("старт    метка {:>2} -> ({:.1f}, {:.1f})   (и точка возврата)".format(
    start, start_x, start_y))
print("груз     метка {:>2} -> ({:.1f}, {:.1f})   (садимся и ждём, захвата нет)".format(
    cargo, cargo_x, cargo_y))
print("станция  метка {:>2} -> ({:.1f}, {:.1f})   (садимся сразу, без запроса)".format(
    station, station_x, station_y))
print("путь     {:.1f} + {:.1f} + {:.1f} = {:.2f} м, ~{:.0f} с на {:.1f} м/с".format(
    leg1, leg2, leg3, route_length, route_length / SPEED, SPEED))
print("высота   {:.1f} м, взлёт ~{:.0f} с на {:.1f} м/с".format(
    ALT, ALT / CLIMB_SPEED, CLIMB_SPEED))
print("ожидание {:.0f} с на грузе, зарядка {:.0f} с (последние {:.0f} с — зелёная)".format(
    CARGO_WAIT, CHARGE, CHARGE_GREEN))
print("─" * 58)

# ─── подключение к дрону ──────────────────────────────────────────────────
rospy.init_node("fly_uav2")

navigate = rospy.ServiceProxy("navigate", srv.Navigate)
get_telemetry = rospy.ServiceProxy("get_telemetry", srv.GetTelemetry)
land = rospy.ServiceProxy("land", Trigger)
set_effect = rospy.ServiceProxy("led/set_effect", srv.SetLEDEffect)


def led(effect, color, what):
    """Лента: ``effect`` — fill/blink/fade/flash/rainbow, ``color`` — (r, g, b).

    Отказ ленты полёт не прерывает: балл за индикацию потеряется, но дрон
    висит в воздухе и его надо посадить.
    """
    print("  лента: {}".format(what))
    try:
        set_effect(effect=effect, r=color[0], g=color[1], b=color[2])
    except Exception as error:
        print("  лента не отозвалась:", error)


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


# ─── 1. взлёт ─────────────────────────────────────────────────────────────
print("ВЗЛЁТ на {:.1f} м".format(ALT))
led("blink", YELLOW, "жёлтая мигающая — взлёт")
navigate_wait(z=ALT, speed=CLIMB_SPEED, frame_id="body", auto_arm=True)
rospy.sleep(3.0)  # повисеть, дать нодам увидеть карту меток
report("НАЧАЛО", start, start_x, start_y)

# ─── 2. полёт в зону груза ────────────────────────────────────────────────
# Лента остаётся жёлтой мигающей: по регламенту БВС-2 идёт к грузу под тем же
# цветом, что и взлетал, — красная включится только «после захвата».
print("ЛЕТИМ к метке {} (груз)".format(cargo))
led("blink", YELLOW, "жёлтая мигающая — летим за грузом")
navigate_wait(x=cargo_x, y=cargo_y, z=ALT, frame_id="aruco_map")
rospy.sleep(3.0)
report("НАД ГРУЗОМ", cargo, cargo_x, cargo_y)

# ─── 3. посадка на груз и ожидание вместо захвата ─────────────────────────
# Захвата нет: сервопривод не трогаем, просто садимся и стоим. Лента красная —
# ровно то, что регламент требует показывать «после захвата груза».
print("ПОСАДКА на метку груза")
land()
rospy.sleep(LANDING)

print("ЖДЁМ {:.0f} с (вместо захвата груза)".format(CARGO_WAIT))
led("fill", RED, "красная — груз «взят»")
rospy.sleep(CARGO_WAIT)

# ─── 4. полёт к своей зарядной станции ────────────────────────────────────
print("ВЗЛЁТ с груза")
navigate_wait(z=ALT, speed=CLIMB_SPEED, frame_id="body", auto_arm=True)
rospy.sleep(3.0)

print("ЛЕТИМ к метке {} (зарядная станция БВС-2)".format(station))
navigate_wait(x=station_x, y=station_y, z=ALT, frame_id="aruco_map")
rospy.sleep(3.0)
report("НАД СТАНЦИЕЙ", station, station_x, station_y)

# ─── 5. посадка на станцию и зарядка ──────────────────────────────────────
# Запроса на посадку нет: станцию не спрашиваем и разрешения не ждём.
print("ПОСАДКА на станцию (без запроса — садимся сразу)")
land()
rospy.sleep(LANDING)

print("ЗАРЯДКА {:.0f} с".format(CHARGE))
led("blink", RED, "красная мигающая — идёт зарядка")
rospy.sleep(CHARGE - CHARGE_GREEN)
led("fill", GREEN, "зелёная — {:.0f} с до взлёта".format(CHARGE_GREEN))
rospy.sleep(CHARGE_GREEN)

# ─── 6. возврат на старт и посадка ────────────────────────────────────────
print("ЗАРЯДКА ОКОНЧЕНА, летим домой")
print("ВЗЛЁТ со станции")
led("blink", GREEN, "зелёная мигающая — возвращаемся")
navigate_wait(z=ALT, speed=CLIMB_SPEED, frame_id="body", auto_arm=True)
rospy.sleep(3.0)

print("ВОЗВРАТ к метке {}".format(start))
navigate_wait(x=start_x, y=start_y, z=ALT, frame_id="aruco_map")
rospy.sleep(3.0)
report("ДОМА", start, start_x, start_y)

print("ПОСАДКА")
land()
rospy.sleep(LANDING)

led("fill", OFF, "гасим — миссия окончена")
print("ГОТОВО")
