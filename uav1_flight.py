#!/usr/bin/env python3
"""Полный сценарий БВС-1: взлёт, полёт к зарядке, зарядка, возврат, посадка.

Цель задаётся НОМЕРОМ ArUco-метки, координаты берутся из
``config/field_map.txt``. Летим по прямой на 2 м (деревья на поле метровые,
проходим сверху). Проверок нет: скрипт печатает, что происходит, но полёт
ничем не прерывает.

ЧТО ДЕЛАЕТ (шаги 1, 2, 3, 5 алгоритма из docs/TASK.md)
-------------------------------------------------------
1. взлёт на 2 м                          — лента ЖЁЛТАЯ МИГАЮЩАЯ
2. полёт к зарядной станции              — лента КРАСНАЯ
3. посадка на станцию, зарядка 15 с      — лента КРАСНАЯ МИГАЮЩАЯ,
   за 5 с до взлёта                      — лента ЗЕЛЁНАЯ
4. взлёт, возврат на старт, посадка      — лента ЗЕЛЁНАЯ МИГАЮЩАЯ
5. лента гаснет

Возврат начинается сам, сразу после зарядки: в регламенте он идёт «по команде
с клавиатуры», но здесь никого не ждём — всё автономно от старта до посадки.

Почему не ``frame_id='aruco_5'``: документация Skyris разрешает лететь прямо
к метке (``navigate(frame_id='aruco_5', x=0, y=0, z=1)``), но фрейм ``aruco_N``
живёт, только пока метка видна в кадре. Метку зарядной станции закрывает куб —
она не появится в кадре никогда. Поэтому номер переводим в координаты по карте
и летим в ``aruco_map``.

ЗАПУСК НА ДРОНЕ
---------------
1. Скопировать на борт файл и карту поля::

       scp uav1_flight.py orangepi@10.42.0.1:~
       scp -r config orangepi@10.42.0.1:~

2. Зайти на борт и подготовить окружение::

       ssh orangepi@10.42.0.1
       source /opt/ros/noetic/setup.bash

3. Поставить дрон на площадку «Н» (метка 48) и запустить::

       python3 uav1_flight.py 5      # к метке 5 (зарядка БВС-1) и обратно
       python3 uav1_flight.py        # без аргументов — то же самое
       python3 uav1_flight.py 5 47   # взлетаем не с 48, а с метки 47

Третий аргумент — своя карта поля, для проверки в Gazebo с другой раскладкой::

       python3 uav1_flight.py 12 0 ~/catkin_ws/src/my_field.txt

Карту надо брать ИЗ ТОГО ЖЕ файла, который скормлен ноде ``aruco_map``
(проверить: ``rosparam get /aruco_map/map``). Если файлы разные, дрон считает
место по одной раскладке, а летит по другой — и уверенно улетает не туда.

Второй аргумент — метка старта. Это ТОЧКА ВОЗВРАТА: именно в неё дрон летит
после зарядки. На взлёт она не влияет (взлёт идёт в ``frame_id='body'`` —
«вверх оттуда, где стою»), и видеть её камерой не нужно: место дрон берёт у
ноды ``aruco_map`` по любым видимым меткам, а стартовую всё равно закрывает
площадка «Н».

Сводка «старт — цель — путь» печатается ДО взлёта. Если точки не те, жать
Ctrl+C, дрон ещё стоит на земле.

Пульт держать в руках: аварийный переход в ручной режим — только с него.
"""

import math
import sys
import time

import rospy
from technic import srv
from std_srvs.srv import Trigger

# ─── настройки ────────────────────────────────────────────────────────────
ALT = 2.0                       # рабочая высота, м. Деревья метровые — 2 м их проходят
CLIMB_SPEED = 1.0               # скорость набора высоты, м/с
SPEED = 0.5                     # скорость перелёта, м/с
TOLERANCE = 0.2                 # «долетели», м
TIMEOUT = 40.0                  # дольше этого одну команду не ждём, с
LANDING = 6.0                   # пауза на посадку и остановку моторов, с
CHARGE = 15.0                   # имитация зарядки, с (по регламенту ~15 с)
CHARGE_GREEN = 5.0              # из них последние — с зелёной лентой
MAP = "config/field_map.txt"    # карта поля: id size x y z rot_z rot_y rot_x
DEFAULT_MARKER = 5              # метка зарядной станции БВС-1
DEFAULT_START = 48              # метка площадки «Н»: взлёт и точка возврата

# Цвета ленты по регламенту (docs/TASK.md, шаги 1-5)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
OFF = (0, 0, 0)

# ─── карта поля ───────────────────────────────────────────────────────────
# Перенесено из бывшего lib/marker_nav.py: это всё, что от него было нужно.
# Отдельного модуля больше нет — скрипт возят на борт одним файлом.


def read_field_map(path):
    """Карта поля в формате aruco_pose: ``id size x y z rot_z rot_y rot_x``.

    Возвращает ``{id: (x, y)}``. Высота меток не нужна: они все лежат на полу.
    Углы поворота тоже не нужны нам — но нужны ноде ``aruco_map``, поэтому
    файл общий, и колонки после ``y`` мы просто пропускаем.
    """
    field = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = line.split("#", 1)[0].split()
            if not row:
                continue
            if len(row) < 4:
                raise ValueError(
                    "{}:{}: ожидалось 'id size x y ...', получено {!r}".format(
                        path, line_number, line.strip()
                    )
                )
            field[int(row[0])] = (float(row[2]), float(row[3]))
    if not field:
        raise ValueError("Карта поля пуста: " + path)
    return field


def marker_xy(field, mid):
    """Координаты метки по карте. Нет такой метки — понятная ошибка, не KeyError."""
    if mid not in field:
        raise SystemExit("Метки {} нет в карте поля".format(mid))
    return field[mid]


# ─── откуда и куда ────────────────────────────────────────────────────────
# Считаем до подключения к дрону: опечатка в номере метки выясняется на земле,
# а не в воздухе.
target = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MARKER
start = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_START
# Третий аргумент — своя карта поля. Нужен в Gazebo: там раскладка меток
# другая, и брать её надо ИЗ ТОГО ЖЕ файла, что скормлен ноде aruco_map,
# иначе дрон считает место по одной карте, а летит по другой.
map_path = sys.argv[3] if len(sys.argv) > 3 else MAP

field = read_field_map(map_path)
target_x, target_y = marker_xy(field, target)
start_x, start_y = marker_xy(field, start)
route_length = math.hypot(target_x - start_x, target_y - start_y)

print("─" * 58)
print("карта  {}".format(map_path))
print("старт  метка {:>2} -> ({:.1f}, {:.1f})   (и точка возврата)".format(
    start, start_x, start_y))
print("цель   метка {:>2} -> ({:.1f}, {:.1f})".format(target, target_x, target_y))
print("путь   {:.2f} м в одну сторону, ~{:.0f} с на {:.1f} м/с".format(
    route_length, route_length / SPEED, SPEED))
print("высота {:.1f} м, взлёт ~{:.0f} с на {:.1f} м/с".format(
    ALT, ALT / CLIMB_SPEED, CLIMB_SPEED))
print("зарядка {:.0f} с, из них последние {:.0f} с — зелёная лента".format(
    CHARGE, CHARGE_GREEN))
print("─" * 58)

# ─── подключение к дрону ──────────────────────────────────────────────────
rospy.init_node("fly_to_marker")

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

# ─── 2. полёт к зарядной станции ──────────────────────────────────────────
print("ЛЕТИМ к метке {} (зарядная станция)".format(target))
led("fill", RED, "красная — ищем зарядную станцию")
navigate_wait(x=target_x, y=target_y, z=ALT, frame_id="aruco_map")
rospy.sleep(3.0)
report("НАД СТАНЦИЕЙ", target, target_x, target_y)

# ─── 3. посадка на станцию и зарядка ──────────────────────────────────────
print("ПОСАДКА на станцию")
land()
rospy.sleep(LANDING)

print("ЗАРЯДКА {:.0f} с".format(CHARGE))
led("blink", RED, "красная мигающая — идёт зарядка")
rospy.sleep(CHARGE - CHARGE_GREEN)
led("fill", GREEN, "зелёная — {:.0f} с до взлёта".format(CHARGE_GREEN))
rospy.sleep(CHARGE_GREEN)

# ─── 4. возврат на старт и посадка ────────────────────────────────────────
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
