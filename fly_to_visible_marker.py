#!/usr/bin/env python3
"""Полёт к ВИДИМОЙ метке без VPE: ``navigate(frame_id='aruco_5', x=0, y=0, z=1)``.

Взлетает, ищет в кадре метку с заданным номером, встаёт ровно над ней на
заданной высоте и садится. Карта поля не нужна, глобальная локализация
(``aruco_map`` → ``/mavros/vision_pose/pose``) не нужна: единственный
источник цели — TF-фрейм ``aruco_N``, который нода ``aruco_detect`` публикует
для каждой метки, попавшей в кадр.

ЧЕМ ОТЛИЧАЕТСЯ ОТ fly_to_marker.py
----------------------------------
====================  ===============================  ========================
что                   fly_to_marker.py                 этот скрипт
====================  ===============================  ========================
цель                  координаты из field_map.txt      фрейм видимой метки
чем задана            ``frame_id='aruco_map'``         ``frame_id='aruco_5'``
нужна карта поля      да, и та же, что у aruco_map     нет
нужен VPE             да (aruco_map + aruco_vpe)       нет
дальность             любая точка поля                 только то, что в кадре
====================  ===============================  ========================

«БЕЗ VPE» — ЧТО ЭТО ЗНАЧИТ И ЧЕГО НЕ ЗНАЧИТ
-------------------------------------------
Не нужна цепочка ``aruco_map`` → ``aruco_vpe`` → ``/mavros/vision_pose/pose``:
скрипт не спрашивает, где дрон на поле, он спрашивает, где метка относительно
дрона. Ноды ``aruco_map``/``aruco_vpe`` можно вообще не поднимать (поднятые —
не мешают).

Но точку в воздухе всё равно удерживает PX4, и **какой-то** источник места ему
нужен: без VPE это оптический поток. Если нет ни того, ни другого, дрон не
держит точку ни в одном режиме, и этот скрипт положения не спасёт — он только
говорит «хочу быть вон там». Проверять до вылета::

    rostopic hz /mavros/px4flow/raw/optical_flow_rad   # поток идёт?
    rosnode list | grep aruco_detect                   # нода меток поднята?
    rostopic echo -n1 /aruco_detect/markers            # метка в кадре?
    rosrun tf tf_echo main_camera_optical aruco_5      # фрейм метки живой?

Последняя проверка — ключевая: если ``tf_echo`` молчит, ``navigate`` с таким
``frame_id`` откажет, и лететь некуда. Размер метки скрипт нигде не задаёт,
его знает сама нода ``aruco_detect`` (параметр ``length``).

ГРАНИЦЫ ПРИМЕНИМОСТИ
--------------------
Фрейм ``aruco_N`` существует, только пока метка в кадре. Отсюда:

- **дальняя метка не годится**: с 2 м камера видит пятачок пары метров, метку
  зарядки в 6 м от старта отсюда не увидеть — туда только по карте
  (``fly_to_marker.py``);
- **закрытая метка не годится** совсем: 0, 5, 37, 48 физически перекрыты
  площадкой «Н», кубами станций и грузом, в кадре они не появятся никогда.
  Для посадки на станцию наводись на её соседа, а не на метку под кубом;
- **на земле метки под собой не видно** — дрон её закрывает. Поиск начинается
  после взлёта, это норма.

Наводка идёт в несколько заходов, и это не перестраховка: ``navigate``
переводит цель во фрейм метки один раз, в момент команды, — дальше дрон летит
по счислению PX4 и за перелёт успевает уехать. Каждая следующая команда берёт
свежий снимок метки, поэтому промах от захода к заходу уменьшается.

Курс не трогаем: ``yaw=NaN`` + ``yaw_rate=0`` — «держать текущий». Здесь это
важнее, чем в ``body``: ``yaw=0`` во фрейме метки означает «развернуться по её
осям», и дрон крутнётся на произвольный угол ещё до перелёта.

ЗАПУСК НА ДРОНЕ
---------------
1. Скопировать файл на борт::

       scp fly_to_visible_marker.py orangepi@10.42.0.1:~

2. Зайти на борт и подготовить окружение::

       ssh orangepi@10.42.0.1
       source /opt/ros/noetic/setup.bash

3. Положить метку под дрон (или поставить дрон рядом с ней) и запустить::

       python3 fly_to_visible_marker.py 5        # встать над меткой 5, сесть
       python3 fly_to_visible_marker.py 5 1.5    # зависнуть на 1.5 м над ней
       python3 fly_to_visible_marker.py          # метка по умолчанию

Пульт держать в руках: аварийный переход в ручной режим — только с него.
"""

import math
import sys
import time

import rospy
from technic import srv
from std_srvs.srv import Trigger

# ─── настройки ────────────────────────────────────────────────────────────
DEFAULT_MARKER = 5       # к какой метке лететь, если номер не задан
DEFAULT_HEIGHT = 1.0     # на сколько зависать над меткой, м (z в её фрейме)
TAKEOFF = 1.5            # высота взлёта, м: ниже метка рядом не попадёт в кадр
CLIMB_STEP = 0.5         # шаг подъёма, если метки не видно, м
CLIMB_MAX = 3.0          # выше не поднимаемся, м
CLIMB_SPEED = 1.0        # скорость набора высоты, м/с
SPEED = 0.5              # скорость перелёта, м/с
TOLERANCE = 0.2          # «долетели» по navigate_target, м
AIM_TOLERANCE = 0.15     # «встали над меткой» по горизонтали, м
AIM_TRIES = 5            # столько заходов на уточнение наводки
TIMEOUT = 20.0           # дольше этого одну команду не ждём, с
SETTLE = 2.0             # пауза на успокоение и свежий кадр, с
LANDING = 6.0            # пауза на посадку и остановку моторов, с
FRAME = "aruco_{}"       # как aruco_detect называет фрейм метки

# ─── что делаем ───────────────────────────────────────────────────────────
# Разбираем аргументы до подключения к дрону: опечатка выясняется на земле.
marker = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MARKER
height = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_HEIGHT
frame = FRAME.format(marker)

print("─" * 58)
print("цель   метка {} -> фрейм {} (карта поля не нужна)".format(marker, frame))
print("встать {:.1f} м над меткой, до {} заходов, допуск {:.2f} м".format(
    height, AIM_TRIES, AIM_TOLERANCE))
print("взлёт  {:.1f} м, поиск метки подъёмом до {:.1f} м шагом {:.1f} м".format(
    TAKEOFF, CLIMB_MAX, CLIMB_STEP))
print("─" * 58)

# ─── подключение к дрону ──────────────────────────────────────────────────
rospy.init_node("fly_to_visible_marker")

navigate = rospy.ServiceProxy("navigate", srv.Navigate)
get_telemetry = rospy.ServiceProxy("get_telemetry", srv.GetTelemetry)
land = rospy.ServiceProxy("land", Trigger)


def go(x=0.0, y=0.0, z=0.0, frame_id="body", speed=SPEED, auto_arm=False):
    """Одна команда ``navigate`` с сохранением курса.

    Курс держим текущий (``yaw=NaN`` + ``yaw_rate=0``): во фрейме метки
    ``yaw=0`` означает «развернуться по осям метки». Если в этой сборке
    ``Navigate`` окажется без поля ``yaw_rate`` (прошивка экспериментальная,
    состав сервисов не гарантирован) — летим с ``yaw=0``: дрон довернётся по
    метке, но долетит.
    """
    try:
        return navigate(x=x, y=y, z=z, yaw=float("nan"), yaw_rate=0.0,
                        speed=speed, frame_id=frame_id, auto_arm=auto_arm)
    except (TypeError, AttributeError):
        print("  в Navigate нет yaw_rate — летим с yaw=0 (курс развернётся по метке)")
        return navigate(x=x, y=y, z=z, yaw=0.0, speed=speed,
                        frame_id=frame_id, auto_arm=auto_arm)


def wait_arrival(timeout=TIMEOUT):
    """Ждать, пока долетим (пример из документации Skyris).

    Ждём по фрейму ``navigate_target`` — «сколько осталось до цели»: сервис
    ``navigate`` возвращается сразу, дрон летит уже сам. Не дождались — идём
    дальше: висеть в ожидании дороже, чем сделать следующий заход.
    """
    deadline = time.time() + timeout
    while not rospy.is_shutdown() and time.time() < deadline:
        left = get_telemetry(frame_id="navigate_target")
        if math.sqrt(left.x ** 2 + left.y ** 2 + left.z ** 2) < TOLERANCE:
            return True
        rospy.sleep(0.2)
    print("  за {:.0f} с не долетели — продолжаем".format(timeout))
    return False


def offset():
    """Где дрон относительно метки: ``(x, y, z)`` в её фрейме, или ``None``.

    ``None`` — метки в кадре нет. Штатный ответ на пропавший фрейм —
    NaN в телеметрии, но экспериментальная прошивка может и ругнуться
    исключением: оба случая значат одно и то же — наводиться не по чему.
    """
    try:
        here = get_telemetry(frame_id=frame)
    except rospy.ServiceException as error:
        print("  {} не отдался: {}".format(frame, error))
        return None
    if math.isnan(here.x) or math.isnan(here.y) or math.isnan(here.z):
        return None
    return here.x, here.y, here.z


def touchdown(why):
    """Сесть и заглушить моторы. Единственный способ закончить полёт."""
    print("ПОСАДКА: {}".format(why))
    land()
    rospy.sleep(LANDING)


# ─── 1. взлёт ─────────────────────────────────────────────────────────────
# frame_id='body' — «вверх оттуда, где стою»: ни карта, ни метки не нужны.
print("ВЗЛЁТ на {:.1f} м".format(TAKEOFF))
go(z=TAKEOFF, frame_id="body", speed=CLIMB_SPEED, auto_arm=True)
wait_arrival()
rospy.sleep(SETTLE)

# ─── 2. поиск метки ───────────────────────────────────────────────────────
# Единственный манёвр поиска — подъём: чем выше, тем шире кадр. В стороны без
# карты не уходим — вернуться будет некуда.
alt = TAKEOFF
seen = offset()
while seen is None and alt < CLIMB_MAX - 1e-6:
    step = min(CLIMB_STEP, CLIMB_MAX - alt)
    alt += step
    print("метки {} не видно — поднимаемся до ~{:.1f} м".format(marker, alt))
    go(z=step, frame_id="body", speed=CLIMB_SPEED)
    wait_arrival()
    rospy.sleep(SETTLE)
    seen = offset()

if seen is None:
    touchdown("метку {} так и не увидели с {:.1f} м".format(marker, alt))
    print("ГОТОВО (без полёта к метке)")
    sys.exit(1)

# Печатаем как есть — место дрона в осях метки. Пересчитывать это в «метка
# правее на столько-то» нельзя: оси метки повёрнуты относительно дрона на
# неизвестный угол, и знаки получатся выдуманные.
print("ВИЖУ метку {}: дрон в её осях ({:.2f}, {:.2f}), выше неё на {:.2f} м".format(
    marker, seen[0], seen[1], seen[2]))

# ─── 3. наводка на метку ──────────────────────────────────────────────────
# Здесь и происходит то самое navigate(frame_id='aruco_N', x=0, y=0, z=height).
# Каждый заход — свежий снимок фрейма метки, поэтому промах падает от захода
# к заходу; выходим раньше, как только попали в допуск.
for attempt in range(1, AIM_TRIES + 1):
    print("ЗАХОД {}/{}: navigate(frame_id='{}', x=0, y=0, z={:.1f})".format(
        attempt, AIM_TRIES, frame, height))
    result = go(x=0.0, y=0.0, z=height, frame_id=frame)
    if not result.success:
        # Обычная причина — фрейм метки протух: она вышла из кадра, пока мы
        # летели. Ждём кадр и пробуем снова, дрон висит на месте.
        print("  navigate отказал: {}".format(result.message))
        rospy.sleep(SETTLE)
        continue

    wait_arrival()
    rospy.sleep(SETTLE)

    seen = offset()
    if seen is None:
        print("  метка ушла из кадра — ждём кадр")
        rospy.sleep(SETTLE)
        continue

    away = math.hypot(seen[0], seen[1])
    print("  промах {:.2f} м по горизонтали, высота над меткой {:.2f} м".format(
        away, seen[2]))
    if away < AIM_TOLERANCE:
        print("ВСТАЛИ над меткой {}".format(marker))
        break
else:
    print("за {} заходов ближе {:.2f} м не подошли — садимся как есть".format(
        AIM_TRIES, AIM_TOLERANCE))

# ─── 4. посадка ───────────────────────────────────────────────────────────
touchdown("наводка окончена")
print("ГОТОВО")
