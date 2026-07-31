#!/usr/bin/env python3
"""Полёт БВС-1: взлёт → зарядная станция на кубе → ожидание → возврат.

Реализует шаги 1, 3, 5 алгоритма из TASK.md. Зарядная станция стоит на
тёмно-синем кубе 80 см и закрывает собой ArUco-метку, поэтому лететь «по
видимой метке станции» нельзя — навигация идёт по глобальной карте поля
(``aruco_map``, docs.skyris.ru/technic6S/ArucoMap.html). Посадка на куб
выполняется управляемым спуском с контролем лазерного дальномера
(``flight_core.controlled_descent_and_disarm``), а не штатным ``land()`` —
его поведение на приподнятой поверхности не описано в документации Skyris.
Обычная ``land()`` используется только при возврате на стартовую метку, где
под дроном ровный пол.

**Алгоритм перелёта переработан после аварии 2026-08-01** (дрон взлетел,
пошёл «по меткам» в сторону станции, но фактически ушёл за пределы поля и
упал). Полный разбор причин — в докстринге ``lib/flight_nav.py``; здесь
кратко, что изменилось в сценарии:

* маршрут больше не строится по узлам решётки меток от метки-якоря
  (``flight_core.marker_path``) — дрон летит **по прямой к цели**,
  нарезанной на короткие перегоны, и позиция перечитывается перед каждым
  перегоном (``flight_nav.fly_to``);
* локализация не просто «видно хоть одну метку», а сверяется с этими
  метками: если ``aruco_map`` показывает позицию далеко от метки, которую
  камера видит прямо сейчас, миссия останавливается (``read_localization``);
* набор высоты разбит на два этапа: сперва малая высота
  (``--check-altitude``) и проверка, что позиция совпадает со стартовой
  меткой, и только потом крейсерская — ошибочная локализация обнаруживается
  на метре, а не на скорости в трёх метрах от сетки;
* каждый перегон идёт под сторожем геозоны (``flight_nav.geofence_guard``),
  который прерывает уже отданную команду navigate() при выходе за поле или
  пропаже позиции;
* любая такая ошибка гасит движение (``hover_in_place``) и приводит к
  посадке, а не к продолжению миссии вслепую.

Расположение объектов на поле может меняться от попытки к попытке, но
известно заранее — поэтому id меток и высота куба вынесены в ``MissionConfig``
и параметры командной строки, а не зашиты в логику.

Запуск на дроне::

    python3 bvs1_flight.py --map config/field_map.txt --start-marker 48 \\
        --station-marker 5 --station-height 0.8

Проверка модуля без ROS 1/дрона::

    python3 bvs1_flight.py --self-test
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Callable, Dict, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import flight_core as fc
import flight_nav as nav
import led_interface as led


@dataclass(frozen=True)
class MissionConfig:
    """Все настраиваемые параметры попытки — редактируются здесь/через CLI,
    без правки логики ``run_mission``."""

    map_path: str = "config/field_map.txt"
    start_marker_id: int = 48
    station_marker_id: int = 5
    station_height_m: float = 0.8
    landing_safety_margin_m: float = 0.1
    cruise_altitude_m: float = 2.0
    # Высота первой (проверочной) ступени взлёта: на ней подтверждается, что
    # aruco_map показывает стартовую метку, и только потом набирается
    # крейсерская. Ошибка локализации на 1 м безопасна, на крейсерской - нет.
    check_altitude_m: float = 1.0
    approach_speed: float = 0.5
    cruise_speed: float = 0.4
    descent_speed: float = 0.3
    descent_step_m: float = 0.15
    touchdown_threshold_m: float = 0.12
    stabilize_tolerance_m: float = 0.25
    stabilize_hold_s: float = 1.0
    stabilize_timeout_s: float = 10.0
    navigate_tolerance_m: float = 0.25
    navigate_timeout_s: float = 30.0
    # Перелёт: длина одного перегона по прямой к цели и режим навигации
    # ('map' - абсолютные точки в aruco_map, 'relative' - те же точки,
    # пересчитанные в смещение в body; см. flight_nav.fly_to).
    nav_mode: str = "map"
    leg_length_m: float = 1.0
    leg_timeout_s: float = 20.0
    arrival_tolerance_m: float = 0.25
    # Геозона: поле 7x7 плюс запас; за её пределами дрон уже вне зоны меток,
    # а дальше сетка (регламент: вылет за полётную зону - дисквалификация).
    geofence_margin_m: float = 0.5
    geofence_max_z_m: float = 3.0
    # Контроль достоверности локализации (flight_nav.read_localization):
    # насколько далеко aruco_map имеет право «отставать» от метки, которую
    # камера видит прямо сейчас, и сколько ждать восстановления.
    max_marker_distance_m: float = 2.5
    localization_timeout_s: float = 8.0
    localization_read_timeout_s: float = 1.0
    max_blind_time_s: float = 2.0
    # Допуск при сверке позиции со стартовой меткой после взлёта.
    start_position_tolerance_m: float = 1.0
    # Дрон стоит НА стартовой метке и закрывает её, поэтому на земле камера
    # может не видеть ничего - по умолчанию предполётная проверка только
    # печатает результат. Жёсткий режим (--require-preflight) отменяет взлёт.
    require_preflight_localization: bool = False
    # Курс в полёте не меняется (fc.HOLD_YAW = yaw=NaN, «сохранять текущий
    # курс»). Разворачиваться по yaw миссии незачем: камера смотрит вниз,
    # метки читаются при любой ориентации. Выключается --no-hold-yaw.
    hold_yaw: bool = True
    # Реальный тайминг регламента (Табл.1): 15с красная мигающая + 5с зелёная
    # перед взлётом. Переопределяется через --charge-wait для тестовых
    # попыток, если нужно ускорить итерацию.
    charge_wait_total_s: float = 15.0
    charge_green_before_takeoff_s: float = 5.0
    node_name: str = "bvs1_flight"

    def flight_limits(self) -> nav.FlightLimits:
        return nav.FlightLimits(
            leg_length_m=self.leg_length_m,
            speed=self.cruise_speed,
            arrival_tolerance_m=self.arrival_tolerance_m,
            navigate_tolerance_m=self.navigate_tolerance_m,
            navigate_timeout_s=self.leg_timeout_s,
            stabilize_tolerance_m=self.stabilize_tolerance_m,
            stabilize_hold_s=self.stabilize_hold_s,
            stabilize_timeout_s=self.stabilize_timeout_s,
            localization_timeout_s=self.localization_timeout_s,
            localization_read_timeout_s=self.localization_read_timeout_s,
            max_marker_distance_m=self.max_marker_distance_m,
            max_blind_time_s=self.max_blind_time_s,
        )

    def yaw(self) -> Optional[float]:
        return fc.HOLD_YAW if self.hold_yaw else None


def _confirm_localization(
    proxies: fc.FlightProxies,
    marker_id_reader: Callable[[float], Sequence[int]],
    markers: Dict[int, Tuple[float, float, float]],
    config: MissionConfig,
    *,
    stage: str,
    expected_xy: Optional[Tuple[float, float]] = None,
    sleep_fn: Callable[[float], None],
    time_fn: Callable[[], float],
    verbose: bool,
) -> nav.LocalizationStatus:
    """Подтвердить локализацию перед следующим этапом или прервать миссию.

    В отличие от прежней проверки «видна хоть одна метка», сверяет позицию
    aruco_map с реально видимыми метками (и, если задан ``expected_xy``, с
    ожидаемой точкой). Перед тем как бросить ошибку, гасит движение — дрон
    к этому моменту уже в воздухе.
    """
    try:
        status = nav.wait_for_localization(
            proxies.get_telemetry,
            marker_id_reader,
            markers,
            timeout=config.localization_timeout_s,
            expected_xy=expected_xy,
            expected_tolerance=config.start_position_tolerance_m,
            max_marker_distance=config.max_marker_distance_m,
            read_timeout=config.localization_read_timeout_s,
            sleep_fn=sleep_fn,
            time_fn=time_fn,
            verbose=verbose,
        )
    except nav.LocalizationLost as exc:
        fc.hover_in_place(proxies.navigate, speed=config.cruise_speed, verbose=verbose)
        raise nav.LocalizationLost("{}: {}".format(stage, exc)) from exc
    print("[{}] локализация: {}".format(stage, status.describe()))
    return status


def run_mission(
    proxies: fc.FlightProxies,
    markers: Dict[int, Tuple[float, float, float]],
    config: MissionConfig,
    *,
    rangefinder_reader: Callable[[float], float] = fc.read_rangefinder_ros,
    marker_id_reader: Callable[[float], Sequence[int]] = fc.read_visible_marker_ids_ros,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
    verbose: bool = False,
) -> None:
    """Выполнить полную миссию БВС-1. Не зависит от rospy напрямую — все
    ROS-вызовы приходят через ``proxies``, поэтому логику можно проверить
    самотестом с заглушками (см. ``_self_test``)."""
    start_x, start_y = fc.marker_xy(markers, config.start_marker_id)
    station_x, station_y = fc.marker_xy(markers, config.station_marker_id)
    fence = nav.Geofence.from_markers(
        markers, margin=config.geofence_margin_m, max_z=config.geofence_max_z_m
    )
    limits = config.flight_limits()
    if verbose:
        print(
            "[run_mission] старт=({:.2f}, {:.2f}) [метка {}], станция=({:.2f}, {:.2f}) "
            "[метка {}], высота станции={:.2f}м, крейсерская высота={:.2f}м, "
            "режим перелёта={!r}, перегон={:.2f}м, геозона: {}".format(
                start_x, start_y, config.start_marker_id,
                station_x, station_y, config.station_marker_id,
                config.station_height_m, config.cruise_altitude_m,
                config.nav_mode, config.leg_length_m, fence.describe(),
            )
        )

    # 0. Предполётная проверка на земле. Дрон стоит на стартовой метке и
    # закрывает её, поэтому камера может не видеть вообще ничего — это не
    # повод не взлетать (по умолчанию только печатаем). Но если метки видны
    # и aruco_map при этом показывает другой конец поля, взлетать уже нельзя:
    # карта на платформе не соответствует полю.
    preflight = nav.read_localization(
        proxies.get_telemetry,
        marker_id_reader,
        markers,
        expected_xy=(start_x, start_y),
        expected_tolerance=config.start_position_tolerance_m,
        max_marker_distance=config.max_marker_distance_m,
        read_timeout=config.localization_read_timeout_s,
        verbose=verbose,
    )
    print("[run_mission] предполётная локализация: {}".format(preflight.describe()))
    if not preflight.ok:
        if config.require_preflight_localization:
            raise nav.LocalizationLost(
                "предполётная проверка не пройдена: {}".format(preflight.reason)
            )
        if preflight.visible_ids:
            raise nav.LocalizationLost(
                "на земле видны метки {}, но позиция им противоречит: {}".format(
                    list(preflight.visible_ids), preflight.reason
                )
            )
        print(
            "[run_mission] на земле метки не видны (дрон закрывает стартовую "
            "метку) — локализация будет подтверждена после взлёта на {:.2f}м".format(
                config.check_altitude_m
            )
        )

    # 1. Взлёт на проверочную высоту — жёлтый мигающий (Табл.1 регламента).
    # frame_id='body' (не 'aruco_map'): сразу после включения aruco_map ещё
    # может быть не готов, поэтому первый подъём — вертикальный, в системе
    # координат самого дрона (как в примере TASK.md/CodeExamples.html). Дрон
    # уже стоит над start_marker_id, боковое смещение не нужно.
    if verbose:
        print("[run_mission] === Шаг 1: взлёт на проверочную высоту (frame_id=body) ===")
    led.set_led("blink", "yellow")
    fc.navigate_wait(
        proxies.navigate,
        proxies.get_telemetry,
        x=0.0,
        y=0.0,
        z=config.check_altitude_m,
        speed=config.approach_speed,
        frame_id="body",
        auto_arm=True,
        yaw=config.yaw(),
        tolerance=config.navigate_tolerance_m,
        timeout=config.navigate_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 2. Подтверждение локализации на малой высоте: позиция обязана совпасть
    # со стартовой меткой. Именно этой проверки не было в аварийной попытке —
    # дрон уходил в полёт по карте, ни разу не сверив, что карта показывает
    # то место, где он физически стоит.
    if verbose:
        print("[run_mission] === Шаг 2: сверка позиции со стартовой меткой ===")
    _confirm_localization(
        proxies,
        marker_id_reader,
        markers,
        config,
        stage="старт",
        expected_xy=(start_x, start_y),
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 3. Набор крейсерской высоты над стартовой меткой и повторная сверка —
    # на высоте меняется набор видимых меток, локализация может «переехать».
    if verbose:
        print("[run_mission] === Шаг 3: набор крейсерской высоты ===")
    fc.navigate_wait(
        proxies.navigate,
        proxies.get_telemetry,
        x=start_x,
        y=start_y,
        z=config.cruise_altitude_m,
        speed=config.approach_speed,
        frame_id="aruco_map",
        yaw=config.yaw(),
        tolerance=config.navigate_tolerance_m,
        timeout=config.navigate_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )
    fc.wait_until_stable(
        proxies.get_telemetry,
        x=start_x,
        y=start_y,
        frame_id="aruco_map",
        tolerance=config.stabilize_tolerance_m,
        hold_time=config.stabilize_hold_s,
        timeout=config.stabilize_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )
    _confirm_localization(
        proxies,
        marker_id_reader,
        markers,
        config,
        stage="крейсерская высота",
        expected_xy=(start_x, start_y),
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 4. Перелёт к станции — красный (Табл.1, «поиск зарядной станции»).
    # Летим по прямой к координатам станции из карты (метка под кубом не
    # видна), короткими перегонами с перепроверкой позиции перед каждым.
    if verbose:
        print("[run_mission] === Шаг 4: перелёт к станции ===")
    led.set_led("solid", "red")
    nav.fly_to(
        proxies.navigate,
        proxies.get_telemetry,
        marker_id_reader,
        markers,
        fence,
        target_x=station_x,
        target_y=station_y,
        z=config.cruise_altitude_m,
        mode=config.nav_mode,
        limits=limits,
        yaw=config.yaw(),
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 5. Стабилизация точно над станцией перед спуском
    if verbose:
        print("[run_mission] === Шаг 5: стабилизация над станцией ===")
    fc.navigate_wait(
        proxies.navigate,
        proxies.get_telemetry,
        x=station_x,
        y=station_y,
        z=config.cruise_altitude_m,
        speed=config.descent_speed,
        frame_id="aruco_map",
        yaw=config.yaw(),
        tolerance=config.navigate_tolerance_m,
        timeout=config.navigate_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )
    fc.wait_until_stable(
        proxies.get_telemetry,
        x=station_x,
        y=station_y,
        frame_id="aruco_map",
        tolerance=config.stabilize_tolerance_m,
        hold_time=config.stabilize_hold_s,
        timeout=config.stabilize_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 6. Управляемый спуск на куб зарядной станции + ручной дизарм по
    # дальномеру (не полагаемся на land() — см. докстринг модуля)
    if verbose:
        print("[run_mission] === Шаг 6: управляемый спуск на станцию ===")
    min_z = config.station_height_m + config.landing_safety_margin_m
    descent = fc.controlled_descent_and_disarm(
        proxies.navigate,
        proxies.arming,
        rangefinder_reader,
        x=station_x,
        y=station_y,
        start_z=config.cruise_altitude_m,
        min_z=min_z,
        step=config.descent_step_m,
        touchdown_threshold=config.touchdown_threshold_m,
        speed=config.descent_speed,
        frame_id="aruco_map",
        yaw=config.yaw(),
        sleep_fn=sleep_fn,
        verbose=verbose,
    )
    if not descent.touchdown_detected:
        # Дизарм в controlled_descent_and_disarm уже произошёл (по safety-
        # высоте min_z, не по факту касания) — но продолжать миссию вслепую
        # (имитация зарядки, повторный взлёт) на неподтверждённой посадке
        # нельзя. Останавливаем миссию явной ошибкой, как и остальные шаги
        # run_mission() (см. docstring diagnosis.py про разницу с run_mission).
        raise RuntimeError(
            "Касание станции не подтверждено дальномером (last_range={}); "
            "дизарм выполнен по safety-высоте {:.2f} м".format(
                descent.last_range, descent.final_z
            )
        )

    # 7. Имитация зарядки: заглушка вместо честного протокола станции —
    # общая длительность конфигурируется, но соотношение «зелёный за N секунд
    # до взлёта» сохранено (Табл.1).
    if verbose:
        print("[run_mission] === Шаг 7: имитация зарядки ===")
    fc.simulate_charging(
        led.set_led,
        total_s=config.charge_wait_total_s,
        green_before_takeoff_s=config.charge_green_before_takeoff_s,
        sleep_fn=sleep_fn,
    )

    # 8. Повторный взлёт с куба — снова вертикально в 'body', а не по карте:
    # на кубе камера в 80 см от пола видит мало меток, и aruco_map тут
    # наименее надёжен. Поднимаемся на (крейсерская - высота куба) над собой,
    # то есть на ту же крейсерскую высоту над полом.
    if verbose:
        print("[run_mission] === Шаг 8: повторный взлёт с куба ===")
    fc.navigate_wait(
        proxies.navigate,
        proxies.get_telemetry,
        x=0.0,
        y=0.0,
        z=max(config.cruise_altitude_m - config.station_height_m, config.check_altitude_m),
        speed=config.approach_speed,
        frame_id="body",
        auto_arm=True,
        yaw=config.yaw(),
        tolerance=config.navigate_tolerance_m,
        timeout=config.navigate_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )
    _confirm_localization(
        proxies,
        marker_id_reader,
        markers,
        config,
        stage="взлёт со станции",
        expected_xy=(station_x, station_y),
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 9. Возврат на исходную позицию — зелёный мигающий (Табл.1). Тот же
    # способ перелёта, что и к станции.
    if verbose:
        print("[run_mission] === Шаг 9: возврат на старт ===")
    led.set_led("blink", "green")
    nav.fly_to(
        proxies.navigate,
        proxies.get_telemetry,
        marker_id_reader,
        markers,
        fence,
        target_x=start_x,
        target_y=start_y,
        z=config.cruise_altitude_m,
        mode=config.nav_mode,
        limits=limits,
        yaw=config.yaw(),
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )
    fc.wait_until_stable(
        proxies.get_telemetry,
        x=start_x,
        y=start_y,
        frame_id="aruco_map",
        tolerance=config.stabilize_tolerance_m,
        hold_time=config.stabilize_hold_s,
        timeout=config.stabilize_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 10. Штатная посадка — под стартовой меткой ровный пол, куба нет
    if verbose:
        print("[run_mission] === Шаг 10: посадка на старте ===")
    fc.land_wait(
        proxies.land,
        proxies.get_telemetry,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )
    if verbose:
        print("[run_mission] === Миссия завершена ===")


def _self_test() -> int:
    import flight_test_support as fts

    recorded_leds, _led_backend = fts.record_led_calls()

    # Полная карта поля 7x7 - нужна и для геозоны, и для проверки
    # согласованности позиции с видимыми метками.
    markers = fc.read_map("config/field_map.txt")

    config = MissionConfig(
        start_marker_id=48,
        station_marker_id=37,
        station_height_m=0.8,
        cruise_altitude_m=1.5,
        check_altitude_m=1.0,
        stabilize_hold_s=0.5,
        stabilize_timeout_s=5.0,
        charge_wait_total_s=0.3,
        charge_green_before_takeoff_s=0.1,
    )

    for mode in ("map", "relative"):
        flight = fts.FakeFlight(start_x=6.0, start_y=6.0)
        clock = fts.FakeClock()
        run_mission(
            flight.proxies,
            markers,
            MissionConfig(**{**config.__dict__, "nav_mode": mode}),
            rangefinder_reader=fts.instant_touchdown_rangefinder,
            # камера видит метки под дроном - и на старте, и в пути
            marker_id_reader=fts.visible_marker_reader(flight, markers),
            sleep_fn=clock.sleep,
            time_fn=clock.time,
        )
        calls = flight.navigate_calls

        # Взлёт двумя ступенями: сначала проверочная высота в body, потом
        # крейсерская по карте - ошибочная локализация ловится на 1 м.
        assert calls[0]["frame_id"] == "body" and calls[0]["z"] == 1.0, mode
        assert calls[0]["auto_arm"] is True, mode
        assert calls[1]["frame_id"] == "aruco_map", mode
        assert (calls[1]["x"], calls[1]["y"], calls[1]["z"]) == (6.0, 6.0, 1.5), mode

        if mode == "map":
            # Перелёт идёт к станции (метка 37 -> (2, 5)) и обратно по одной и
            # той же прямой, а не по узлам решётки: каждая промежуточная точка
            # лежит на отрезке (6, 6) -> (2, 5).
            cruise_calls = [
                call for call in calls
                if call["frame_id"] == "aruco_map" and abs(call["z"] - 1.5) < 1e-9
            ]
            assert cruise_calls
            previous = (6.0, 6.0)
            for call in cruise_calls:
                cross = (call["x"] - 6.0) * (5.0 - 6.0) - (call["y"] - 6.0) * (2.0 - 6.0)
                assert abs(cross) < 1e-6, call
                # и ни один перегон не длиннее заданного шага
                step = ((call["x"] - previous[0]) ** 2 + (call["y"] - previous[1]) ** 2) ** 0.5
                assert step <= config.leg_length_m + 1e-6, call
                previous = (call["x"], call["y"])
        else:
            # В relative перегоны отдаются смещением в body — проверяем, что
            # каждое смещение короткое (карта в контуре управления не участвует)
            body_legs = [
                call for call in calls
                if call["frame_id"] == "body" and call["auto_arm"] is False
            ]
            assert body_legs
            for call in body_legs:
                step = (call["x"] ** 2 + call["y"] ** 2) ** 0.5
                assert step <= config.leg_length_m + 1e-6, call

        assert flight.disarm_calls == [False], mode  # дизарм на кубе станции
        # Повторный взлёт с куба - тоже вертикально в body, а не по карте
        takeoff_from_cube = [
            call for call in calls if call["frame_id"] == "body" and call["auto_arm"] is True
        ]
        assert len(takeoff_from_cube) == 2, mode
        # подъём на (крейсерская - высота куба), но не ниже проверочной высоты
        assert takeoff_from_cube[1]["z"] == max(1.5 - 0.8, 1.0), mode

        assert flight.state["armed"] is False, mode  # завершили штатным land_wait

        # Курс не меняется ни на одной команде: yaw=NaN («держать текущий»).
        # Без этого сервис подставляет yaw=0, что в кадре aruco_map означает
        # разворот носом по оси X карты — дрон крутился на каждом перегоне.
        assert all(math.isnan(call["yaw"]) for call in calls), mode

    # Запасной режим --no-hold-yaw: yaw не передаётся вовсе (на случай, если
    # прошивка не поймёт NaN), а не подставляется числом.
    assert MissionConfig(hold_yaw=False).yaw() is None
    assert math.isnan(MissionConfig().yaw())

    patterns = [entry for entry in recorded_leds if entry[0] in ("fill", "blink", "rainbow")]
    assert patterns[0] == ("blink", (255, 255, 0))  # взлёт
    assert patterns[1] == ("fill", (255, 0, 0))  # поиск станции
    assert patterns[2] == ("blink", (255, 0, 0))  # имитация зарядки (красная фаза)
    assert patterns[3] == ("fill", (0, 255, 0))  # зелёная лента перед взлётом (не мигающая — TASK.md)
    assert patterns[4] == ("blink", (0, 255, 0))  # возврат

    # Если дальномер так и не подтвердит касание, миссия должна остановиться
    # явной ошибкой вместо того, чтобы вслепую продолжить зарядку/взлёт на
    # неподтверждённой посадке (controlled_descent_and_disarm всё равно
    # дизармит по safety-высоте min_z — это лишь останавливает саму миссию).
    def never_touchdown_rangefinder(_timeout: float) -> float:
        return 5.0

    flight_no_touchdown = fts.FakeFlight(start_x=6.0, start_y=6.0)
    clock_no_touchdown = fts.FakeClock()
    try:
        run_mission(
            flight_no_touchdown.proxies,
            markers,
            config,
            rangefinder_reader=never_touchdown_rangefinder,
            marker_id_reader=fts.visible_marker_reader(flight_no_touchdown, markers),
            sleep_fn=clock_no_touchdown.sleep,
            time_fn=clock_no_touchdown.time,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Ожидался RuntimeError при неподтверждённом касании")

    # Локализация врёт: камера видит метку под дроном, а aruco_map уверенно
    # показывает другой конец поля. Прежний код взлетал и улетал по этой
    # позиции - теперь миссия обязана остановиться ещё на земле.
    flight_lying = fts.FakeFlight(start_x=0.0, start_y=0.0)  # карта «считает», что дрон в углу
    clock_lying = fts.FakeClock()
    try:
        run_mission(
            flight_lying.proxies,
            markers,
            config,
            rangefinder_reader=fts.instant_touchdown_rangefinder,
            marker_id_reader=fts.fixed_marker_reader(48),  # а видно метку 48 = (6, 6)
            sleep_fn=clock_lying.sleep,
            time_fn=clock_lying.time,
        )
    except nav.LocalizationLost:
        pass
    else:
        raise AssertionError("Ожидался LocalizationLost при расхождении карты с метками")
    assert flight_lying.navigate_calls == []  # ни одной команды, моторы не включались

    # Метка станции вне геозоны (опечатка в карте/аргументах) - перелёт
    # обязан отказаться, а не вести дрон за пределы поля.
    fence = nav.Geofence.from_markers(markers, margin=0.5)
    assert not fence.contains(7.5, 3.0)

    print("SELF-TEST: OK")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    defaults = MissionConfig()
    parser = argparse.ArgumentParser(description="Полёт БВС-1 (Энергоэстафета)")
    parser.add_argument("--map", default=defaults.map_path, help="путь к карте поля (field_map.txt)")
    parser.add_argument("--start-marker", type=int, default=defaults.start_marker_id)
    parser.add_argument("--station-marker", type=int, default=defaults.station_marker_id)
    parser.add_argument("--station-height", type=float, default=defaults.station_height_m)
    parser.add_argument("--cruise-altitude", type=float, default=defaults.cruise_altitude_m)
    parser.add_argument(
        "--check-altitude",
        type=float,
        default=defaults.check_altitude_m,
        help="высота первой ступени взлёта, на которой сверяется локализация",
    )
    parser.add_argument(
        "--nav-mode",
        choices=("map", "relative"),
        default=defaults.nav_mode,
        help="'map' — перегон задаётся точкой в aruco_map; 'relative' — тем же "
        "смещением в body (карта не участвует в контуре управления)",
    )
    parser.add_argument(
        "--leg-length",
        type=float,
        default=defaults.leg_length_m,
        help="длина одного перегона по прямой к цели, м",
    )
    parser.add_argument(
        "--cruise-speed",
        type=float,
        default=defaults.cruise_speed,
        help="скорость перелёта между перегонами, м/с",
    )
    parser.add_argument(
        "--geofence-margin",
        type=float,
        default=defaults.geofence_margin_m,
        help="запас геозоны за крайними метками поля, м",
    )
    parser.add_argument(
        "--max-marker-distance",
        type=float,
        default=defaults.max_marker_distance_m,
        help="насколько далеко позиция aruco_map может отстоять от видимой "
        "метки, прежде чем локализация считается недостоверной, м",
    )
    parser.add_argument(
        "--require-preflight",
        action="store_true",
        help="не взлетать, если локализация не подтверждена ещё на земле "
        "(по умолчанию дрон закрывает стартовую метку и на земле её не видит)",
    )
    parser.add_argument(
        "--no-hold-yaw",
        dest="hold_yaw",
        action="store_false",
        help="не передавать yaw в navigate() (прежнее поведение: дрон "
        "доворачивает носом по оси X карты и крутится в полёте) — запасной "
        "вариант, если прошивка не понимает yaw=NaN",
    )
    parser.add_argument("--charge-wait", type=float, default=defaults.charge_wait_total_s)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="проверить логику миссии без ROS 1/дрона",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="без подробного пошагового вывода (по умолчанию вывод включён)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.self_test:
        return _self_test()

    config = MissionConfig(
        map_path=args.map,
        start_marker_id=args.start_marker,
        station_marker_id=args.station_marker,
        station_height_m=args.station_height,
        cruise_altitude_m=args.cruise_altitude,
        check_altitude_m=args.check_altitude,
        nav_mode=args.nav_mode,
        leg_length_m=args.leg_length,
        cruise_speed=args.cruise_speed,
        geofence_margin_m=args.geofence_margin,
        max_marker_distance_m=args.max_marker_distance,
        require_preflight_localization=args.require_preflight,
        hold_yaw=args.hold_yaw,
        charge_wait_total_s=args.charge_wait,
    )

    markers = fc.read_map(config.map_path)
    proxies = fc.init_flight(config.node_name)
    led.use_technic_ros1_backend()
    try:
        try:
            run_mission(proxies, markers, config, verbose=not args.quiet)
        except Exception as exc:
            # Любая необработанная ошибка миссии (потеря локализации, выход за
            # геозону, таймаут навигации) раньше оставляла дрон висеть armed
            # без дальнейших команд - воспроизведено на площадке 2026-08-01.
            # Сначала гасим движение (команда navigate() асинхронная и иначе
            # продолжает выполняться), затем штатная посадка на месте: если
            # дрон уже у границы поля, лететь куда-то ещё опаснее, чем сесть.
            print(
                "[main] авария в run_mission ({}: {})".format(type(exc).__name__, exc)
            )
            try:
                telemetry = proxies.get_telemetry()
                airborne = bool(getattr(telemetry, "armed", True))
            except Exception:  # noqa: BLE001 — нет телеметрии, считаем что в воздухе
                airborne = True
            if not airborne:
                # Ошибка на земле (например, предполётная проверка): дрон не
                # armed, слать ему navigate()/land() незачем и вредно.
                print("[main] дрон не armed — команды не отправляются")
                raise
            print("[main] гашу движение и сажусь")
            try:
                fc.hover_in_place(proxies.navigate, verbose=not args.quiet)
                fc.land_wait(proxies.land, proxies.get_telemetry, verbose=not args.quiet)
            except Exception as land_exc:
                print(
                    "[main] аварийная посадка тоже не удалась ({}: {}) - "
                    "дрон может остаться armed, вмешаться вручную (RC/killswitch)".format(
                        type(land_exc).__name__, land_exc
                    )
                )
            raise
    finally:
        led.close_led_backend()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
