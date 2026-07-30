#!/usr/bin/env python3
"""Полёт БВС-1: взлёт → зарядная станция на кубе → ожидание → возврат.

Реализует шаги 1, 3, 5 алгоритма из TASK.md и уточнение пользователя:
зарядная станция стоит на тёмно-синем кубе 80 см и закрывает собой ArUco-метку
(в этой раскладке — id 37), поэтому лететь «по видимой метке станции»
нельзя — навигация идёт по глобальной карте поля (``frame_id='aruco_map'``,
docs.skyris.ru/technic6S/ArucoMap.html), а не по прямому визуальному
слежению за одной меткой. Посадка на куб выполняется управляемым спуском с
контролем лазерного дальномера (``flight_core.controlled_descent_and_disarm``),
а не штатным ``land()`` — его поведение на приподнятой поверхности не описано
в документации Skyris. Обычная ``land()`` используется только при возврате на
стартовую метку, где под дроном ровный пол.

Расположение объектов на поле может меняться от попытки к попытке, но
известно заранее — поэтому id меток и высота куба вынесены в ``MissionConfig``
и параметры командной строки, а не зашиты в логику.

Запуск на дроне::

    python3 bvs1_flight.py --map field_map.txt --start-marker 48 \\
        --station-marker 37 --station-height 0.8

Проверка модуля без ROS 1/дрона::

    python3 bvs1_flight.py --self-test
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from time import sleep
from typing import Callable, Dict, Tuple

import flight_core as fc
import led_interface as led


@dataclass(frozen=True)
class MissionConfig:
    """Все настраиваемые параметры попытки — редактируются здесь/через CLI,
    без правки логики ``run_mission``."""

    map_path: str = "field_map.txt"
    start_marker_id: int = 48
    station_marker_id: int = 37
    station_height_m: float = 0.8
    landing_safety_margin_m: float = 0.1
    cruise_altitude_m: float = 1.5
    approach_speed: float = 0.5
    descent_speed: float = 0.3
    descent_step_m: float = 0.15
    touchdown_threshold_m: float = 0.12
    stabilize_tolerance_m: float = 0.2
    stabilize_hold_s: float = 1.5
    stabilize_timeout_s: float = 10.0
    navigate_tolerance_m: float = 0.2
    navigate_timeout_s: float = 30.0
    # По просьбе пользователя — 30с вместо честных 15с/5с зарядки, чтобы
    # тестовые попытки шли быстрее. Для зачётной попытки заменить на реальный
    # тайминг регламента (15с красная мигающая + 5с зелёная перед взлётом).
    charge_wait_total_s: float = 30.0
    charge_green_before_takeoff_s: float = 5.0
    node_name: str = "bvs1_flight"


def run_mission(
    proxies: fc.FlightProxies,
    markers: Dict[int, Tuple[float, float, float]],
    config: MissionConfig,
    *,
    rangefinder_reader: Callable[[float], float] = fc.read_rangefinder_ros,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
) -> None:
    """Выполнить полную миссию БВС-1. Не зависит от rospy напрямую — все
    ROS-вызовы приходят через ``proxies``, поэтому логику можно проверить
    самотестом с заглушками (см. ``_self_test``)."""
    start_x, start_y = fc.marker_xy(markers, config.start_marker_id)
    station_x, station_y = fc.marker_xy(markers, config.station_marker_id)

    # 1. Взлёт с исходной точки — жёлтый мигающий (Табл.1 регламента)
    led.set_led("blink", "yellow")
    fc.navigate_wait(
        proxies.navigate,
        proxies.get_telemetry,
        x=start_x,
        y=start_y,
        z=config.cruise_altitude_m,
        speed=config.approach_speed,
        frame_id="aruco_map",
        auto_arm=True,
        tolerance=config.navigate_tolerance_m,
        timeout=config.navigate_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )

    # 2. Поиск/подлёт к зарядной станции — красный. Летим по координатам из
    # карты поля, а не по видимой метке: метка станции физически закрыта.
    led.set_led("solid", "red")
    fc.navigate_wait(
        proxies.navigate,
        proxies.get_telemetry,
        x=station_x,
        y=station_y,
        z=config.cruise_altitude_m,
        speed=config.approach_speed,
        frame_id="aruco_map",
        tolerance=config.navigate_tolerance_m,
        timeout=config.navigate_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )

    # 3. Стабилизация над станцией перед спуском
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
    )

    # 4. Управляемый спуск на куб зарядной станции + ручной дизарм по
    # дальномеру (не полагаемся на land() — см. докстринг модуля)
    min_z = config.station_height_m + config.landing_safety_margin_m
    fc.controlled_descent_and_disarm(
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
        sleep_fn=sleep_fn,
    )

    # 5. Имитация зарядки: заглушка вместо честных 15с/5с (Табл.1) — общая
    # длительность configурируется, но соотношение «зелёный за N секунд до
    # взлёта» сохранено.
    fc.simulate_charging(
        led.set_led,
        total_s=config.charge_wait_total_s,
        green_before_takeoff_s=config.charge_green_before_takeoff_s,
        sleep_fn=sleep_fn,
    )

    # 6. Повторный взлёт с куба до крейсерской высоты
    fc.navigate_wait(
        proxies.navigate,
        proxies.get_telemetry,
        x=station_x,
        y=station_y,
        z=config.cruise_altitude_m,
        speed=config.approach_speed,
        frame_id="aruco_map",
        auto_arm=True,
        tolerance=config.navigate_tolerance_m,
        timeout=config.navigate_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )

    # 7. Возврат на исходную позицию — зелёный мигающий
    led.set_led("blink", "green")
    fc.navigate_wait(
        proxies.navigate,
        proxies.get_telemetry,
        x=start_x,
        y=start_y,
        z=config.cruise_altitude_m,
        speed=config.approach_speed,
        frame_id="aruco_map",
        tolerance=config.navigate_tolerance_m,
        timeout=config.navigate_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )

    # 8. Штатная посадка — под стартовой меткой ровный пол, куба нет
    fc.land_wait(
        proxies.land,
        proxies.get_telemetry,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )


def _self_test() -> int:
    import flight_test_support as fts

    recorded_leds, _led_backend = fts.record_led_calls()

    markers = {48: (6.0, 6.0, 0.0), 37: (2.0, 5.0, 0.0)}

    flight = fts.FakeFlight(start_x=6.0, start_y=6.0)
    navigate_calls = flight.navigate_calls
    disarm_calls = flight.disarm_calls

    clock = fts.FakeClock()

    config = MissionConfig(
        start_marker_id=48,
        station_marker_id=37,
        station_height_m=0.8,
        cruise_altitude_m=1.5,
        stabilize_hold_s=0.5,
        stabilize_timeout_s=5.0,
        charge_wait_total_s=0.3,
        charge_green_before_takeoff_s=0.1,
    )

    run_mission(
        flight.proxies,
        markers,
        config,
        rangefinder_reader=fts.instant_touchdown_rangefinder,
        sleep_fn=clock.sleep,
        time_fn=clock.time,
    )

    assert navigate_calls[0]["x"] == 6.0 and navigate_calls[0]["y"] == 6.0
    assert navigate_calls[0]["auto_arm"] is True
    assert navigate_calls[0]["frame_id"] == "aruco_map"

    assert navigate_calls[1]["x"] == 2.0 and navigate_calls[1]["y"] == 5.0
    assert navigate_calls[1]["auto_arm"] is False

    assert disarm_calls == [False]

    assert navigate_calls[-2]["x"] == 2.0 and navigate_calls[-2]["y"] == 5.0
    assert navigate_calls[-2]["auto_arm"] is True  # повторный взлёт с куба
    assert navigate_calls[-1]["x"] == 6.0 and navigate_calls[-1]["y"] == 6.0

    assert flight.state["armed"] is False  # завершили штатным land_wait

    patterns = [entry for entry in recorded_leds if entry[0] in ("fill", "blink", "rainbow")]
    assert patterns[0] == ("blink", (255, 255, 0))  # взлёт
    assert patterns[1] == ("fill", (255, 0, 0))  # поиск станции
    assert patterns[2] == ("blink", (255, 0, 0))  # имитация зарядки
    assert patterns[3] == ("blink", (0, 255, 0))  # зелёный перед взлётом
    assert patterns[4] == ("blink", (0, 255, 0))  # возврат

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
    parser.add_argument("--charge-wait", type=float, default=defaults.charge_wait_total_s)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="проверить логику миссии без ROS 1/дрона",
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
        charge_wait_total_s=args.charge_wait,
    )

    markers = fc.read_map(config.map_path)
    proxies = fc.init_flight(config.node_name)
    led.use_technic_ros1_backend()
    try:
        run_mission(proxies, markers, config)
    finally:
        led.close_led_backend()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
