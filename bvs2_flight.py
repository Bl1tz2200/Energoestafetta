#!/usr/bin/env python3
"""Полёт БВС-2: взлёт → груз → своя зарядная станция → сброс груза → возврат.

Реализует шаги 1, 2, 4, 5, 6, 8 алгоритма из TASK.md: синхронный взлёт с
БВС-1 → полёт в зону захвата груза (метка 0) → захват зависанием на заданной
высоте → полёт со грузом к собственной зарядной станции БВС-2 (в этой
раскладке — метка 37, физически отдельный тёмно-синий куб 80 см, НЕ та же
станция, что использует БВС-1 на метке 5 — у каждого БВС своя станция) →
управляемая посадка на куб + имитация зарядки → сброс груза → повторный
взлёт → возврат на старт → штатная посадка.

Метка станции физически закрыта кубом, поэтому, как и в bvs1_flight.py,
навигация идёт по глобальной карте поля (``frame_id='aruco_map'``,
docs.skyris.ru/technic6S/ArucoMap.html), а не по прямому визуальному
слежению за меткой. Посадка на куб станции выполняется управляемым спуском
с контролем лазерного дальномера (``flight_core.controlled_descent_and_disarm``),
а не штатным ``land()`` — см. обоснование в bvs1_flight.py/TASK.md. Обычная
``land()`` используется только при возврате на стартовую метку, где под
дроном ровный пол.

Груз (метка 0) лежит на полу, а не на кубе — захват выполняется простым
зависанием на настраиваемой высоте ``pickup_altitude_m`` над меткой груза
(без контроля дальномера и без дизарма), затем ``gripper_close()``.

Расположение объектов на поле может меняться от попытки к попытке, но
известно заранее — поэтому id меток и высота куба вынесены в
``MissionConfig`` и параметры командной строки, а не зашиты в логику.
Стартовая метка БВС-2 по регламенту случайна и объявляется перед попыткой —
у ``--start-marker`` намеренно нет дефолта в CLI.

Запуск на дроне::

    python3 bvs2_flight.py --map config/field_map.txt --start-marker <N> \\
        --cargo-marker 0 --station-marker 37 --station-height 0.8 \\
        --gripper-pin <N> --gripper-open-pulse <N> --gripper-close-pulse <N>

Проверка модуля без ROS 1/дрона::

    python3 bvs2_flight.py --self-test
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
import gripper_control as gripper
import led_interface as led


@dataclass(frozen=True)
class GripperProxy:
    """DI-обёртка над gripper_control, по образцу fc.FlightProxies —
    позволяет проверить run_mission самотестом без реального захвата."""

    open: Callable[[], None]
    close: Callable[[], None]


@dataclass(frozen=True)
class MissionConfig:
    """Все настраиваемые параметры попытки — редактируются здесь/через CLI,
    без правки логики ``run_mission``."""

    map_path: str = "config/field_map.txt"
    start_marker_id: int = 0
    cargo_marker_id: int = 0
    station_marker_id: int = 37
    station_height_m: float = 0.8
    landing_safety_margin_m: float = 0.1
    cruise_altitude_m: float = 1.5
    pickup_altitude_m: float = 0.3
    approach_speed: float = 0.5
    descent_speed: float = 0.3
    descent_step_m: float = 0.15
    touchdown_threshold_m: float = 0.12
    stabilize_tolerance_m: float = 0.2
    stabilize_hold_s: float = 1.5
    stabilize_timeout_s: float = 10.0
    navigate_tolerance_m: float = 0.2
    navigate_timeout_s: float = 30.0
    # Сколько ждать после взлёта, пока камера не увидит хоть одну метку поля
    # (см. flight_core.wait_for_markers_visible) - до этого aruco_map
    # недостоверен, лететь по нему нельзя.
    markers_visible_timeout_s: float = 10.0
    gripper_close_delay_s: float = 0.5
    gripper_pin: int = 17
    gripper_open_pulse: int = 1000
    gripper_close_pulse: int = 2000
    # Реальный тайминг регламента (Табл.1), см. bvs1_flight.py: 15с красная
    # мигающая + 5с зелёная перед взлётом. Переопределяется через
    # --charge-wait для тестовых попыток, если нужно ускорить итерацию.
    charge_wait_total_s: float = 15.0
    charge_green_before_takeoff_s: float = 5.0
    # Курс в полёте не меняется (fc.HOLD_YAW = yaw=NaN) — как в bvs1_flight.py,
    # см. комментарий у самой константы. Выключается --no-hold-yaw.
    hold_yaw: bool = True
    node_name: str = "bvs2_flight"

    def yaw(self) -> Optional[float]:
        return fc.HOLD_YAW if self.hold_yaw else None


def run_mission(
    proxies: fc.FlightProxies,
    gripper_proxy: GripperProxy,
    markers: Dict[int, Tuple[float, float, float]],
    config: MissionConfig,
    *,
    rangefinder_reader: Callable[[float], float] = fc.read_rangefinder_ros,
    marker_id_reader: Callable[[float], Sequence[int]] = fc.read_visible_marker_ids_ros,
    sleep_fn: Callable[[float], None] = sleep,
    time_fn: Callable[[], float] = time.monotonic,
    verbose: bool = False,
) -> None:
    """Выполнить полную миссию БВС-2. Не зависит от rospy/gripper_control
    напрямую — все внешние вызовы приходят через ``proxies``/``gripper_proxy``,
    поэтому логику можно проверить самотестом с заглушками (см. ``_self_test``)."""
    start_x, start_y = fc.marker_xy(markers, config.start_marker_id)
    cargo_x, cargo_y = fc.marker_xy(markers, config.cargo_marker_id)
    station_x, station_y = fc.marker_xy(markers, config.station_marker_id)

    # 0. Гарантируем, что захват открыт до подлёта к грузу — после
    # предыдущей попытки или диагностики сервопривод мог остаться закрытым,
    # а первой аппаратной командой не должна быть close() над самим грузом.
    gripper_proxy.open()

    # 1. Взлёт с исходной точки — жёлтый мигающий (Табл.1 регламента).
    # frame_id='body' (не 'aruco_map'): сразу после включения aruco_map ещё
    # может быть не готов (TF появляется только после того, как дрон увидел
    # метки поля), поэтому первый взлёт — вертикальный подъём в собственной
    # системе координат дрона, как в примере TASK.md/CodeExamples.html
    # (navigate(x=0, y=0, z=1.5, frame_id='body', auto_arm=True)). Дрон уже
    # стоит над start_marker_id, так что боковое смещение не нужно — карта
    # поля используется начиная со следующего шага.
    if verbose:
        print("[run_mission] === Шаг 1: взлёт (frame_id=body) ===")
    led.set_led("blink", "yellow")
    fc.navigate_wait(
        proxies.navigate,
        proxies.get_telemetry,
        x=0.0,
        y=0.0,
        z=config.cruise_altitude_m,
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

    # 2. Подтверждение локализации по реально видимым меткам: сразу после
    # взлёта aruco_map может быть ещё недостоверен - см. подробное объяснение
    # и историю аварии на площадке в docstring
    # flight_core.wait_for_markers_visible/bvs1_flight.run_mission. Стартовая
    # метка БВС-2 случайна и объявляется перед попыткой, а дрон на ней может
    # стоять повёрнутым по yaw в любую сторону - привязка идёт по факту
    # видимых меток и их известным координатам из карты, а не по тому, куда
    # "смотрит" дрон, поэтому ориентация на старте роли не играет.
    if verbose:
        print("[run_mission] === Шаг 2: подтверждение локализации по видимым меткам ===")
    visible_marker_ids = fc.wait_for_markers_visible(
        marker_id_reader,
        timeout=config.markers_visible_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )
    print("[run_mission] видимые метки поля после взлёта: {}".format(list(visible_marker_ids)))

    telemetry_now = proxies.get_telemetry(frame_id="aruco_map")
    anchor_marker_id = fc.nearest_marker_id(
        markers, visible_marker_ids, telemetry_now.x, telemetry_now.y
    )
    anchor_x, anchor_y = fc.marker_xy(markers, anchor_marker_id)
    if verbose:
        print(
            "[run_mission] привязка к метке {} ({:.2f}, {:.2f})".format(
                anchor_marker_id, anchor_x, anchor_y
            )
        )
    fc.navigate_wait(
        proxies.navigate,
        proxies.get_telemetry,
        x=anchor_x,
        y=anchor_y,
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
        x=anchor_x,
        y=anchor_y,
        frame_id="aruco_map",
        tolerance=config.stabilize_tolerance_m,
        hold_time=config.stabilize_hold_s,
        timeout=config.stabilize_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 3. Полёт в зону захвата груза поочерёдно по меткам сетки, не
    # перепрыгивая их (flight_core.marker_path) — индикация не меняется,
    # жёлтый мигающий по тексту алгоритма покрывает и взлёт, и полёт к грузу.
    # Дрон крупный, поэтому каждый перегон летит на пониженной скорости
    # (descent_speed, не approach_speed) и завершается стабилизацией
    # (flight_core.fly_marker_path) - без резких перемещений между метками.
    if verbose:
        print("[run_mission] === Шаг 3: подлёт к грузу по меткам сетки (frame_id=aruco_map) ===")
    cargo_path = fc.marker_path(markers, anchor_marker_id, config.cargo_marker_id)
    if verbose:
        print("[run_mission] путь по меткам до груза: {}".format(list(cargo_path)))
    fc.fly_marker_path(
        proxies.navigate,
        proxies.get_telemetry,
        markers,
        cargo_path,
        z=config.cruise_altitude_m,
        speed=config.descent_speed,
        yaw=config.yaw(),
        navigate_tolerance=config.navigate_tolerance_m,
        navigate_timeout=config.navigate_timeout_s,
        stabilize_tolerance=config.stabilize_tolerance_m,
        stabilize_hold=config.stabilize_hold_s,
        stabilize_timeout=config.stabilize_timeout_s,
        marker_id_reader=marker_id_reader,
        markers_visible_timeout=config.markers_visible_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 4. Стабилизация над грузом перед снижением
    if verbose:
        print("[run_mission] === Шаг 4: стабилизация над грузом ===")
    fc.wait_until_stable(
        proxies.get_telemetry,
        x=cargo_x,
        y=cargo_y,
        frame_id="aruco_map",
        tolerance=config.stabilize_tolerance_m,
        hold_time=config.stabilize_hold_s,
        timeout=config.stabilize_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 5. Захват груза — красная лента, зависание на заданной высоте (груз на
    # полу, не на кубе — дальномер/дизарм не нужны), затем захват и подъём.
    if verbose:
        print("[run_mission] === Шаг 5: захват груза ===")
    led.set_led("solid", "red")
    fc.navigate_wait(
        proxies.navigate,
        proxies.get_telemetry,
        x=cargo_x,
        y=cargo_y,
        z=config.pickup_altitude_m,
        speed=config.descent_speed,
        frame_id="aruco_map",
        yaw=config.yaw(),
        tolerance=config.navigate_tolerance_m,
        timeout=config.navigate_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )
    gripper_proxy.close()
    if config.gripper_close_delay_s > 0:
        sleep_fn(config.gripper_close_delay_s)
    fc.navigate_wait(
        proxies.navigate,
        proxies.get_telemetry,
        x=cargo_x,
        y=cargo_y,
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

    # 6. Полёт к собственной зарядной станции БВС-2 с грузом поочерёдно по
    # меткам сетки, не перепрыгивая их (flight_core.marker_path) — индикация
    # не меняется (красная лента, по тексту шага 5 алгоритма). Дрон крупный,
    # поэтому, как и на пути к грузу в шаге 3, каждый перегон летит на
    # пониженной скорости и завершается стабилизацией
    # (flight_core.fly_marker_path), а не просто прибытием в допуск
    # navigate_wait.
    if verbose:
        print("[run_mission] === Шаг 6: подлёт к станции по меткам сетки (frame_id=aruco_map) ===")
    station_path = fc.marker_path(markers, config.cargo_marker_id, config.station_marker_id)
    if verbose:
        print("[run_mission] путь по меткам до станции: {}".format(list(station_path)))
    fc.fly_marker_path(
        proxies.navigate,
        proxies.get_telemetry,
        markers,
        station_path,
        z=config.cruise_altitude_m,
        speed=config.descent_speed,
        yaw=config.yaw(),
        navigate_tolerance=config.navigate_tolerance_m,
        navigate_timeout=config.navigate_timeout_s,
        stabilize_tolerance=config.stabilize_tolerance_m,
        stabilize_hold=config.stabilize_hold_s,
        stabilize_timeout=config.stabilize_timeout_s,
        marker_id_reader=marker_id_reader,
        markers_visible_timeout=config.markers_visible_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 7. Стабилизация и управляемый спуск на куб станции + ручной дизарм по
    # дальномеру (не полагаемся на land() — см. bvs1_flight.py/TASK.md)
    if verbose:
        print("[run_mission] === Шаг 7: стабилизация и спуск на станцию ===")
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
        # (сброс груза, повторный взлёт) на неподтверждённой посадке нельзя.
        raise RuntimeError(
            "Касание станции не подтверждено дальномером (last_range={}); "
            "дизарм выполнен по safety-высоте {:.2f} м".format(
                descent.last_range, descent.final_z
            )
        )

    # 8. Имитация зарядки: заглушка вместо честных 15с/5с (Табл.1) — общая
    # длительность конфигурируется, соотношение «зелёный за N секунд до
    # взлёта» сохранено (см. bvs1_flight.py/flight_core.simulate_charging).
    if verbose:
        print("[run_mission] === Шаг 8: имитация зарядки ===")
    fc.simulate_charging(
        led.set_led,
        total_s=config.charge_wait_total_s,
        green_before_takeoff_s=config.charge_green_before_takeoff_s,
        sleep_fn=sleep_fn,
    )

    # 9. Сброс груза — после имитации зарядки, перед повторным взлётом
    # (порядок из текста шага 6 алгоритма).
    if verbose:
        print("[run_mission] === Шаг 9: сброс груза ===")
    gripper_proxy.open()

    # 10. Повторный взлёт с куба станции до крейсерской высоты
    if verbose:
        print("[run_mission] === Шаг 10: повторный взлёт с куба ===")
    fc.navigate_wait(
        proxies.navigate,
        proxies.get_telemetry,
        x=station_x,
        y=station_y,
        z=config.cruise_altitude_m,
        speed=config.approach_speed,
        frame_id="aruco_map",
        auto_arm=True,
        yaw=config.yaw(),
        tolerance=config.navigate_tolerance_m,
        timeout=config.navigate_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 11. Возврат на исходную позицию поочерёдно по меткам сетки — зелёный
    # мигающий (та же логика "не перепрыгивать метки" и плавных перегонов со
    # стабилизацией, что и на пути к станции в шаге 6).
    if verbose:
        print("[run_mission] === Шаг 11: возврат на старт по меткам сетки ===")
    led.set_led("blink", "green")
    return_path = fc.marker_path(markers, config.station_marker_id, config.start_marker_id)
    if verbose:
        print("[run_mission] путь по меткам на старт: {}".format(list(return_path)))
    fc.fly_marker_path(
        proxies.navigate,
        proxies.get_telemetry,
        markers,
        return_path,
        z=config.cruise_altitude_m,
        speed=config.descent_speed,
        yaw=config.yaw(),
        navigate_tolerance=config.navigate_tolerance_m,
        navigate_timeout=config.navigate_timeout_s,
        stabilize_tolerance=config.stabilize_tolerance_m,
        stabilize_hold=config.stabilize_hold_s,
        stabilize_timeout=config.stabilize_timeout_s,
        marker_id_reader=marker_id_reader,
        markers_visible_timeout=config.markers_visible_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 12. Штатная посадка — под стартовой меткой ровный пол, куба нет
    if verbose:
        print("[run_mission] === Шаг 12: посадка на старте ===")
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

    # Полная карта поля 7x7 (не три точки) - нужна, чтобы marker_path() мог
    # растеризовать путь по промежуточным меткам решётки между стартом,
    # грузом и станцией, а не только знать сами эти три точки.
    markers = fc.read_map("config/field_map.txt")

    flight = fts.FakeFlight(start_x=6.0, start_y=2.0)
    navigate_calls = flight.navigate_calls
    disarm_calls = flight.disarm_calls

    gripper_calls = []
    gripper_proxy = GripperProxy(
        open=lambda: gripper_calls.append("open"),
        close=lambda: gripper_calls.append("close"),
    )

    clock = fts.FakeClock()

    config = MissionConfig(
        start_marker_id=20,
        cargo_marker_id=0,
        station_marker_id=5,
        station_height_m=0.8,
        cruise_altitude_m=1.5,
        pickup_altitude_m=0.3,
        stabilize_hold_s=0.5,
        stabilize_timeout_s=5.0,
        gripper_close_delay_s=0.1,
        charge_wait_total_s=0.3,
        charge_green_before_takeoff_s=0.1,
    )

    # Камера "видит" ровно стартовую метку 20 сразу после взлёта - этого
    # достаточно, чтобы подтвердить локализацию (см. flight_core.
    # wait_for_markers_visible/nearest_marker_id) без реального ROS 1.
    run_mission(
        flight.proxies,
        gripper_proxy,
        markers,
        config,
        rangefinder_reader=fts.instant_touchdown_rangefinder,
        marker_id_reader=fts.fixed_marker_reader(20),
        sleep_fn=clock.sleep,
        time_fn=clock.time,
    )

    # Захват открыт заранее, до подлёта к грузу (защита от предыдущего
    # состояния сервопривода)
    assert gripper_calls[0] == "open"

    # Взлёт со старта
    assert navigate_calls[0]["x"] == 0.0 and navigate_calls[0]["y"] == 0.0
    assert navigate_calls[0]["auto_arm"] is True
    assert navigate_calls[0]["frame_id"] == "body"  # взлёт: TF aruco_map ещё может быть не готов

    # Привязка локализации к видимой стартовой метке (20, (6, 2))
    assert navigate_calls[1]["x"] == 6.0 and navigate_calls[1]["y"] == 2.0
    assert navigate_calls[1]["auto_arm"] is False

    # Путь до груза идёт по соседним узлам решётки поля (20 -> 19 -> 11 ->
    # 10 -> 9 -> 1 -> 0), не перепрыгивая метки одним диагональным прыжком.
    cargo_leg_xy = [(call["x"], call["y"]) for call in navigate_calls[1:8]]
    assert cargo_leg_xy == [
        (6.0, 2.0), (5.0, 2.0), (4.0, 1.0), (3.0, 1.0), (2.0, 1.0), (1.0, 0.0), (0.0, 0.0),
    ]
    assert navigate_calls[7]["z"] == 1.5  # крейсерская высота над грузом

    # Снижение к грузу для захвата
    assert navigate_calls[8]["x"] == 0.0 and navigate_calls[8]["y"] == 0.0
    assert navigate_calls[8]["z"] == 0.3

    # Захват груза произошёл между снижением и подъёмом обратно
    assert gripper_calls[1] == "close"

    # Подъём обратно над грузом
    assert navigate_calls[9]["x"] == 0.0 and navigate_calls[9]["z"] == 1.5

    # Путь к станции с грузом — тоже по соседним меткам решётки (0 -> 1 -> 2
    # -> 3 -> 4 -> 5), без диагональных прыжков.
    station_leg_xy = [(call["x"], call["y"]) for call in navigate_calls[10:15]]
    assert station_leg_xy == [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0)]

    # Управляемый спуск на станцию завершился дизармом
    assert disarm_calls == [False]

    # Сброс груза произошёл после зарядки, до повторного взлёта
    assert gripper_calls == ["open", "close", "open"]

    # Повторный взлёт со станции
    assert navigate_calls[16]["x"] == 5.0 and navigate_calls[16]["y"] == 0.0
    assert navigate_calls[16]["auto_arm"] is True

    # Возврат на старт — по соседним меткам решётки (5 -> 13 -> 20)
    return_leg_xy = [(call["x"], call["y"]) for call in navigate_calls[17:]]
    assert return_leg_xy == [(6.0, 1.0), (6.0, 2.0)]
    assert navigate_calls[-1]["auto_arm"] is False

    assert flight.state["armed"] is False  # завершили штатным land_wait

    # Курс не меняется ни на одной команде: yaw=NaN («держать текущий»). Без
    # этого сервис подставляет yaw=0, что в кадре aruco_map означает разворот
    # носом по оси X карты — дрон крутился на каждом перегоне (как и в БВС-1).
    assert all(math.isnan(call["yaw"]) for call in navigate_calls)
    # Запасной режим --no-hold-yaw: yaw не передаётся вовсе, а не числом
    assert MissionConfig(hold_yaw=False).yaw() is None
    assert math.isnan(MissionConfig().yaw())

    patterns = [entry for entry in recorded_leds if entry[0] in ("fill", "blink", "rainbow")]
    assert patterns[0] == ("blink", (255, 255, 0))  # взлёт / полёт к грузу
    assert patterns[1] == ("fill", (255, 0, 0))  # захват груза
    assert patterns[2] == ("blink", (255, 0, 0))  # имитация зарядки (красная фаза)
    assert patterns[3] == ("fill", (0, 255, 0))  # зелёная лента перед взлётом (не мигающая — TASK.md)
    assert patterns[4] == ("blink", (0, 255, 0))  # возврат

    # Если дальномер так и не подтвердит касание станции, миссия должна
    # остановиться явной ошибкой, а не продолжить сброс груза/взлёт вслепую.
    def never_touchdown_rangefinder(_timeout: float) -> float:
        return 5.0

    flight_no_touchdown = fts.FakeFlight(start_x=6.0, start_y=2.0)
    gripper_proxy_no_touchdown = GripperProxy(
        open=lambda: None,
        close=lambda: None,
    )
    clock_no_touchdown = fts.FakeClock()
    try:
        run_mission(
            flight_no_touchdown.proxies,
            gripper_proxy_no_touchdown,
            markers,
            config,
            rangefinder_reader=never_touchdown_rangefinder,
            marker_id_reader=fts.fixed_marker_reader(20),
            sleep_fn=clock_no_touchdown.sleep,
            time_fn=clock_no_touchdown.time,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Ожидался RuntimeError при неподтверждённом касании")

    print("SELF-TEST: OK")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    defaults = MissionConfig()
    parser = argparse.ArgumentParser(description="Полёт БВС-2 (Энергоэстафета)")
    parser.add_argument("--map", default=defaults.map_path, help="путь к карте поля (field_map.txt)")
    parser.add_argument(
        "--start-marker",
        type=int,
        default=None,
        help=(
            "метка старта БВС-2 (по регламенту случайна, объявляется перед "
            "попыткой); обязательна для реального запуска, не нужна для --self-test"
        ),
    )
    parser.add_argument("--cargo-marker", type=int, default=defaults.cargo_marker_id)
    parser.add_argument("--station-marker", type=int, default=defaults.station_marker_id)
    parser.add_argument("--station-height", type=float, default=defaults.station_height_m)
    parser.add_argument("--cruise-altitude", type=float, default=defaults.cruise_altitude_m)
    parser.add_argument("--pickup-altitude", type=float, default=defaults.pickup_altitude_m)
    parser.add_argument("--charge-wait", type=float, default=defaults.charge_wait_total_s)
    parser.add_argument("--gripper-pin", type=int, default=defaults.gripper_pin)
    parser.add_argument("--gripper-open-pulse", type=int, default=defaults.gripper_open_pulse)
    parser.add_argument("--gripper-close-pulse", type=int, default=defaults.gripper_close_pulse)
    parser.add_argument(
        "--no-hold-yaw",
        dest="hold_yaw",
        action="store_false",
        help="не передавать yaw в navigate() (прежнее поведение: дрон "
        "доворачивает носом по оси X карты и крутится в полёте) — запасной "
        "вариант, если прошивка не понимает yaw=NaN",
    )
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
    parser = _build_parser()
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if args.start_marker is None:
        parser.error("--start-marker обязателен (кроме --self-test)")

    config = MissionConfig(
        map_path=args.map,
        start_marker_id=args.start_marker,
        cargo_marker_id=args.cargo_marker,
        station_marker_id=args.station_marker,
        station_height_m=args.station_height,
        cruise_altitude_m=args.cruise_altitude,
        pickup_altitude_m=args.pickup_altitude,
        charge_wait_total_s=args.charge_wait,
        gripper_pin=args.gripper_pin,
        gripper_open_pulse=args.gripper_open_pulse,
        gripper_close_pulse=args.gripper_close_pulse,
        hold_yaw=args.hold_yaw,
    )

    markers = fc.read_map(config.map_path)
    proxies = fc.init_flight(config.node_name)
    led.use_technic_ros1_backend()
    backend = gripper.use_technic_gpio_gripper(
        config.gripper_pin,
        open_pulse=config.gripper_open_pulse,
        close_pulse=config.gripper_close_pulse,
    )
    gripper_proxy = GripperProxy(
        open=lambda: gripper.gripper_open(),
        close=lambda: gripper.gripper_close(),
    )
    try:
        try:
            run_mission(proxies, gripper_proxy, markers, config)
        except Exception as exc:
            # См. bvs1_flight.py: необработанная ошибка миссии раньше
            # оставляла дрон armed без дальнейших команд - вместо этого
            # пробуем штатную посадку, прежде чем пробросить исключение.
            print(
                "[main] авария в run_mission ({}: {}) - аварийная посадка".format(
                    type(exc).__name__, exc
                )
            )
            try:
                fc.land_wait(proxies.land, proxies.get_telemetry)
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
        backend.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
