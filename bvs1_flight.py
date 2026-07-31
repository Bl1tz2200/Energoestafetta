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

    python3 bvs1_flight.py --map config/field_map.txt --start-marker 48 \\
        --station-marker 37 --station-height 0.8

Проверка модуля без ROS 1/дрона::

    python3 bvs1_flight.py --self-test
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Callable, Dict, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import flight_core as fc
import led_interface as led


@dataclass(frozen=True)
class MissionConfig:
    """Все настраиваемые параметры попытки — редактируются здесь/через CLI,
    без правки логики ``run_mission``."""

    map_path: str = "config/field_map.txt"
    start_marker_id: int = 48
    station_marker_id: int = 37
    station_height_m: float = 0.8
    landing_safety_margin_m: float = 0.1
    cruise_altitude_m: float = 2.0
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
    # Реальный тайминг регламента (Табл.1): 15с красная мигающая + 5с зелёная
    # перед взлётом. Переопределяется через --charge-wait для тестовых
    # попыток, если нужно ускорить итерацию.
    charge_wait_total_s: float = 15.0
    charge_green_before_takeoff_s: float = 5.0
    node_name: str = "bvs1_flight"


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
    if verbose:
        print(
            "[run_mission] старт=({:.2f}, {:.2f}) [метка {}], станция=({:.2f}, {:.2f}) "
            "[метка {}], высота станции={:.2f}м, крейсерская высота={:.2f}м".format(
                start_x, start_y, config.start_marker_id,
                station_x, station_y, config.station_marker_id,
                config.station_height_m, config.cruise_altitude_m,
            )
        )

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
        tolerance=config.navigate_tolerance_m,
        timeout=config.navigate_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 2. Подтверждение локализации по реально видимым меткам: сразу после
    # взлёта aruco_map может быть ещё недостоверен (TF готов только после
    # того, как дрон реально увидел метку) - воспроизведённая на площадке
    # авария (см. docstring flight_core.wait_for_markers_visible) случилась
    # именно из-за навигации по aruco_map вслепую. Дрон при этом может стоять
    # на старте повёрнутым по yaw в любую сторону - это не проблема: и
    # aruco_map, и данный шаг ориентируются по факту видимых меток/их
    # известным координатам из карты, а не по тому, куда "смотрит" дрон.
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

    # 3. Подлёт к зарядной станции поочерёдно по меткам сетки поля, не
    # перепрыгивая через них (flight_core.marker_path) - летим по координатам
    # из карты поля, а не по видимой метке станции: она физически закрыта.
    # Дрон крупный, поэтому каждый перегон летит на пониженной скорости
    # (descent_speed, не approach_speed) и завершается стабилизацией
    # (flight_core.fly_marker_path) - без резких перемещений между метками.
    if verbose:
        print("[run_mission] === Шаг 3: подлёт к станции по меткам сетки (frame_id=aruco_map) ===")
    led.set_led("solid", "red")
    station_path = fc.marker_path(markers, anchor_marker_id, config.station_marker_id)
    if verbose:
        print("[run_mission] путь по меткам до станции: {}".format(list(station_path)))
    fc.fly_marker_path(
        proxies.navigate,
        proxies.get_telemetry,
        markers,
        station_path,
        z=config.cruise_altitude_m,
        speed=config.descent_speed,
        navigate_tolerance=config.navigate_tolerance_m,
        navigate_timeout=config.navigate_timeout_s,
        stabilize_tolerance=config.stabilize_tolerance_m,
        stabilize_hold=config.stabilize_hold_s,
        stabilize_timeout=config.stabilize_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 4. Стабилизация над станцией перед спуском
    if verbose:
        print("[run_mission] === Шаг 4: стабилизация над станцией ===")
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

    # 5. Управляемый спуск на куб зарядной станции + ручной дизарм по
    # дальномеру (не полагаемся на land() — см. докстринг модуля)
    if verbose:
        print("[run_mission] === Шаг 5: управляемый спуск на станцию ===")
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

    # 6. Имитация зарядки: заглушка вместо честных 15с/5с (Табл.1) — общая
    # длительность configурируется, но соотношение «зелёный за N секунд до
    # взлёта» сохранено.
    if verbose:
        print("[run_mission] === Шаг 6: имитация зарядки ===")
    fc.simulate_charging(
        led.set_led,
        total_s=config.charge_wait_total_s,
        green_before_takeoff_s=config.charge_green_before_takeoff_s,
        sleep_fn=sleep_fn,
    )

    # 7. Повторный взлёт с куба до крейсерской высоты
    if verbose:
        print("[run_mission] === Шаг 7: повторный взлёт с куба ===")
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
        verbose=verbose,
    )

    # 8. Возврат на исходную позицию поочерёдно по меткам сетки — зелёный
    # мигающий (та же логика "не перепрыгивать метки" и плавных перегонов со
    # стабилизацией, что и на пути к станции в шаге 3).
    if verbose:
        print("[run_mission] === Шаг 8: возврат на старт по меткам сетки ===")
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
        navigate_tolerance=config.navigate_tolerance_m,
        navigate_timeout=config.navigate_timeout_s,
        stabilize_tolerance=config.stabilize_tolerance_m,
        stabilize_hold=config.stabilize_hold_s,
        stabilize_timeout=config.stabilize_timeout_s,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        verbose=verbose,
    )

    # 9. Штатная посадка — под стартовой меткой ровный пол, куба нет
    if verbose:
        print("[run_mission] === Шаг 9: посадка на старте ===")
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

    # Полная карта поля 7x7 (не пара точек) - нужна, чтобы marker_path() мог
    # растеризовать путь по промежуточным меткам решётки между стартом (48)
    # и станцией (37), а не только знать сами эти две точки.
    markers = fc.read_map("config/field_map.txt")

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

    # Камера "видит" ровно стартовую метку 48 сразу после взлёта - этого
    # достаточно, чтобы подтвердить локализацию (см. flight_core.
    # wait_for_markers_visible/nearest_marker_id) без реального ROS 1.
    run_mission(
        flight.proxies,
        markers,
        config,
        rangefinder_reader=fts.instant_touchdown_rangefinder,
        marker_id_reader=fts.fixed_marker_reader(48),
        sleep_fn=clock.sleep,
        time_fn=clock.time,
    )

    assert navigate_calls[0]["x"] == 0.0 and navigate_calls[0]["y"] == 0.0
    assert navigate_calls[0]["auto_arm"] is True
    assert navigate_calls[0]["frame_id"] == "body"  # взлёт: TF aruco_map ещё может быть не готов

    # Привязка локализации к видимой стартовой метке (48, (6, 6))
    assert navigate_calls[1]["x"] == 6.0 and navigate_calls[1]["y"] == 6.0
    assert navigate_calls[1]["auto_arm"] is False

    # Путь до станции идёт по соседним узлам решётки поля (48 -> 47 -> 39 ->
    # 38 -> 37), не перепрыгивая метки одним диагональным прыжком.
    station_leg_xy = [(call["x"], call["y"]) for call in navigate_calls[1:6]]
    assert station_leg_xy == [(6.0, 6.0), (5.0, 6.0), (4.0, 5.0), (3.0, 5.0), (2.0, 5.0)]

    assert disarm_calls == [False]

    assert navigate_calls[7]["x"] == 2.0 and navigate_calls[7]["y"] == 5.0
    assert navigate_calls[7]["auto_arm"] is True  # повторный взлёт с куба

    # Возврат на старт — тоже по соседним узлам решётки, ни одной метки не
    # перепрыгивая (растеризация в обратную сторону не обязана быть зеркальной
    # копией прямого пути - это тот же алгоритм Брезенхэма от станции к
    # старту, а не разворот station_leg_xy).
    return_leg_xy = [(call["x"], call["y"]) for call in navigate_calls[8:]]
    assert return_leg_xy == [(3.0, 5.0), (4.0, 6.0), (5.0, 6.0), (6.0, 6.0)]
    assert navigate_calls[-1]["auto_arm"] is False

    assert flight.state["armed"] is False  # завершили штатным land_wait

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
            marker_id_reader=fts.fixed_marker_reader(48),
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
        charge_wait_total_s=args.charge_wait,
    )

    markers = fc.read_map(config.map_path)
    proxies = fc.init_flight(config.node_name)
    led.use_technic_ros1_backend()
    try:
        try:
            run_mission(proxies, markers, config, verbose=not args.quiet)
        except Exception as exc:
            # Любая необработанная ошибка миссии (таймаут навигации и т.д.)
            # раньше оставляла дрон висеть armed без дальнейших команд -
            # воспроизведено на площадке 2026-08-01. Вместо этого пробуем
            # штатную посадку, прежде чем пробросить исключение дальше.
            print(
                "[main] авария в run_mission ({}: {}) - аварийная посадка".format(
                    type(exc).__name__, exc
                )
            )
            try:
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
