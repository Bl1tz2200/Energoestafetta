# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Что это

Код для инженерного соревнования «Энергоэстафета» (Архипелаг 2026): два БВС
(дрона) Skyris Technic 6S (PX4 + Orange Pi 5 Pro, **ROS 1/rospy, не ROS 2**)
автономно взаимодействуют с двумя зарядными станциями и доставляют груз.
Полное описание задания и регламент — `TASK.md`; план работ по приоритетам —
`PLAN.md`. Обе страницы на русском — пиши комментарии/докстринги/сообщения в
коде на русском, чтобы соответствовать существующему стилю.

Никакого ROS 1, реального дрона или `rospy` в среде разработки нет и не
будет — весь код спроектирован так, чтобы работать и тестироваться без них
(см. «Архитектура» ниже).

## Команды

Нет build/lint-тулинга и pytest — каждый модуль сам себе тест через
`--self-test`/прямой запуск. Это основной способ проверить, что код не
сломан, без ROS 1 и дрона:

```bash
python3 bvs1_flight.py --self-test
python3 bvs2_flight.py --self-test
python3 flight_core.py            # без флага, самотест — единственная точка входа
python3 led_interface.py --self-test
python3 gripper_control.py --self-test
python3 energy_relay_vision.py --self-test
python3 station_protocol.py --self-test
python3 mission_sync.py --self-test
```

Запускать все восемь при любом изменении в общих модулях (`flight_core.py`,
`led_interface.py`, `gripper_control.py`) — сценарии БВС и протоколы зависят
от них напрямую. `gripper_control.py` тестируется через `TechnicGPIOBackend(
runner=...)` — конструктор принимает `runner` для подмены реального процесса
`gpio`, поэтому самотест возможен без Orange Pi (не путать с самим
сервоприводом/захватом — их поведение самотест не проверяет).
`energy_relay_vision.py --self-test` требует `opencv-contrib-python`/`numpy`
в окружении разработки — без них модуль сразу завершится понятной ошибкой
(это его собственная защита, не баг).

`bvs1_flight.py`/`bvs2_flight.py` берут общие тестовые заглушки
(`FakeFlight`, `FakeClock`, запись LED-команд) из `flight_test_support.py` —
меняя логику самотеста одного из сценариев, проверь, не нужно ли то же
самое во втором и не сломалась ли сама заглушка для обоих.

Запуск реальных полётных сценариев на дроне и параметры CLI (`--map`,
`--start-marker`, `--station-marker`, `--station-height`, `--gripper-pin` и
т.д.) — см. README.md, там расписано пошагово вместе с подготовкой окружения
ROS 1, картой `aruco_map` и переносом кода на Orange Pi (раздел «Перенос
кода на дрон»).

## Архитектура

**Принцип, общий для всех модулей**: рабочая логика не импортирует `rospy`/
`technic`/`mavros_msgs`/`cv_bridge` на верхнем уровне. Импорт ROS-специфики
спрятан внутри «backend»/«init»-функций (`flight_core.init_flight`,
`led_interface.use_technic_ros1_backend`,
`gripper_control.use_technic_gpio_gripper`,
`energy_relay_vision.TechnicROS1Vision`) и вызывается один раз в `main()`
каждого скрипта. Вся остальная логика получает нужные операции как явно
переданные callable/прокси (см. `FlightProxies` в `flight_core.py`) — поэтому
её можно прогнать в `--self-test` с обычными функциями-заглушками на
десктопе без ROS. **При добавлении новой ROS-зависимости следуй этому же
паттерну**, иначе модуль перестанет тестироваться без дрона.

Слои (снизу вверх):

1. **Инфраструктурные примитивы** — не знают про конкретную миссию:
   - `flight_core.py` — навигация/посадка/дальномер (`navigate_wait`,
     `land_wait`, `wait_until_stable`, `controlled_descent_and_disarm`,
     `read_map`/`marker_xy` для карты `aruco_map`, `simulate_charging` для
     имитации зарядки по Табл.1). Посадка на куб станции сделана управляемым
     спуском с контролем `/rangefinder/range` и ручным дизармом, а не
     штатным `land()` — поведение `land()` на приподнятой поверхности не
     описано в документации Skyris. `simulate_charging` принимает
     `set_led_fn` явным аргументом, а не импортирует `led_interface` —
     слой primitives не должен знать про LED-модуль.
   - `led_interface.py` — `set_led(pattern, color)` через
     `LEDController`/`LEDBackend`; есть `ConsoleBackend` и `CallbackBackend`
     для тестов/симуляции помимо `TechnicROS1Backend`.
   - `gripper_control.py` — `gripper_open()`/`gripper_close()`; захват
     управляется **не через ROS**, а напрямую CLI-утилитой `gpio` (WiringPi)
     на Orange Pi (`TechnicGPIOBackend`/`_run_gpio_cli`).
   - `energy_relay_vision.py` — `StationVision` (HSV-детектор цвета станции +
     опциональный ArUco) поверх кадров; `TechnicROS1Vision` подключает его к
     топику `main_camera/image_raw` через `cv_bridge`. Позицию/ID меток даёт
     сам `aruco_pose` (топик `aruco_detect/markers`), здесь решается только
     задача цвета станции.

2. **Межбортовой протокол (Приоритет 4)** — самостоятельные модули, пока
   **не подключённые** к полётным сценариям (интеграция — Приоритет 5):
   - `station_protocol.py` — сигналы БВС↔станция (`Signal` enum,
     `send_signal`/`wait_for_signal`/`wait_for_takeoff_command`), включая
     ожидание команды на взлёт с клавиатуры оператора.
   - `mission_sync.py` — синхронный взлёт двух БВС через `TakeoffBarrier` по
     TCP (`serve_takeoff_barrier` на ноутбуке оператора,
     `wait_at_takeoff_barrier` на каждом Orange Pi) и `run_concurrently` для
     запуска асинхронных задач с ожиданием по событию/таймауту вместо
     жёсткого `sleep`.

3. **Полётные сценарии** — собирают слои 1 и 2 в конкретную миссию:
   - `bvs1_flight.py` — `MissionConfig` + `run_mission`: взлёт с
     `--start-marker` → навигация к `--station-marker` по карте `aruco_map`
     → стабилизация → управляемый спуск на куб → зарядка через
     `fc.simulate_charging` (сейчас заглушка на таймере вместо честного
     протокола станции) → возврат и обычная посадка.
   - `bvs2_flight.py` — то же плюс захват груза с `--cargo-marker` через
     `gripper_control` и своя (физически отдельная от БВС-1) зарядная
     станция `--station-marker`.

   Оба сценария читают позиции меток из `field_map.txt` (формат
   `aruco_pose`/`aruco_map`: `id size x y z rot_z rot_y rot_x`, нумерация
   построчно от левого верхнего угла поля 7×7) — id меток и высота станции
   передаются как аргументы CLI, а не хардкодятся, потому что раскладка поля
   объявляется организаторами перед каждой попыткой.

   `flight_test_support.py` — тестовые заглушки, общие для `_self_test()`
   обоих сценариев (не боевой код, на дрон переносить не обязательно).

## Важные допущения, требующие проверки на площадке

Отмечены в README.md/TASK.md как непроверенные без реального ROS 1/дрона —
если меняешь код рядом с ними, не удаляй эти пометки, пока допущение не
подтверждено:

- Направление нумерации `field_map.txt` соответствует физическому полю —
  **подтверждено** на площадке (метка 0 — верхний левый угол).
- Имя сервиса армирования `mavros/cmd/arming` — стандартное для MAVROS, но
  не подтверждено явно документацией Skyris (используется в
  `flight_core.init_flight`).
- Пороги дальномера/шаг спуска в `controlled_descent_and_disarm` — начальные
  значения, требуют калибровки по факту на кубе 80 см.
