#!/usr/bin/env python3
"""Распознавание ArUco-меток и цвета зарядной станции.

Модуль относится к пункту 2.4 PLAN.md и не управляет полётом. Его можно:

1. импортировать в полётный скрипт::

       detector = StationVision()
       result = detector.detect(frame_bgr)
       print(result.to_dict())

2. проверять отдельно на изображении, видео или USB-камере::

       python3 energy_relay_vision.py photo.jpg --output marked.jpg
       python3 energy_relay_vision.py video.mp4 --output marked.mp4
       python3 energy_relay_vision.py --camera 0 --display
       python3 energy_relay_vision.py --self-test

Кадр должен быть в формате BGR, стандартном для OpenCV. По умолчанию
используется словарь DICT_4X4_50, номера меток ограничены полем 0..48, а цвета
станций — red и green. На очном этапе конкретный ID станции назначают эксперты,
поэтому он намеренно не зашит в код.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:  # понятная ошибка вместо падения глубоко внутри кода
    raise SystemExit(
        "Нужны OpenCV с модулем aruco и NumPy. "
        "Установите: pip install opencv-contrib-python numpy"
    ) from exc


Point = Tuple[float, float]
Box = Tuple[int, int, int, int]
HSVRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


# У красного два диапазона: шкала оттенка OpenCV замыкается на 0/179.
# Диапазоны намеренно чуть шире идеальных цветов Gazebo, чтобы выдерживать
# изменение яркости. На реальном полигоне их нужно уточнить по одному кадру.
DEFAULT_COLOR_RANGES: Dict[str, Tuple[HSVRange, ...]] = {
    "red": (
        ((0, 100, 60), (12, 255, 255)),
        ((168, 100, 60), (179, 255, 255)),
    ),
    "green": (
        ((35, 70, 50), (90, 255, 255)),
    ),
}


@dataclass(frozen=True)
class ArucoDetection:
    aruco_id: int
    center: Point
    corners: Tuple[Point, Point, Point, Point]
    area_px: float
    side_px: float
    angle_deg: float
    offset: Point


@dataclass(frozen=True)
class StationDetection:
    color: str
    center: Point
    bbox: Box
    area_px: float
    frame_ratio: float
    fill_ratio: float
    confidence: float
    aruco_id: Optional[int]


@dataclass(frozen=True)
class VisionResult:
    width: int
    height: int
    markers: Tuple[ArucoDetection, ...]
    stations: Tuple[StationDetection, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    def station_by_marker(self, aruco_id: int) -> Optional[StationDetection]:
        candidates = [
            station
            for station in self.stations
            if station.aruco_id == aruco_id
        ]
        return max(candidates, key=lambda item: item.confidence, default=None)


class StationVision:
    """Детектор без состояния: один вызов ``detect`` обрабатывает один кадр."""

    def __init__(
        self,
        dictionary_name: str = "DICT_4X4_50",
        allowed_marker_ids: Optional[Iterable[int]] = range(49),
        min_marker_area_ratio: float = 0.0002,
        min_color_area_ratio: float = 0.003,
        min_color_fill_ratio: float = 0.18,
        color_ranges: Optional[Dict[str, Sequence[HSVRange]]] = None,
    ) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "В этой сборке OpenCV нет cv2.aruco. "
                "Установите opencv-contrib-python."
            )
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError("Неизвестный словарь ArUco: " + dictionary_name)
        if min_marker_area_ratio <= 0 or min_color_area_ratio <= 0:
            raise ValueError("Минимальные площади должны быть больше нуля")
        if not 0 < min_color_fill_ratio <= 1:
            raise ValueError("min_color_fill_ratio должен лежать в (0, 1]")

        dictionary_id = getattr(cv2.aruco, dictionary_name)
        dictionary_factory = getattr(
            cv2.aruco,
            "getPredefinedDictionary",
            getattr(cv2.aruco, "Dictionary_get", None),
        )
        if dictionary_factory is None:
            raise RuntimeError("OpenCV не умеет создавать словарь ArUco")

        self.dictionary_name = dictionary_name
        self.dictionary = dictionary_factory(dictionary_id)
        self.allowed_marker_ids = (
            None
            if allowed_marker_ids is None
            else frozenset(int(value) for value in allowed_marker_ids)
        )
        self.min_marker_area_ratio = float(min_marker_area_ratio)
        self.min_color_area_ratio = float(min_color_area_ratio)
        self.min_color_fill_ratio = float(min_color_fill_ratio)
        self.color_ranges = {
            name: tuple(ranges)
            for name, ranges in (color_ranges or DEFAULT_COLOR_RANGES).items()
        }

        if hasattr(cv2.aruco, "DetectorParameters"):
            parameters = cv2.aruco.DetectorParameters()
        else:
            parameters = cv2.aruco.DetectorParameters_create()
        self._aruco_detector = (
            cv2.aruco.ArucoDetector(self.dictionary, parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self._aruco_parameters = parameters
        self._open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )
        self._close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (11, 11)
        )

    def detect(self, frame_bgr: "np.ndarray") -> VisionResult:
        self._validate_frame(frame_bgr)
        height, width = frame_bgr.shape[:2]
        markers = self._detect_markers(frame_bgr)
        stations = self._detect_stations(frame_bgr, markers)
        return VisionResult(
            width=width,
            height=height,
            markers=tuple(markers),
            stations=tuple(stations),
        )

    @staticmethod
    def _validate_frame(frame_bgr: "np.ndarray") -> None:
        if frame_bgr is None:
            raise ValueError("Получен пустой кадр")
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("Кадр должен быть numpy.ndarray")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.size == 0:
            raise ValueError("Ожидается непустой BGR-кадр с тремя каналами")
        if frame_bgr.dtype != np.uint8:
            raise ValueError("Ожидается BGR-кадр типа uint8")

    def _detect_markers(
        self, frame_bgr: "np.ndarray"
    ) -> List[ArucoDetection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._aruco_detector is not None:
            corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.dictionary,
                parameters=self._aruco_parameters,
            )

        if ids is None:
            return []

        height, width = frame_bgr.shape[:2]
        frame_area = float(width * height)
        min_area = self.min_marker_area_ratio * frame_area
        detections: List[ArucoDetection] = []

        for raw_corners, raw_id in zip(corners, ids.flatten()):
            aruco_id = int(raw_id)
            if (
                self.allowed_marker_ids is not None
                and aruco_id not in self.allowed_marker_ids
            ):
                continue

            points = np.asarray(raw_corners, np.float32).reshape(4, 2)
            area = abs(float(cv2.contourArea(points)))
            if area < min_area:
                continue

            center_x, center_y = points.mean(axis=0)
            edge_lengths = [
                float(np.linalg.norm(points[index] - points[(index + 1) % 4]))
                for index in range(4)
            ]
            top_edge = points[1] - points[0]
            angle = math.degrees(
                math.atan2(float(top_edge[1]), float(top_edge[0]))
            )
            offset_x = (float(center_x) - width / 2.0) / (width / 2.0)
            offset_y = (float(center_y) - height / 2.0) / (height / 2.0)

            detections.append(
                ArucoDetection(
                    aruco_id=aruco_id,
                    center=(float(center_x), float(center_y)),
                    corners=tuple(
                        (float(point[0]), float(point[1])) for point in points
                    ),
                    area_px=area,
                    side_px=sum(edge_lengths) / 4.0,
                    angle_deg=angle,
                    offset=(offset_x, offset_y),
                )
            )

        return sorted(detections, key=lambda item: item.aruco_id)

    def _detect_stations(
        self,
        frame_bgr: "np.ndarray",
        markers: Sequence[ArucoDetection],
    ) -> List[StationDetection]:
        blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        height, width = frame_bgr.shape[:2]
        frame_area = float(width * height)
        min_area = self.min_color_area_ratio * frame_area
        stations: List[StationDetection] = []

        for color, ranges in self.color_ranges.items():
            mask = np.zeros((height, width), dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(
                    hsv,
                    np.asarray(lower, dtype=np.uint8),
                    np.asarray(upper, dtype=np.uint8),
                )

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                contour_area = abs(float(cv2.contourArea(contour)))
                if contour_area < min_area:
                    continue

                x, y, box_width, box_height = cv2.boundingRect(contour)
                box_area = float(box_width * box_height)
                if box_area <= 0:
                    continue
                fill_ratio = float(
                    cv2.countNonZero(mask[y : y + box_height, x : x + box_width])
                ) / box_area
                if fill_ratio < self.min_color_fill_ratio:
                    continue

                moments = cv2.moments(contour)
                if abs(float(moments["m00"])) > 1e-9:
                    center_x = float(moments["m10"] / moments["m00"])
                    center_y = float(moments["m01"] / moments["m00"])
                else:
                    center_x = x + box_width / 2.0
                    center_y = y + box_height / 2.0

                frame_ratio = contour_area / frame_area
                area_score = min(
                    1.0, frame_ratio / max(self.min_color_area_ratio * 5.0, 1e-9)
                )
                confidence = min(
                    1.0, 0.65 * fill_ratio + 0.35 * area_score
                )
                marker_id = self._associate_marker(
                    contour,
                    (x, y, box_width, box_height),
                    (center_x, center_y),
                    markers,
                )
                stations.append(
                    StationDetection(
                        color=color,
                        center=(center_x, center_y),
                        bbox=(x, y, box_width, box_height),
                        area_px=contour_area,
                        frame_ratio=frame_ratio,
                        fill_ratio=fill_ratio,
                        confidence=confidence,
                        aruco_id=marker_id,
                    )
                )

        return sorted(
            stations,
            key=lambda item: (
                item.aruco_id is None,
                item.aruco_id if item.aruco_id is not None else 10**9,
                -item.confidence,
            ),
        )

    @staticmethod
    def _associate_marker(
        contour: "np.ndarray",
        bbox: Box,
        color_center: Point,
        markers: Sequence[ArucoDetection],
    ) -> Optional[int]:
        """Связать цветную площадку с меткой внутри неё или рядом с ней."""
        if not markers:
            return None

        x, y, width, height = bbox
        contained: List[Tuple[float, int]] = []
        for marker in markers:
            marker_x, marker_y = marker.center
            inside_contour = cv2.pointPolygonTest(
                contour, (marker_x, marker_y), False
            ) >= 0
            inside_bbox = (
                x <= marker_x <= x + width and y <= marker_y <= y + height
            )
            if inside_contour or inside_bbox:
                distance = math.hypot(
                    marker_x - color_center[0], marker_y - color_center[1]
                )
                contained.append((distance, marker.aruco_id))

        if contained:
            return min(contained)[1]

        # Если цветная рамка разорвана бликом или закрыта корпусом, её контур
        # может не охватить центр метки. Тогда допускается только близкая метка.
        nearest = min(
            (
                (
                    math.hypot(
                        marker.center[0] - color_center[0],
                        marker.center[1] - color_center[1],
                    )
                    / max(marker.side_px, 1.0),
                    marker.aruco_id,
                )
                for marker in markers
            ),
            default=None,
        )
        return nearest[1] if nearest is not None and nearest[0] <= 2.5 else None


def draw_result(
    frame_bgr: "np.ndarray", result: VisionResult
) -> "np.ndarray":
    """Вернуть копию кадра с рамками и подписями."""
    output = frame_bgr.copy()
    color_bgr = {
        "red": (0, 0, 255),
        "green": (0, 200, 0),
    }

    for station in result.stations:
        x, y, width, height = station.bbox
        color = color_bgr.get(station.color, (255, 255, 0))
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        marker_text = (
            "?"
            if station.aruco_id is None
            else str(station.aruco_id)
        )
        label = (
            f"{station.color} id={marker_text} "
            f"conf={station.confidence:.2f}"
        )
        cv2.putText(
            output,
            label,
            (x, max(18, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    for marker in result.markers:
        points = np.asarray(marker.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [points], True, (255, 180, 0), 2)
        center = (int(round(marker.center[0])), int(round(marker.center[1])))
        cv2.circle(output, center, 4, (255, 180, 0), -1)
        cv2.putText(
            output,
            f"ArUco {marker.aruco_id}",
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def _parse_marker_ids(value: str) -> Optional[Tuple[int, ...]]:
    value = value.strip().lower()
    if value in {"any", "all", "*"}:
        return None
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("Список ID пуст")
    if any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("ID ArUco не может быть отрицательным")
    return result


def _json_line(frame_index: int, result: VisionResult) -> str:
    payload = {"frame": frame_index, **result.to_dict()}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _make_test_frame(dictionary: object) -> "np.ndarray":
    frame = np.full((540, 960, 3), 215, dtype=np.uint8)
    stations = (
        (8, (70, 130, 300, 300), (0, 0, 230)),
        (33, (590, 130, 300, 300), (0, 190, 0)),
    )
    for marker_id, (x, y, width, height), color in stations:
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
        marker_size = 150
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker = cv2.aruco.generateImageMarker(
                dictionary, marker_id, marker_size
            )
        else:
            marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
            cv2.aruco.drawMarker(dictionary, marker_id, marker_size, marker, 1)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        marker_x = x + (width - marker_size) // 2
        marker_y = y + (height - marker_size) // 2
        frame[
            marker_y : marker_y + marker_size,
            marker_x : marker_x + marker_size,
        ] = marker_bgr
    return frame


def _run_self_test(
    detector: StationVision, output_path: Optional[Path]
) -> int:
    frame = _make_test_frame(detector.dictionary)
    result = detector.detect(frame)
    pairs = {
        (station.aruco_id, station.color)
        for station in result.stations
        if station.aruco_id is not None
    }
    expected = {(8, "red"), (33, "green")}
    print(_json_line(0, result))
    if output_path is not None:
        if not cv2.imwrite(str(output_path), draw_result(frame, result)):
            print("Не удалось сохранить " + str(output_path), file=sys.stderr)
            return 2
    if expected.issubset(pairs):
        print("SELF-TEST: OK")
        return 0
    print(
        "SELF-TEST: FAIL; ожидались пары "
        + repr(sorted(expected))
        + ", получены "
        + repr(sorted(pairs)),
        file=sys.stderr,
    )
    return 1


def _open_writer(
    path: Path, fps: float, size: Tuple[int, int]
) -> "cv2.VideoWriter":
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 25.0,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError("Не удалось открыть видео для записи: " + str(path))
    return writer


def _process_stream(
    source: object,
    detector: StationVision,
    output_path: Optional[Path],
    display: bool,
    print_every: int,
) -> int:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print("Не удалось открыть источник: " + str(source), file=sys.stderr)
        return 2

    writer = None
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = detector.detect(frame)
            if frame_index % print_every == 0:
                print(_json_line(frame_index, result), flush=True)

            marked = draw_result(frame, result)
            if output_path is not None:
                if writer is None:
                    fps = float(capture.get(cv2.CAP_PROP_FPS))
                    writer = _open_writer(
                        output_path, fps, (marked.shape[1], marked.shape[0])
                    )
                writer.write(marked)

            if display:
                cv2.imshow("energy_relay_vision", marked)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ArUco + red/green detector for charging stations"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="изображение или видео",
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="индекс USB-камеры, например 0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="сохранить размеченное изображение/видео",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="показывать размеченные кадры; q/Esc — выход",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="проверить модуль на сгенерированных станциях 8/red и 33/green",
    )
    parser.add_argument(
        "--dictionary",
        default="DICT_4X4_50",
        help="имя словаря cv2.aruco (по умолчанию DICT_4X4_50)",
    )
    parser.add_argument(
        "--marker-ids",
        type=_parse_marker_ids,
        default=tuple(range(49)),
        help='разрешённые ID через запятую или "any" (по умолчанию 0..48)',
    )
    parser.add_argument(
        "--min-marker-area",
        type=float,
        default=0.0002,
        help="минимальная доля кадра для ArUco (по умолчанию 0.0002)",
    )
    parser.add_argument(
        "--min-color-area",
        type=float,
        default=0.003,
        help="минимальная доля кадра для станции (по умолчанию 0.003)",
    )
    parser.add_argument(
        "--min-color-fill",
        type=float,
        default=0.18,
        help="минимальная заполненность цветной области (по умолчанию 0.18)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="выводить JSON для каждого N-го кадра (по умолчанию 1)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.print_every < 1:
        print("--print-every должен быть >= 1", file=sys.stderr)
        return 2
    if args.camera is not None and args.input is not None:
        print("Укажите либо файл, либо --camera, но не оба", file=sys.stderr)
        return 2

    try:
        detector = StationVision(
            dictionary_name=args.dictionary,
            allowed_marker_ids=args.marker_ids,
            min_marker_area_ratio=args.min_marker_area,
            min_color_area_ratio=args.min_color_area,
            min_color_fill_ratio=args.min_color_fill,
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.self_test:
        return _run_self_test(detector, args.output)
    if args.camera is not None:
        return _process_stream(
            args.camera,
            detector,
            args.output,
            args.display,
            args.print_every,
        )
    if args.input is None:
        print(
            "Укажите изображение/видео, --camera N или --self-test",
            file=sys.stderr,
        )
        return 2
    if not args.input.is_file():
        print("Файл не найден: " + str(args.input), file=sys.stderr)
        return 2

    image = cv2.imread(str(args.input))
    if image is not None:
        result = detector.detect(image)
        print(_json_line(0, result))
        if args.output is not None:
            if not cv2.imwrite(str(args.output), draw_result(image, result)):
                print(
                    "Не удалось сохранить " + str(args.output),
                    file=sys.stderr,
                )
                return 2
        if args.display:
            cv2.imshow("energy_relay_vision", draw_result(image, result))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    return _process_stream(
        str(args.input),
        detector,
        args.output,
        args.display,
        args.print_every,
    )


if __name__ == "__main__":
    raise SystemExit(main())
