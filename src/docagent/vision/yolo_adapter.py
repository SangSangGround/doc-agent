"""YOLO 기반 기입란 탐지 어댑터(선택적 의존).

ultralytics 는 **선택적 패키지**다. 이 모듈은 최상단에서 ultralytics·torch 를
import 하지 않으므로, 두 패키지가 없는 환경에서도 ``import`` 자체는 항상 성공한다.
실제 사용 시점(생성자·함수 내부)에 지연 import 하고, ``ImportError`` 는
:class:`~docagent.errors.AdapterUnavailable` 로 감싸 한국어 설치 안내를 준다.

클래스 매핑은 :mod:`docagent.vision.dataset` 과 공유한다
(``0=signature_field``, ``1=checkbox``).

출력 계약은 규칙 기반 탐지기와 동일하다. 픽셀 결과를
:class:`~docagent.vision.geometry.A4CoordinateSystem` 으로 **A4 mm** 좌표의
:class:`~docagent.contracts.Detection` 으로 변환해 돌려준다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from docagent.contracts import BoxPx, Detection, FieldType
from docagent.errors import AdapterUnavailable, VisionError
from docagent.vision.dataset import CLASS_MAP, FIELD_TYPE_TO_CLASS_ID
from docagent.vision.geometry import A4CoordinateSystem
from docagent.vision.heuristic import coordinate_system_from_image

__all__ = [
    "SOURCE_YOLO",
    "CLASS_ID_TO_FIELD_TYPE",
    "YOLO_FEATURE",
    "YOLO_EXTRA",
    "ultralytics_available",
    "require_yolo_class",
    "YoloDetector",
    "export_onnx",
]

_LOG = logging.getLogger(__name__)

#: YOLO 탐지 결과의 :attr:`Detection.source` 값.
SOURCE_YOLO: str = "yolo"
#: :class:`AdapterUnavailable` 메시지에 쓰는 기능 설명.
YOLO_FEATURE: str = "YOLO 기반 기입란 탐지"
#: ``pip install "docagent[<extra>]"`` 안내에 쓰는 optional 그룹 이름.
YOLO_EXTRA: str = "yolo"

#: 클래스 id → 도메인 :class:`FieldType` (:data:`CLASS_MAP` 의 역방향).
CLASS_ID_TO_FIELD_TYPE: dict[int, FieldType] = {
    class_id: field_type for field_type, class_id in FIELD_TYPE_TO_CLASS_ID.items()
}


# --------------------------------------------------------------------------
# 지연 import
# --------------------------------------------------------------------------


def require_yolo_class() -> Any:
    """ultralytics 의 ``YOLO`` 클래스를 지연 import 해서 돌려준다.

    :returns: ``ultralytics.YOLO`` 클래스 객체.
    :raises docagent.errors.AdapterUnavailable: ultralytics 가 설치되지 않은 경우.
        예외 메시지에 저장소 venv 기준 설치 명령이 한국어로 담긴다.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415 - 선택적 의존이라 지연 import 한다.
    except ImportError as exc:
        raise AdapterUnavailable(
            package="ultralytics", feature=YOLO_FEATURE, extra=YOLO_EXTRA
        ) from exc
    return YOLO


def ultralytics_available() -> bool:
    """ultralytics 사용 가능 여부를 조용히 확인한다(예외를 던지지 않는다).

    탐지기 선택(:func:`docagent.vision.detector_factory.build_detector`)처럼
    "있으면 쓰고 없으면 폴백" 하는 자리에서만 쓴다. 실제 사용 경로에서는
    :func:`require_yolo_class` 를 써서 실패 이유가 드러나게 한다.

    :returns: 설치되어 있으면 True.
    """
    try:
        require_yolo_class()
    except AdapterUnavailable:
        return False
    return True


# --------------------------------------------------------------------------
# 탐지기
# --------------------------------------------------------------------------


class YoloDetector:
    """학습된 YOLO 가중치로 체크박스·서명란을 탐지하는 어댑터.

    :class:`docagent.interfaces.Detector` 프로토콜을 만족한다.

    :param weights: 학습된 가중치 파일 경로(``.pt`` 또는 ``.onnx``).
    :param conf: 신뢰도 임계값(0.0~1.0).
    :param iou: NMS IoU 임계값(0.0~1.0).
    :param imgsz: 추론 입력 크기(px, 32 이상).
    :param device: 추론 장치 문자열(예: ``"cpu"``, ``"0"``). ``None`` 이면 ultralytics 기본값.
    :param class_map: 클래스 id → 이름. 기본값은 :data:`~docagent.vision.dataset.CLASS_MAP`.
    :raises docagent.errors.AdapterUnavailable: ultralytics 가 설치되지 않은 경우.
    :raises FileNotFoundError: 가중치 파일이 없는 경우(한국어 메시지).
    :raises ValueError: 파라미터가 허용 범위를 벗어난 경우.

    사용 예::

        detector = YoloDetector("runs/detect/train/weights/best.pt", device="cpu")
        detections = detector.detect(image_bgr)
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
        class_map: dict[int, str] | None = None,
    ) -> None:
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"conf 는 0.0~1.0 이어야 합니다: {conf}")
        if not 0.0 <= iou <= 1.0:
            raise ValueError(f"iou 는 0.0~1.0 이어야 합니다: {iou}")
        if imgsz < 32:
            raise ValueError(f"imgsz 는 32 이상이어야 합니다: {imgsz}")

        yolo_class = require_yolo_class()  # ultralytics 없으면 여기서 AdapterUnavailable

        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"YOLO 가중치 파일을 찾을 수 없습니다: {self.weights} "
                "(scripts/train_yolo.py 로 학습한 뒤 best.pt 경로를 지정하십시오.)"
            )
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device
        self.class_map = dict(class_map) if class_map is not None else dict(CLASS_MAP)

        try:
            self._model = yolo_class(str(self.weights))
        except Exception as exc:  # noqa: BLE001 - 하위 예외를 도메인 예외로 감싼다.
            raise VisionError(
                f"YOLO 가중치를 불러오지 못했습니다: {self.weights} ({exc})"
            ) from exc
        _LOG.info("YOLO 가중치를 불러왔습니다: %s", self.weights)

    # -- 공개 API --------------------------------------------------------

    def detect(self, image: Any) -> list[Detection]:
        """이미지에서 기입란 후보를 탐지한다.

        :param image: A4 로 정립된 문서 이미지.
            ``(H, W, 3)`` uint8 BGR 또는 ``(H, W)`` uint8 그레이스케일.
        :returns: A4 mm 좌표의 :class:`~docagent.contracts.Detection` 목록.
            ``(type, y_mm, x_mm)`` 순으로 정렬된다. 후보가 없으면 빈 리스트.
        :raises docagent.errors.VisionError: 입력이 유효하지 않거나 추론에 실패한 경우.
        """
        if not isinstance(image, np.ndarray):
            raise VisionError(
                f"입력 이미지는 numpy 배열이어야 합니다: {type(image).__name__}"
            )
        if image.size == 0:
            raise VisionError("입력 이미지가 비어 있습니다.")

        coords = coordinate_system_from_image(image)
        try:
            results = self._model.predict(
                source=image,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - 하위 예외를 도메인 예외로 감싼다.
            raise VisionError(f"YOLO 추론에 실패했습니다: {exc}") from exc

        detections = self._convert_results(results, coords)
        detections.sort(
            key=lambda d: (d.type.value, round(d.box_mm.y_mm, 3), round(d.box_mm.x_mm, 3))
        )
        _LOG.debug("YOLO 탐지 %d건(conf>=%.2f).", len(detections), self.conf)
        return detections

    # -- 변환 ------------------------------------------------------------

    def _convert_results(
        self, results: Any, coords: A4CoordinateSystem
    ) -> list[Detection]:
        """ultralytics 결과 객체를 :class:`Detection` 목록으로 바꾼다.

        :param results: ``model.predict`` 가 돌려준 ``Results`` 목록.
        :param coords: 픽셀 → mm 변환에 쓸 좌표계.
        :returns: :class:`Detection` 목록.
        :raises docagent.errors.VisionError: 결과 구조를 해석할 수 없는 경우.
        """
        detections: list[Detection] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            try:
                xyxy = np.asarray(boxes.xyxy, dtype=np.float64).reshape(-1, 4)
                confidences = np.asarray(boxes.conf, dtype=np.float64).reshape(-1)
                class_ids = np.asarray(boxes.cls, dtype=np.int64).reshape(-1)
            except (AttributeError, TypeError, ValueError) as exc:
                raise VisionError(
                    f"YOLO 결과 구조를 해석할 수 없습니다: {exc}"
                ) from exc
            if not (len(xyxy) == len(confidences) == len(class_ids)):
                raise VisionError(
                    "YOLO 결과의 상자·신뢰도·클래스 개수가 일치하지 않습니다: "
                    f"{len(xyxy)}/{len(confidences)}/{len(class_ids)}"
                )

            for (x1, y1, x2, y2), confidence, class_id in zip(
                xyxy, confidences, class_ids
            ):
                field_type = CLASS_ID_TO_FIELD_TYPE.get(int(class_id))
                if field_type is None:
                    _LOG.warning(
                        "도메인 유형에 매핑되지 않은 YOLO 클래스라 제외합니다: id=%d, 이름=%s",
                        int(class_id),
                        self.class_map.get(int(class_id), "알 수 없음"),
                    )
                    continue
                box_px = BoxPx(
                    x=int(round(min(x1, x2))),
                    y=int(round(min(y1, y2))),
                    w=max(0, int(round(abs(x2 - x1)))),
                    h=max(0, int(round(abs(y2 - y1)))),
                )
                detections.append(
                    Detection(
                        type=field_type,
                        box_mm=coords.to_mm(box_px, clamp=True),
                        confidence=float(min(1.0, max(0.0, confidence))),
                        source=SOURCE_YOLO,
                    )
                )
        return detections


# --------------------------------------------------------------------------
# 내보내기
# --------------------------------------------------------------------------


def export_onnx(
    weights: str | Path,
    *,
    imgsz: int = 640,
    opset: int = 12,
    simplify: bool = True,
    dynamic: bool = False,
) -> Path:
    """학습된 가중치를 ONNX 로 내보낸다(엣지 추론용).

    :param weights: 원본 ``.pt`` 가중치 경로.
    :param imgsz: 내보낼 입력 크기(px, 32 이상).
    :param opset: ONNX opset 버전.
    :param simplify: 그래프 단순화 여부.
    :param dynamic: 동적 입력 크기 사용 여부.
    :returns: 생성된 ``.onnx`` 파일 경로.
    :raises docagent.errors.AdapterUnavailable: ultralytics 가 설치되지 않은 경우.
    :raises FileNotFoundError: 가중치 파일이 없는 경우.
    :raises ValueError: ``imgsz`` 가 32 미만인 경우.
    :raises docagent.errors.VisionError: 내보내기에 실패했거나 결과 파일을 찾지 못한 경우.
    """
    if imgsz < 32:
        raise ValueError(f"imgsz 는 32 이상이어야 합니다: {imgsz}")
    yolo_class = require_yolo_class()
    weights_path = Path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO 가중치 파일을 찾을 수 없습니다: {weights_path}")

    try:
        model = yolo_class(str(weights_path))
        exported = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
        )
    except Exception as exc:  # noqa: BLE001 - 하위 예외를 도메인 예외로 감싼다.
        raise VisionError(f"ONNX 내보내기에 실패했습니다: {weights_path} ({exc})") from exc

    candidate = Path(str(exported)) if exported else weights_path.with_suffix(".onnx")
    if not candidate.is_file():
        raise VisionError(
            f"ONNX 내보내기 결과 파일을 찾지 못했습니다: {candidate}"
        )
    _LOG.info("ONNX 내보내기 완료: %s", candidate)
    return candidate
