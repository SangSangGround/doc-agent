"""탐지기 선택 팩토리 — 학습 가중치가 있으면 YOLO, 없으면 규칙 기반.

파이프라인 상위 계층이 "어떤 탐지기를 쓸지" 를 몰라도 되게 하는 유일한 진입점이다.
선택 결과와 **그 이유**를 항상 로그로 남긴다(조용한 폴백 금지).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from docagent.errors import AdapterUnavailable
from docagent.interfaces import Detector
from docagent.vision.heuristic import HeuristicDetector, HeuristicParams

__all__ = ["DetectorPreference", "build_detector", "describe_detector"]

_LOG = logging.getLogger(__name__)

#: :func:`build_detector` 의 ``prefer`` 인자 허용 값.
DetectorPreference = Literal["auto", "yolo", "heuristic", "roboflow"]

_ALLOWED_PREFERENCES: tuple[str, ...] = ("auto", "yolo", "heuristic", "roboflow")


def build_detector(
    weights: str | Path | None = None,
    prefer: DetectorPreference = "auto",
    *,
    params: HeuristicParams | None = None,
    conf: float | None = None,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str | None = None,
) -> Detector:
    """상황에 맞는 :class:`~docagent.interfaces.Detector` 구현을 만든다.

    선택 규칙:

    * ``prefer="roboflow"`` — 서버리스 API 사용. 실패 시 폴백하지 않는다.
    * ``prefer="heuristic"`` — 항상 :class:`~docagent.vision.heuristic.HeuristicDetector`.
    * ``prefer="yolo"`` — 반드시 YOLO 를 쓴다. 가중치가 없거나 ultralytics 가
      설치되지 않았으면 **폴백하지 않고 예외를 던진다**(사용자가 YOLO 를 명시했으므로).
    * ``prefer="auto"``(기본) — ``weights`` 가 주어지고 ultralytics 가 설치되어 있으면
      YOLO, 그 밖에는 규칙 기반으로 폴백하고 그 이유를 ``INFO`` 로그로 남긴다.

    :param weights: 학습된 YOLO 가중치 경로. ``None`` 이면 규칙 기반을 쓴다.
    :param prefer: 선택 전략. ``"auto"`` / ``"yolo"`` / ``"heuristic"`` / ``"roboflow"``.
    :param params: 규칙 기반 탐지기의 임계값. ``None`` 이면 기본값.
    :param conf: 탐지 신뢰도 임계값. None 이면 YOLO 0.25, Roboflow 0.5.
    :param iou: YOLO NMS IoU 임계값.
    :param imgsz: YOLO 추론 입력 크기(px).
    :param device: YOLO 추론 장치(예: ``"cpu"``).
    :returns: :class:`Detector` 프로토콜을 만족하는 탐지기.
    :raises ValueError: ``prefer`` 가 허용 값이 아닌 경우.
    :raises docagent.errors.AdapterUnavailable: ``prefer="yolo"`` 인데 ultralytics 가 없는 경우.
    :raises FileNotFoundError: ``prefer="yolo"`` 인데 가중치 파일이 없는 경우.
    """
    if prefer not in _ALLOWED_PREFERENCES:
        raise ValueError(
            f"prefer 는 {', '.join(_ALLOWED_PREFERENCES)} 중 하나여야 합니다: {prefer!r}"
        )

    if prefer == "roboflow":
        from docagent.detector.roboflow_detector import DEFAULT_CONFIDENCE, RoboflowDetector

        _LOG.info("Roboflow 서버리스 탐지기를 사용합니다(prefer='roboflow').")
        return RoboflowDetector(conf=DEFAULT_CONFIDENCE if conf is None else conf)

    conf = 0.25 if conf is None else conf
    if prefer == "heuristic":
        _LOG.info("규칙 기반 탐지기를 사용합니다(prefer='heuristic').")
        return HeuristicDetector(params)

    if prefer == "yolo":
        if weights is None:
            raise FileNotFoundError(
                "prefer='yolo' 인데 가중치 경로가 지정되지 않았습니다. "
                "weights 인자에 학습된 .pt 경로를 주십시오."
            )
        from docagent.vision.yolo_adapter import YoloDetector  # 선택적 의존 지연 import

        _LOG.info("YOLO 탐지기를 사용합니다(prefer='yolo'): %s", weights)
        return YoloDetector(
            weights, conf=conf, iou=iou, imgsz=imgsz, device=device
        )

    # prefer == "auto"
    if weights is None:
        _LOG.info("가중치가 지정되지 않아 규칙 기반 탐지기로 폴백합니다.")
        return HeuristicDetector(params)

    weights_path = Path(weights)
    if not weights_path.is_file():
        _LOG.info(
            "가중치 파일이 없어 규칙 기반 탐지기로 폴백합니다: %s", weights_path
        )
        return HeuristicDetector(params)

    from docagent.vision.yolo_adapter import YoloDetector  # 선택적 의존 지연 import

    try:
        detector = YoloDetector(
            weights_path, conf=conf, iou=iou, imgsz=imgsz, device=device
        )
    except AdapterUnavailable as exc:
        _LOG.info("ultralytics 가 없어 규칙 기반 탐지기로 폴백합니다: %s", exc)
        return HeuristicDetector(params)
    _LOG.info("YOLO 탐지기를 사용합니다: %s", weights_path)
    return detector


def describe_detector(detector: Detector) -> str:
    """탐지기 종류를 사람이 읽는 한국어 한 줄로 설명한다.

    로그·데모 출력에서 "지금 무엇이 돌고 있는지" 를 밝히는 용도다.

    :param detector: :func:`build_detector` 가 만든 탐지기.
    :returns: 한국어 설명 문자열.
    """
    if isinstance(detector, HeuristicDetector):
        return "규칙 기반 탐지기(OpenCV 윤곽선·형태학, 학습 가중치 불필요)"
    from docagent.detector.roboflow_detector import RoboflowDetector

    if isinstance(detector, RoboflowDetector):
        return "Roboflow RF-DETR 탐지기(서버리스 Workflow)"
    weights = getattr(detector, "weights", None)
    if weights is not None:
        return f"YOLO 탐지기(가중치: {weights})"
    return f"알 수 없는 탐지기 구현: {type(detector).__name__}"
