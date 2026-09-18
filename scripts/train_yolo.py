"""YOLO 기입란 탐지기 파인튜닝 스크립트(로드맵 Phase 1).

합성·실사 데이터로 만든 ``data.yaml`` 을 받아 ultralytics YOLO 를 파인튜닝하고,
로드맵 목표치(mAP50 >= 0.85, mAP50-95 >= 0.75) 대비 달성 여부를 판정한 뒤
ONNX / TorchScript 로 내보낸다.

ultralytics 는 선택적 패키지이므로 미설치 환경에서는 한국어 설치 안내를 출력하고
종료 코드 1 로 끝난다(설치를 자동으로 시도하지 않는다).

사용 예::

    .venv\\Scripts\\python.exe scripts/train_yolo.py \\
        --data fixtures/data.yaml --model yolov8m.pt --epochs 100 --device cpu

``data.yaml`` 은 :func:`docagent.vision.dataset.write_data_yaml` 로 만들 수 있다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:  # 저장소를 설치하지 않고도 실행 가능하게 한다.
    sys.path.insert(0, str(_SRC_DIR))

from docagent.errors import AdapterUnavailable  # noqa: E402 - sys.path 설정 이후 import
from docagent.vision.dataset import CLASS_NAMES  # noqa: E402

__all__ = [
    "TARGET_MAP50",
    "TARGET_MAP50_95",
    "build_parser",
    "extract_metrics",
    "judge_metrics",
    "main",
]

_LOG = logging.getLogger("train_yolo")

#: 로드맵 목표 mAP50.
TARGET_MAP50: float = 0.85
#: 로드맵 목표 mAP50-95.
TARGET_MAP50_95: float = 0.75


def build_parser() -> argparse.ArgumentParser:
    """명령행 인자 파서를 만든다.

    :returns: 설정이 끝난 :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="train_yolo.py",
        description=(
            "기입란(서명란·체크박스) 탐지 YOLO 파인튜닝. "
            f"클래스: {', '.join(f'{i}={n}' for i, n in enumerate(CLASS_NAMES))}"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="데이터셋 정의 yaml 경로")
    parser.add_argument("--model", default="yolov8m.pt", help="사전학습 가중치 이름 또는 경로")
    parser.add_argument("--epochs", type=int, default=100, help="학습 에폭 수")
    parser.add_argument("--imgsz", type=int, default=640, help="학습 입력 크기(px)")
    parser.add_argument("--batch", type=int, default=16, help="배치 크기")
    parser.add_argument("--patience", type=int, default=20, help="조기 종료 인내 에폭")
    parser.add_argument("--device", default=None, help="학습 장치(예: cpu, 0, 0,1)")
    parser.add_argument("--project", default="runs/detect", help="결과 저장 상위 디렉터리")
    parser.add_argument("--name", default="docagent_fields", help="실행 이름(하위 디렉터리)")
    parser.add_argument("--seed", type=int, default=20260909, help="난수 시드(재현성)")
    parser.add_argument("--workers", type=int, default=0, help="데이터 로더 워커 수")
    parser.add_argument(
        "--export",
        nargs="*",
        default=["onnx", "torchscript"],
        choices=["onnx", "torchscript", "none"],
        help="학습 후 내보낼 형식(none 이면 내보내지 않음)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="학습 후 검증(val) 단계를 건너뛴다",
    )
    parser.add_argument("--quiet", action="store_true", help="진행 로그를 줄인다")
    return parser


def extract_metrics(metrics: Any) -> dict[str, float]:
    """ultralytics 검증 결과에서 mAP 지표를 뽑는다.

    ultralytics 버전에 따라 ``results_dict`` 키 이름이 달라지므로 여러 후보를 본다.

    :param metrics: ``model.val()`` 이 돌려준 결과 객체.
    :returns: ``{"map50": float, "map50_95": float}``. 찾지 못한 값은 ``float('nan')``.
    """
    values: dict[str, float] = {"map50": float("nan"), "map50_95": float("nan")}
    box = getattr(metrics, "box", None)
    if box is not None:
        for key, attribute in (("map50", "map50"), ("map50_95", "map")):
            value = getattr(box, attribute, None)
            if value is not None:
                try:
                    values[key] = float(value)
                except (TypeError, ValueError):  # pragma: no cover - 방어적 처리
                    pass
    results = getattr(metrics, "results_dict", None)
    if isinstance(results, dict):
        for key, candidates in (
            ("map50", ("metrics/mAP50(B)", "metrics/mAP_0.5")),
            ("map50_95", ("metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95")),
        ):
            if values[key] == values[key]:  # 이미 유효한 값이면 유지(NaN 검사)
                continue
            for candidate in candidates:
                if candidate in results:
                    try:
                        values[key] = float(results[candidate])
                    except (TypeError, ValueError):  # pragma: no cover
                        pass
                    break
    return values


def judge_metrics(values: dict[str, float]) -> tuple[bool, list[str]]:
    """지표를 로드맵 목표치와 비교해 달성 여부와 사람이 읽는 판정문을 만든다.

    :param values: :func:`extract_metrics` 결과.
    :returns: ``(전체 달성 여부, 한국어 판정 줄 목록)``.
    """
    lines: list[str] = []
    passed = True
    for key, label, target in (
        ("map50", "mAP50", TARGET_MAP50),
        ("map50_95", "mAP50-95", TARGET_MAP50_95),
    ):
        value = values.get(key, float("nan"))
        if value != value:  # NaN 검사
            lines.append(f"{label}: 측정값을 확인할 수 없습니다(목표 {target:.2f}).")
            passed = False
            continue
        ok = value >= target
        passed = passed and ok
        lines.append(
            f"{label}: {value:.4f} (목표 {target:.2f}) — {'달성' if ok else '미달'}"
        )
    return (passed, lines)


def _print(message: str) -> None:
    """표준 출력에 한 줄을 적는다(테스트에서 가로채기 쉽게 분리).

    :param message: 출력할 문자열.
    :returns: ``None``.
    """
    print(message)


def main(argv: Sequence[str] | None = None) -> int:
    """스크립트 진입점.

    :param argv: 명령행 인자. ``None`` 이면 :data:`sys.argv` 를 쓴다.
    :returns: 종료 코드. 0=목표 달성, 1=실행 불가(설치·파일 문제), 2=목표 미달.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    data_path = Path(args.data)
    if not data_path.is_file():
        _print(f"[오류] 데이터셋 정의 파일을 찾을 수 없습니다: {data_path}")
        _print(
            "       docagent.vision.dataset.write_data_yaml 로 data.yaml 을 먼저 만드십시오."
        )
        return 1

    try:
        from docagent.vision.yolo_adapter import require_yolo_class

        yolo_class = require_yolo_class()
    except AdapterUnavailable as exc:
        _print("[오류] YOLO 학습에 필요한 선택적 패키지가 없습니다.")
        _print(f"       {exc}")
        return 1

    _print(f"[학습] 데이터셋: {data_path}")
    _print(f"[학습] 사전학습 가중치: {args.model}")
    _print(
        f"[학습] epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, "
        f"patience={args.patience}, device={args.device or '자동'}"
    )

    model = yolo_class(args.model)
    train_kwargs: dict[str, Any] = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "workers": args.workers,
        "exist_ok": True,
        "verbose": not args.quiet,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)

    exit_code = 0
    if args.no_validate:
        _print("[검증] --no-validate 가 지정되어 검증을 건너뜁니다(목표 판정 없음).")
    else:
        val_kwargs: dict[str, Any] = {"data": str(data_path), "imgsz": args.imgsz}
        if args.device is not None:
            val_kwargs["device"] = args.device
        metrics = model.val(**val_kwargs)
        values = extract_metrics(metrics)
        passed, lines = judge_metrics(values)
        _print("[검증] 로드맵 목표 대비 결과")
        for line in lines:
            _print(f"       {line}")
        if passed:
            _print("[검증] 목표를 모두 달성했습니다.")
        else:
            _print(
                "[검증] 목표에 미달했습니다. 데이터 증강·에폭 수·클래스 균형을 점검하십시오."
            )
            exit_code = 2

    formats = [item for item in (args.export or []) if item != "none"]
    for export_format in formats:
        try:
            exported = model.export(format=export_format, imgsz=args.imgsz)
            _print(f"[내보내기] {export_format}: {exported}")
        except Exception as exc:  # noqa: BLE001 - 내보내기 실패는 학습 결과를 무효화하지 않는다.
            _print(f"[경고] {export_format} 내보내기에 실패했습니다: {exc}")

    best = Path(args.project) / args.name / "weights" / "best.pt"
    _print(f"[완료] 최적 가중치 예상 경로: {best}")
    _print(
        "       추론에 쓰려면: build_detector(weights=r'"
        f"{best}', prefer='auto')"
    )
    return exit_code


if __name__ == "__main__":  # pragma: no cover - 스크립트 실행 경로
    raise SystemExit(main())
