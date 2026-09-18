"""합성 신청서 픽스처 생성 CLI.

여러 변형 조합(회전각·해상도·원근·잡음·조명·JPEG)을 렌더링해
``fixtures/<split>/`` 아래에 **PNG 이미지**와 **정답 JSON** 을 저장한다.
로드맵의 데이터셋 계획(train 200 / val 30 / test 20, 각도 0/±10/±20,
150~300 dpi)을 기본값으로 그대로 노출한다.

사용 예::

    .venv\\Scripts\\python.exe scripts\\make_fixtures.py --split test --count 5
    .venv\\Scripts\\python.exe scripts\\make_fixtures.py --split all
    .venv\\Scripts\\python.exe scripts\\make_fixtures.py --out D:\\data\\forms --seed 42

산출물 구조::

    <out>/
      index.json                     # 전체 매니페스트
      train/
        train_0000.png               # 변형이 적용된 문서 이미지
        train_0000.truth.json        # DocumentStructure.to_json() (변형 이전 mm 정답)
        train_0000.written.png       # (--written 지정 시) 체크·서명 기입본
        ...

``fixtures/`` 는 ``.gitignore`` 대상일 수 있으므로 저장 경로를 인자로 받되
기본값은 저장소 하위 ``fixtures/`` 다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import cv2

# 저장소를 설치하지 않고 실행할 수 있도록 src 를 경로에 추가한다.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from docagent.testing.fonts import font_diagnostics  # noqa: E402
from docagent.testing.synthetic import (  # noqa: E402
    DEFAULT_SEED,
    AGREE_LABEL,
    FormSpec,
    iter_spec_grid,
    make_application_form,
    render_written,
)

__all__ = ["SPLIT_DEFAULT_COUNTS", "build_parser", "generate_split", "main"]

#: 로드맵 데이터셋 계획의 split 별 기본 개수.
SPLIT_DEFAULT_COUNTS: dict[str, int] = {"train": 200, "val": 30, "test": 20}

#: 로드맵의 기본 회전각(도) 목록.
DEFAULT_ANGLES: tuple[float, ...] = (0.0, 10.0, -10.0, 20.0, -20.0)
#: 로드맵의 기본 해상도(dpi) 목록.
DEFAULT_DPIS: tuple[int, ...] = (150, 200, 250, 300)

#: split 별 시드 오프셋. split 간 이미지가 겹치지 않게 한다.
_SPLIT_SEED_OFFSET: dict[str, int] = {"train": 0, "val": 100_000, "test": 200_000}


def _parse_floats(raw: str) -> tuple[float, ...]:
    """쉼표 구분 실수 목록을 파싱한다.

    :param raw: 예 ``"0,10,-10,20,-20"``.
    :returns: 실수 튜플.
    :raises argparse.ArgumentTypeError: 파싱에 실패하거나 값이 없는 경우.
    """
    try:
        values = tuple(float(item) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"실수 목록을 해석할 수 없습니다: {raw!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("값이 하나 이상 필요합니다.")
    return values


def _parse_ints(raw: str) -> tuple[int, ...]:
    """쉼표 구분 정수 목록을 파싱한다.

    :param raw: 예 ``"150,200,300"``.
    :returns: 정수 튜플.
    :raises argparse.ArgumentTypeError: 파싱에 실패하거나 값이 없는 경우.
    """
    try:
        values = tuple(int(item) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"정수 목록을 해석할 수 없습니다: {raw!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("값이 하나 이상 필요합니다.")
    return values


def _write_png(path: Path, image) -> None:
    """이미지를 PNG 로 저장한다(한글 경로 안전).

    ``cv2.imwrite`` 는 Windows 에서 비ASCII 경로를 처리하지 못하므로
    메모리에서 인코딩한 뒤 바이트로 기록한다.

    :param path: 저장 경로.
    :param image: ``(H, W, 3)`` uint8 BGR 배열.
    :returns: ``None``.
    :raises RuntimeError: PNG 인코딩에 실패한 경우.
    """
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG 인코딩에 실패했습니다: {path}")
    path.write_bytes(buffer.tobytes())


def generate_split(
    split: str,
    out_dir: Path,
    *,
    count: int,
    seed: int,
    angles: Sequence[float],
    dpis: Sequence[int],
    with_written: bool,
    verbose: bool = True,
) -> list[dict[str, object]]:
    """한 split 을 렌더링해 디스크에 저장한다.

    :param split: split 이름(``"train"`` / ``"val"`` / ``"test"``).
    :param out_dir: 최상위 출력 디렉터리.
    :param count: 생성할 샘플 수(1 이상).
    :param seed: 기본 난수 시드.
    :param angles: 회전각 후보(도).
    :param dpis: 해상도 후보(dpi).
    :param with_written: 체크·서명 기입본도 함께 저장할지 여부.
    :param verbose: 진행 상황을 표준 출력에 표시할지 여부.
    :returns: 매니페스트 항목 목록.
    :raises ValueError: ``count`` 가 1 미만인 경우.
    """
    if count < 1:
        raise ValueError(f"생성 개수는 1 이상이어야 합니다: {count}")
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    base = FormSpec(document_id=f"{split}")
    split_seed = seed + _SPLIT_SEED_OFFSET.get(split, 0)
    entries: list[dict[str, object]] = []

    for index, spec in enumerate(
        iter_spec_grid(
            angles=tuple(angles), dpis=tuple(dpis), count=count, base=base, seed=split_seed
        )
    ):
        stem = f"{split}_{index:04d}"
        form = make_application_form(spec)
        image_path = split_dir / f"{stem}.png"
        truth_path = split_dir / f"{stem}.truth.json"
        _write_png(image_path, form.image)
        truth_path.write_text(form.truth.to_json(), encoding="utf-8")

        entry: dict[str, object] = {
            "stem": stem,
            "split": split,
            "image": image_path.relative_to(out_dir).as_posix(),
            "truth": truth_path.relative_to(out_dir).as_posix(),
            "document_id": spec.document_id,
            "dpi": spec.dpi,
            "rotation_deg": spec.rotation_deg,
            "perspective_strength": spec.perspective_strength,
            "margin_px": spec.margin_px,
            "background_gray": spec.background_gray,
            "noise_sigma": spec.noise_sigma,
            "blur_ksize": spec.blur_ksize,
            "jpeg_quality": spec.jpeg_quality,
            "illumination_gradient": spec.illumination_gradient,
            "include_representative": spec.include_representative,
            "seed": spec.seed,
            "image_size_px": list(form.image_size_px),
        }

        if with_written:
            written_path = split_dir / f"{stem}.written.png"
            _write_png(
                written_path,
                render_written(
                    form,
                    checked_option=AGREE_LABEL,
                    sign=True,
                    sign_representative=spec.include_representative,
                ),
            )
            entry["written_image"] = written_path.relative_to(out_dir).as_posix()

        entries.append(entry)
        if verbose and (index + 1) % 10 == 0:
            print(f"  [{split}] {index + 1}/{count} 생성 완료")

    if verbose:
        print(f"[{split}] 총 {count}건 저장: {split_dir}")
    return entries


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 만든다.

    :returns: 설정이 끝난 :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="make_fixtures",
        description="정답을 아는 합성 한국어 신청서 픽스처(PNG + 정답 JSON)를 생성한다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "fixtures",
        help="출력 디렉터리. 기본값은 저장소 하위 fixtures/ 다.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="난수 시드.")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="test",
        help="생성할 데이터셋 split. all 이면 세 split 을 모두 만든다.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="생성 개수. 지정하면 split 기본 개수(train 200/val 30/test 20)를 덮어쓴다.",
    )
    parser.add_argument(
        "--train-count", type=int, default=SPLIT_DEFAULT_COUNTS["train"], help="train 개수."
    )
    parser.add_argument(
        "--val-count", type=int, default=SPLIT_DEFAULT_COUNTS["val"], help="val 개수."
    )
    parser.add_argument(
        "--test-count", type=int, default=SPLIT_DEFAULT_COUNTS["test"], help="test 개수."
    )
    parser.add_argument(
        "--angles",
        type=_parse_floats,
        default=DEFAULT_ANGLES,
        help="회전각 후보(도), 쉼표 구분. 예: 0,10,-10,20,-20",
    )
    parser.add_argument(
        "--dpis",
        type=_parse_ints,
        default=DEFAULT_DPIS,
        help="해상도 후보(dpi), 쉼표 구분. 예: 150,200,250,300",
    )
    parser.add_argument(
        "--written",
        action="store_true",
        help="체크·서명이 기입된 이미지(.written.png)도 함께 저장한다.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일을 쓰지 않고 생성될 사양만 출력한다.",
    )
    parser.add_argument("--quiet", action="store_true", help="진행 출력을 억제한다.")
    return parser


def _resolve_counts(args: argparse.Namespace) -> list[tuple[str, int]]:
    """실행할 ``(split, count)`` 목록을 결정한다.

    :param args: 파싱된 CLI 인자.
    :returns: ``(split 이름, 개수)`` 목록.
    """
    per_split = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    return [
        (name, args.count if args.count is not None else per_split[name])
        for name in splits
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 진입점.

    :param argv: 명령행 인자. ``None`` 이면 :data:`sys.argv` 를 쓴다.
    :returns: 프로세스 종료 코드(성공 시 0).
    """
    args = build_parser().parse_args(argv)
    verbose = not args.quiet
    # Windows 콘솔 기본 코드페이지에서 한국어 출력이 깨지지 않도록 맞춘다.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass  # 리다이렉트된 스트림 등 재설정이 불가한 경우는 그대로 진행한다.
    out_dir: Path = args.out

    if verbose:
        diag = font_diagnostics()
        if not diag["available"]:
            print("경고: 한글 폰트를 찾지 못했습니다. 텍스트는 기본 폰트로 렌더링됩니다.")
        else:
            print(f"폰트: {diag['regular']}")

    plan = _resolve_counts(args)

    if args.dry_run:
        for split, count in plan:
            base = FormSpec(document_id=f"{split}")
            split_seed = args.seed + _SPLIT_SEED_OFFSET.get(split, 0)
            print(f"[{split}] {count}건 (dry-run, 파일을 쓰지 않음)")
            for index, spec in enumerate(
                iter_spec_grid(
                    angles=tuple(args.angles),
                    dpis=tuple(args.dpis),
                    count=min(count, 5),
                    base=base,
                    seed=split_seed,
                )
            ):
                print(
                    f"  {split}_{index:04d}: dpi={spec.dpi} 회전={spec.rotation_deg}도 "
                    f"원근={spec.perspective_strength} 여백={spec.margin_px}px "
                    f"jpeg={spec.jpeg_quality} 잡음={spec.noise_sigma}"
                )
            if count > 5:
                print(f"  ... 이하 {count - 5}건 생략")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for split, count in plan:
        manifest.extend(
            generate_split(
                split,
                out_dir,
                count=count,
                seed=args.seed,
                angles=args.angles,
                dpis=args.dpis,
                with_written=args.written,
                verbose=verbose,
            )
        )

    index_path = out_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "angles": list(args.angles),
                "dpis": list(args.dpis),
                "with_written": bool(args.written),
                "items": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if verbose:
        print(f"매니페스트 저장: {index_path} (총 {len(manifest)}건)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 진입점
    raise SystemExit(main())
