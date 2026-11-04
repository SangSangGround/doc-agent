"""python -m docagent.home: 데모, 로컬 실행, 양식 준비, 체크 평가."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from docagent.home.demo import create_demo
from docagent.home.evaluation import evaluate_manifest
from docagent.home.templates import load_template, template_digest, validate_manifest


def write_json(path: Path, value: dict) -> None:
    """부분적으로 기록된 양식이나 평가 결과가 남지 않게 교체한다."""
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".home-", suffix=".json", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(body)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def draft_template(args: argparse.Namespace) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("양식 출력은 비어 있는 폴더를 지정하세요.")
    blob = args.reference.read_bytes()
    data = {
        "schema_version": 1,
        "purpose": "production",
        "document_id": args.document_id,
        "document_version": args.version,
        "title": args.title,
        "page_size_mm": [210, 297],
        "reference_image": "reference" + args.reference.suffix.lower(),
        "reference_sha256": hashlib.sha256(blob).hexdigest(),
        "source": {"reference": args.source},
        "pages": [{"page": 1, "source_text": args.text.read_text(encoding="utf-8")}],
        "groups": json.loads(args.groups.read_text(encoding="utf-8")),
        "approval": {"status": "draft"},
    }
    # 구조 확인용 복사본만 데모로 검증하고 실제 양식은 미승인 상태로 저장한다.
    validate_manifest({**data, "purpose": "demo"}, demo=True)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.reference, args.output / data["reference_image"])
    write_json(args.output / "template.json", data)
    print("미승인 양식을 생성했습니다. 전체 원문·선택지·좌표·빈 기준 이미지를 검토해야 합니다.")


def approve_template(path: Path, reviewer: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("purpose") != "production" or not reviewer.strip():
        raise ValueError("실제 문서와 검토자 이름이 필요합니다.")
    data["approval"] = {
        "status": "approved",
        "reviewed_by": reviewer,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "content_sha256": template_digest(data),
    }
    validate_manifest(data)
    # 동일 묶음 안에서 경로와 이미지 해시까지 검증한 뒤 원본을 교체한다.
    fd, name = tempfile.mkstemp(prefix=".home-validation-", suffix=".json", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, allow_nan=False)
        load_template(temporary)
        write_json(path, data)
    finally:
        temporary.unlink(missing_ok=True)
    print("검토자 승인 기록을 저장했습니다. 이 기록은 법적 인증이나 성능 검증을 대신하지 않습니다.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="외부 모델·API 키 없이 실행하는 문서 읽기와 체크 대조")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="가상 동의서와 평가용 이미지 생성")
    demo.add_argument("--output", type=Path, required=True)
    serve = sub.add_parser("serve", help="127.0.0.1에서 카메라·읽기·체크 UI 실행")
    serve.add_argument("--template", type=Path, required=True)
    serve.add_argument("--demo", action="store_true")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--session", type=Path, help="선택 ID와 읽기 설정을 저장할 로컬 JSON 경로")
    evaluate = sub.add_parser("evaluate", help="정답셋에서 체크 대조 지표 계산")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--demo", action="store_true")
    evaluate.add_argument("--output", type=Path, required=True)
    draft = sub.add_parser("draft-template", help="실제 동의서 등록을 위한 미승인 묶음 생성")
    for key in ("reference", "text", "groups", "output"):
        draft.add_argument("--" + key, type=Path, required=True)
    for key in ("document-id", "title", "source"):
        draft.add_argument("--" + key, required=True)
    draft.add_argument("--version", default="1")
    approve = sub.add_parser("approve-template", help="사람이 원문과 좌표를 검토한 뒤 승인 기록 작성")
    approve.add_argument("--template", type=Path, required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--confirm-reviewed", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            print(create_demo(args.output))
        elif args.command == "evaluate":
            report = evaluate_manifest(args.manifest, demo=args.demo)
            write_json(args.output, report)
            print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
        elif args.command == "serve":
            from docagent.home.server import HomeApplication, make_server
            if not 0 <= args.port <= 65535:
                raise ValueError("포트 범위를 확인하세요.")
            template = load_template(args.template, demo=args.demo)
            app = HomeApplication(template, session_path=args.session)
            server = make_server(app, args.port)
            print(f"문서 도우미: http://127.0.0.1:{server.server_port}  종료: Ctrl+C", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        elif args.command == "draft-template":
            draft_template(args)
        elif args.command == "approve-template":
            approve_template(args.template, args.reviewer)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
