"""체크 대조의 오완료율·문제 발견율. 합성 데이터는 실제 성능과 분리해 보고한다."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from docagent.home.session import HomeSession
from docagent.home.templates import load_template
from docagent.vision.checkbox_review import verify_checkbox_groups


def _local_path(root: Path, value: str) -> Path:
    if not isinstance(value,str) or not value.strip() or Path(value).is_absolute():
        raise ValueError("평가 파일은 상대 경로를 사용하세요.")
    path=(root/value).resolve()
    path.relative_to(root.resolve())
    return path


def evaluate_manifest(path: str | Path, *, demo: bool = False) -> dict:
    path=Path(path)
    manifest=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest,dict):
        raise ValueError("평가셋은 JSON 객체여야 합니다.")
    cases=manifest.get("cases")
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("dataset_kind") not in ("synthetic","real")
            or not isinstance(cases,list) or not cases):
        raise ValueError("비어 있지 않은 평가셋과 데이터 유형이 필요합니다.")
    if manifest["dataset_kind"] == "real" and not all(isinstance(manifest.get(k),str) and manifest[k].strip() for k in ("reviewed_by","dataset_version")):
        raise ValueError("실제 평가셋은 정답 검토자와 버전이 필요합니다.")
    template=load_template(_local_path(path.parent,manifest.get("template")),demo=demo)
    if manifest["dataset_kind"] == "real" and template.manifest["purpose"] != "production":
        raise ValueError("합성 데모 양식을 실제 문서 평가셋으로 보고할 수 없습니다.")
    rows=[];seen=set()
    for case in cases:
        if (not isinstance(case,dict) or not isinstance(case.get("id"),str) or not case["id"] or case["id"] in seen
                or type(case.get("expected_complete")) is not bool or type(case.get("expected_problem")) is not bool
                or type(case.get("expected_uncertain",False)) is not bool):
            raise ValueError("평가 사례의 ID와 정답 상태를 확인하세요.")
        if sum((case["expected_complete"],case["expected_problem"],case.get("expected_uncertain",False))) != 1:
            raise ValueError("완료·문제·불확실 중 정확히 하나의 정답 상태가 필요합니다.")
        seen.add(case["id"])
        session=HomeSession(template)
        choices=case.get("choices")
        if not isinstance(choices,dict) or set(choices)!={g["id"] for g in template.manifest["groups"]}:
            raise ValueError("모든 항목의 사용자 선택이 필요합니다.")
        for group_id,options in choices.items():
            session.propose_selection(group_id,options);session.confirm_selection(group_id)
        image_path=_local_path(path.parent,case.get("image"))
        image=cv2.imdecode(np.frombuffer(image_path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        result=verify_checkbox_groups(template.reference,image,template.manifest["groups"],
                                      page_size_mm=tuple(template.manifest["page_size_mm"]),capture_mode=case.get("capture_mode","photo"))
        view=session.apply_verification(result)
        statuses={g["id"]:g["verification"]["status"] for g in view["groups"]}
        problem=any(s in ("mismatch","unmarked","multiple") for s in statuses.values())
        uncertain=any(s=="uncertain" for s in statuses.values())
        rows.append(dict(id=case["id"], expected_complete=case["expected_complete"],expected_problem=case["expected_problem"],
                         expected_uncertain=case.get("expected_uncertain",False),completed=view["complete"],
                         problem_detected=problem,uncertain=uncertain,statuses=statuses,reason=result.get("reason")))
    negatives=[r for r in rows if not r["expected_complete"]]
    problems=[r for r in rows if r["expected_problem"]]
    positives=[r for r in rows if r["expected_complete"]]
    false_completion=sum(r["completed"] for r in negatives)
    detected=sum(r["problem_detected"] for r in problems)
    successful=sum(r["completed"] for r in positives)
    def rate(n: int,d: int) -> float | None:
        return n/d if d else None
    return dict(schema_version=1,dataset_kind=manifest["dataset_kind"],sample_count=len(rows),
                template_sha256=template.manifest_sha256,
                metrics={"false_completion":{"count":false_completion,"denominator":len(negatives),"rate":rate(false_completion,len(negatives))},
                         "problem_detection":{"count":detected,"denominator":len(problems),"rate":rate(detected,len(problems))},
                         "correct_completion":{"count":successful,"denominator":len(positives),"rate":rate(successful,len(positives))},
                         "uncertain":{"count":sum(r["uncertain"] for r in rows),"denominator":len(rows),"rate":rate(sum(r["uncertain"] for r in rows),len(rows))}},
                cases=rows,notice="합성 테스트는 실제 문서 정확도나 시각장애인 사용성을 증명하지 않습니다." if manifest["dataset_kind"]=="synthetic" else "고정된 정답셋에 대한 결과입니다. 사용자 과제 성공률은 별도 평가가 필요합니다.")
