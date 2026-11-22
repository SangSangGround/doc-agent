"""촬영으로 확인한 등록 양식의 서명란과 원문 근거를 제공한다."""
from __future__ import annotations

from collections import Counter
from typing import Any

from docagent.home.templates import HomeTemplate

ROLE_LABELS = {"applicant": "신청인", "representative": "대리인", "staff": "담당자", "unknown": "작성 주체 미확인"}


def position_description(box: list[float], page_size: list[float]) -> str:
    """등록된 페이지 좌표를 종이 기준의 상·중·하와 좌·중·우로 읽는다."""
    x = (box[0] + box[2] / 2) / page_size[0]
    y = (box[1] + box[3] / 2) / page_size[1]
    vertical = "위쪽" if y < 1 / 3 else "가운데" if y < 2 / 3 else "아래쪽"
    horizontal = "왼쪽" if x < 1 / 3 else "중앙" if x < 2 / 3 else "오른쪽"
    return f"종이 {vertical} {horizontal}"


def document_guide(template: HomeTemplate, capture_id: str) -> dict[str, Any]:
    """양식 정합 성공 후에만 호출한다. 서명 정보 미등록을 0개로 해석하지 않는다."""
    manifest = template.manifest
    groups = {group["id"]: group for group in manifest["groups"]}
    signatures = []
    references = [{"id": "group/" + group["id"], "text": group["source_text"], "page": 1}
                  for group in manifest["groups"]]
    for signature in manifest.get("signatures", []):
        signatures.append({
            **signature,
            "role_label": ROLE_LABELS[signature["role"]],
            "position": position_description(signature["box_mm"], manifest["page_size_mm"]),
            "before_signing": [{"id": item, "text": groups[item]["source_text"]}
                               for item in signature.get("before_signing_group_ids", [])],
        })
        references.append({"id": "signature/" + signature["id"],
                           "text": signature["source_text"], "page": 1,
                           "position": signatures[-1]["position"]})
    # 질문에는 선택 항목 이외의 공통 인쇄 문구도 근거로 사용할 수 있다.
    for index, paragraph in enumerate(manifest["pages"][0]["source_text"].splitlines()):
        if paragraph.strip() and not any(paragraph in item["text"] for item in references):
            references.append({"id": f"page/line/{index}", "text": paragraph, "page": 1})
    # 작은 로컬 모델이 서로 다른 경로형 ID의 접두사를 섞지 않도록
    # 현재 촬영 세션 안에서만 사용하는 짧은 근거 번호를 부여한다.
    for index, reference in enumerate(references, start=1):
        reference["id"] = f"S{index}"
    if "signatures" not in manifest:
        summary = "지원 양식을 확인했습니다. 이 양식의 서명란 정보는 아직 등록되지 않았습니다."
    elif not signatures:
        summary = "지원 양식을 확인했습니다. 등록된 양식에는 서명란이 없습니다."
    else:
        counts = Counter(signature["role_label"] for signature in signatures)
        descriptions = [f"{role} 서명란이 {count}개" for role, count in counts.items()]
        summary = "이 문서에는 " + ", ".join(descriptions) + " 있습니다."
        for signature in signatures:
            summary += f" {signature['role_label']} 서명란은 {signature['position']}에 있습니다."
        related_ids = list(dict.fromkeys(item for signature in signatures
                                        for item in signature.get("before_signing_group_ids", [])))
        if related_ids:
            summary += " 서명 전에 다음 원문 항목을 확인해 주세요.\n"
            summary += "\n\n".join(groups[item]["source_text"] for item in related_ids)
    return {"capture_id": capture_id, "title": manifest["title"], "summary": summary,
            "signatures_reviewed": "signatures" in manifest, "signatures": signatures,
            "references": references, "source": "registered_template_matched_to_capture"}
