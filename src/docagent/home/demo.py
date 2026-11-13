"""실제 동의서와 구분되는 합성 문서·정답 사례. 모델 학습에는 사용하지 않는다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from docagent.testing.fonts import load_font


def create_demo(directory: str | Path) -> Path:
    """빈 폴더에 인쇄·촬영할 수 있는 가상 동의서와 평가 사례를 만든다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("데모 출력은 비어 있는 폴더를 지정하세요.")
    scale = 6
    page = Image.new("RGB", (210 * scale, 297 * scale), "white")
    draw = ImageDraw.Draw(page)
    lines: list[str] = []

    def line(text: str, x: float, y: float, size: int = 23) -> None:
        draw.text((round(x * scale), round(y * scale)), text, fill="black", font=load_font(size_px=size))
        lines.append(text)

    title = "문서 읽기·체크 확인 연습 동의서"
    line(title, 18, 15, 34)
    line("가상 문서 / 실제 신청·동의 효력 없음", 18, 27)
    paragraphs = [
        "이 문서는 문서 읽기와 체크 확인 기능을 시험하기 위한 가상 양식입니다.",
        "아래 두 항목의 원문을 확인하고 각 항목에서 한 가지를 선택합니다.",
        "개인정보나 실제 서명을 적지 마세요. 선택 내용은 본인이 결정합니다.",
        "사진이 흐리거나 선택을 확인하기 어려우면 다시 촬영합니다.",
    ]
    for i, text in enumerate(paragraphs):
        line(text, 18, 42 + i * 7)
    groups = []
    for i, (name, text) in enumerate([
        ("읽기 연습", "연습용 문서의 내용을 음성으로 듣는 것에 동의합니다."),
        ("결과 확인 연습", "선택한 내용과 연습지의 체크 표시를 비교하는 것에 동의합니다."),
    ]):
        top = 88 + 75 * i
        draw.line((18*scale, (top-5)*scale, 192*scale, (top-5)*scale), fill="black", width=2)
        heading = f"{i+1}. {name}"
        line(heading, 18, top, 28)
        line(text, 18, top + 10)
        line("동의함 또는 동의하지 않음 중 한 가지에 표시하세요.", 18, top + 19)
        options = []
        for j, label in enumerate(("동의함", "동의하지 않음")):
            x, y = 25 + 75*j, top + 31
            draw.rectangle((x*scale, y*scale, (x+6)*scale, (y+6)*scale), outline="black", width=2)
            draw.text(((x+9)*scale, y*scale), label, font=load_font(size_px=24), fill="black")
            options.append(dict(id=("agree", "disagree")[j], label=label, box_mm=[x,y,6,6]))
        option_text = "□ 동의함  □ 동의하지 않음"
        lines.append(option_text)
        source = "\n".join([heading, text, "동의함 또는 동의하지 않음 중 한 가지에 표시하세요.", option_text])
        groups.append(dict(id=f"consent_{i+1}", source_text=source, min_selected=1, max_selected=1, options=options))
        line("표시를 바꾸려면 새로운 연습지를 사용하세요.", 18, top+46)
        line("작성 위치를 직접 찾는 기능은 이 시연 범위에 포함되지 않습니다.", 18, top+55)
    line("서명 전에 두 항목의 원문과 선택 내용을 확인하세요.", 18, 239)
    signature_source = "신청인 (서명)"
    line(signature_source, 90, 253, 27)
    draw.line((125*scale, 260*scale, 190*scale, 260*scale), fill="black", width=3)
    line("가상 신청서입니다. 실제 서명 대신 연습용 표시를 사용하세요.", 18, 273)
    reference_path = directory / "reference.png"
    page.save(reference_path)
    manifest = dict(schema_version=1, purpose="demo", document_id="synthetic-home-consent",
                    document_version="1", title=title, page_size_mm=[210,297], reference_image="reference.png",
                    reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                    source={"reference":"synthetic://home-consent-v1"}, pages=[dict(page=1, source_text="\n".join(lines))],
                    groups=groups, signatures=[{
                        "id": "applicant_signature", "role": "applicant",
                        "source_text": signature_source, "box_mm": [125, 247, 65, 13],
                        "before_signing_group_ids": [group["id"] for group in groups],
                    }], approval={"status":"draft"})
    path = directory / "template.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    blank = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    cases = []
    desired = {g["id"]:["disagree"] for g in groups}
    scenarios = {"correct":[[1],[1]], "opposite":[[0],[1]], "missing":[[],[1]], "duplicate":[[0,1],[1]]}
    for name, selections in scenarios.items():
        marked = blank.copy()
        for group, chosen in zip(groups,selections):
            for index in chosen:
                x,y,w,h = group["options"][index]["box_mm"]
                points = np.array([[(x+.8)*scale,(y+3)*scale],[(x+2.5)*scale,(y+5)*scale],[(x+5.3)*scale,(y+.8)*scale]], np.int32)
                cv2.polylines(marked,[points],False,(15,15,15),3,cv2.LINE_AA)
        cv2.imencode(".png",marked)[1].tofile(str(directory / f"{name}.png"))
        cases.append(dict(id=name,image=f"{name}.png",capture_mode="scan",choices=desired,expected_complete=name=="correct",
                          expected_problem=name!="correct"))
    blurred=cv2.GaussianBlur(blank,(41,41),0)
    cv2.imencode(".png",blurred)[1].tofile(str(directory/"blurred.png"))
    cases.append(dict(id="blurred",image="blurred.png",capture_mode="scan",choices=desired,expected_complete=False,expected_problem=False,expected_uncertain=True))
    (directory/"evaluation.json").write_text(json.dumps(dict(schema_version=1,dataset_kind="synthetic",template="template.json",cases=cases),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return path
