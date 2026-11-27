"""Gemini-based dynamic workflow extraction, WITHOUT making navigation claims.

Use with a real photo. Existing OCR and Gemini-detection JSON may be supplied as
supplementary raw evidence; no undocumented JSON layout is assumed.
"""
from __future__ import annotations
import argparse
import json
import mimetypes
import os
from pathlib import Path
from workflow_schema import GENERATE_SCHEMA, save_json, validate_workflow

SYSTEM_PROMPT = '''너는 시각장애인 종이 문서 작성 보조 소프트웨어의 문서 구조 분석기다.
사진의 인쇄된 내용과 별도로 주어진 OCR 텍스트에서 근거를 찾을 수 있는 것만 추출한다.
- 문서 제목·용도·신청 유형, 작성란, 담당자/제3자 영역, 상호배타적 체크 그룹을 구조화한다.
- ID는 유형별 type_001, 섹션 section_001, 작성란 F001처럼 유일하게 만든다.
- 동일한 작성란이 여러 신청 유형에 공통이면 필드를 중복 생성하지 말고 applicable_type_ids를 빈 배열로 둔다.
- 신청 유형 구분이 없으면 application_types는 빈 배열로 둔다.
- fields는 종이의 읽기/작성 순서대로 배열한다. section.field_ids도 그 순서를 따른다.
- role은 applicant/official/third_party/unknown 중 하나다. 애매하면 unknown.
- requirement는 required/optional/conditional/unknown 중 하나다. 명시되지 않았으면 unknown.
- 문서에 없는 법률 요건, 서류 제출 의무, 개인정보를 추측하지 말라.
- 조건부 작성은 문서에 실제 인쇄된 문구를 바탕으로 condition을 작성한다.
- 각 field의 source_text는 실제 문서에서 읽힌 짧은 원문 근거다. 불명확하면 비우고 review_required=true로 한다.
- 설명은 짧고 명확한 한국어. 외부 규정이나 단정적 법률 해설을 넣지 않는다.
- review_required는 근거/작성 주체/작성 조건/필드 구분이 불분명할 때 true다.
- 절대 pixel, bbox, 좌표, navigation_target_id, navigation_verified를 생성하지 말라.
- 다중 체크박스는 문서상 명백히 단일선택일 때만 exclusive=true.
- 출력은 제공된 JSON Schema를 따른다.'''


def generate(image_path: Path, model: str, ocr_path: Path | None = None,
             detection_path: Path | None = None) -> dict:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError('Gemini 연동에는 .venv_scan에 google-genai 설치가 필요합니다: pip install google-genai') from exc
    if not os.environ.get('GEMINI_API_KEY'):
        raise RuntimeError('GEMINI_API_KEY 환경변수가 필요합니다')
    mime, _ = mimetypes.guess_type(image_path.name)
    if mime not in {'image/jpeg', 'image/png', 'image/webp'}:
        raise ValueError('지원하는 사진 형식: jpg, png, webp')
    parts = [types.Part.from_text(text='첨부 사진의 문서 내용을 분석해 작성 Workflow를 생성해 주세요.'),
             types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime)]
    if ocr_path:
        ocr = ocr_path.read_text(encoding='utf-8-sig')
        parts.append(types.Part.from_text(text='별도 OCR 텍스트(오류 가능, 문서 사진과 대조):\n' + ocr[:35000]))
    if detection_path:
        raw = detection_path.read_text(encoding='utf-8-sig')
        parts.append(types.Part.from_text(text='기존 탐지 결과 원문(좌표는 승인되지 않은 후보이며 구조만 참고):\n' + raw[:30000]))
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role='user', parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type='application/json',
            response_json_schema=GENERATE_SCHEMA,
            temperature=0.1,
        ),
    )
    if not response.text:
        raise RuntimeError('Gemini가 빈 응답을 반환했습니다')
    workflow = json.loads(response.text)
    validate_workflow(workflow)
    workflow['schema_version'] = '0.1.0'
    workflow['document_id'] = image_path.stem
    workflow['source_image'] = image_path.name
    workflow['generation_note'] = 'AI extracted; requires field/role/condition review. No navigation approved.'
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser(description='문서 사진으로 동적 Workflow 생성')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--model', required=True, help='사용 가능한 Gemini 모델 ID')
    parser.add_argument('--ocr-text', type=Path)
    parser.add_argument('--detection-json', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.image, args.model, args.ocr_text, args.detection_json)
    save_json(args.out, result)
    print(f'[OK] {args.out}: {len(result["fields"])}개 필드, {len(result["application_types"])}개 신청 유형')
    print('[SAFE] 좌표 연결/작성 의무 검수는 아직 완료되지 않았습니다.')


if __name__ == '__main__':
    main()
