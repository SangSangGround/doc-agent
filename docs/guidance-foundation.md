# 지원 문서 선정 전 공통 안내 기능

이 경로는 LLM을 호출하거나 문장을 생성하지 않는다. 문서 종류·버전·페이지를
확인한 뒤 승인된 문구를 검색하고 **공백과 줄바꿈까지 그대로** 반환한다.
기존 `docagent.demo`의 생성형 설명 흐름과 별개이며 실제 지원 문서는 아직 없다.

## 실행 환경

Python 3.10 이상에서 가상환경을 만든다. 이 작업의 검증 환경은 저장소의 `venv/`다.
기존 `.venv`는 수정하지 않았다. Tesseract는 OS에 별도로 설치해야 한다.

```sh
source venv/bin/activate
python -m pip install -e '.[dev,ocr,roboflow]'
```

## 승인된 안내 검색

```sh
python -m docagent.guide \
  --catalog examples/guidance/catalog.demo.json \
  --observation examples/guidance/observation.demo.json --demo
```

종료 코드: 일치 0, 검색 거절/실패 2, 잘못된 파일/실행 오류 1.
`--output outputs/guidance.json`으로 결과를 저장할 수 있다.
데모 목록은 `--demo`가 없으면 거절한다. production 예시는 빈 목록으로 제공한다.
가상 안내의 승인 기록은 테스트 픽스처일 뿐 실제 문서의 사람 검토를 의미하지 않는다.

### 안내 목록 계약

- 루트: `schema_version: 1`, `purpose: demo | production`, `fields: [...]`.
- 항목: `document_id`, `document_version`, `page`(1부터), `field_id`,
  `field_type`(`checkbox` 또는 `signature`), `surrounding_text`, `keywords`,
  `forbidden_keywords`, `instruction`, `approval`.
- `keywords`는 전부 일치해야 하며 `forbidden_keywords`는 하나라도 있으면 제외한다.
  NFKC·대소문자·공백만 정규화한다. 의미 검색·오탈자 보정·부분 점수 순위는 없다.
  키워드가 과도하게 포괄적이면 오매칭할 수 있어 실제 목록의 사람 검토가 필요하다.
- `approval.status`: `draft` 또는 `approved`. 승인에는 `reviewed_by`, `reviewed_at`,
  `content_sha256`가 필요하다. `docagent.guidance.rules.content_digest(entry)`가
  반환하는 값은 문구와 매칭 조건을 묶는다. 변경 후 해시가 달라지면 로딩을 거절한다.
- 승인 해시는 변경 감지용이며 전자서명이나 신원 인증이 아니다. 검토자가 실제로
  검토한 후 승인 파일을 저장해야 하며, 파일 쓰기 권한과 Git 리뷰를 별도로 관리한다.
  OCR 결과나 외부 응답을 안내 목록으로 로드하면 안 된다.

### 관측값 계약

`observation.demo.json`의 형식을 사용한다. 문서 ID·버전·페이지·항목 유형·주변 OCR
문구와 문서 식별/탐지/OCR 각각의 신뢰도가 필요하다. `identity_verified`는 기본
false, `needs_review`는 기본 true다. 모두 충분한 근거가 있을 때만 호출자가 바꾼다.
세 신뢰도는 모두 0.85 이상이어야 한다. 이 숫자는 확률로 보정된 정확도가 아니다.

Step 2 JSON의 `document_id`는 파일명일 수 있으므로 지원 문서 ID로 바로 사용하지
않는다. 문서 자동 식별은 아직 구현되지 않았다. `regions` 결과의 검토 경고를 지워
실제 사용자에게 안내해서도 안 된다. 문서 식별·항목 문맥을 별도로 검토해야 한다.

미등록 문서/버전/페이지, 미승인 항목, 모호한 후보, 낮은 신뢰도는 모두 고정된
실패 문구를 반환한다. 반환 JSON은 관측 OCR 원문을 복사하지 않는다.

## 이미지 마스킹과 저장 정책

```sh
python -m docagent.mask_image blank-form.png --output-dir outputs/mask-review --dpi 300
```

네트워크 호출 없이 `normalized.png`와 미승인 `mask-plan.json`을 만든다.
검토 이미지를 보고 `boxes`에 `[x, y, width, height]` 픽셀 좌표를 입력한다.
`padding_px`는 박스 바깥에 추가할 여백(기본 3픽셀)이다. 필요한 영역이 모두
포함되는지 사람이 검토한 후 아래와 같이 승인 기록과 변경 감지 해시를 저장한다.
가려야 할 영역을 자동 발견하거나 누락이 없다고 인증하는 기능은 아니다.

```python
import json
from pathlib import Path
from docagent.pii.image_mask import plan_digest
p = Path('outputs/mask-review/mask-plan.json')
plan = json.loads(p.read_text())
# 실제 검토 이후에만 실행: boxes 입력과 검토자/시각을 먼저 채운다.
plan['approval'] = {
    'status': 'approved', 'reviewed_by': '실제 검토자',
    'reviewed_at': '실제 검토 시각', 'content_sha256': plan_digest(plan),
}
p.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
```

```sh
python -m docagent.mask_image blank-form.png --output-dir outputs/mask-review \
  --plan outputs/mask-review/mask-plan.json --dpi 300
python -m docagent.analyze blank-form.png --detector roboflow \
  --mask-plan outputs/mask-review/mask-plan.json --dpi 300 --output outputs/analysis.json
```

미리보기 `masked.png`를 확인한 뒤 분석한다. 두 명령에는 **동일한 원본과 DPI**를
사용한다. 계획은 정합 이미지 픽셀 해시·형태·박스·여백에 묶여 있어 다른 촬영본,
해상도나 보정 결과에는 재사용할 수 없다. 잘못된/빈 박스, 미승인·변경된 계획,
다른 이미지는 전송을 차단한다. 원본을 대체 전송하는 fallback은 없다.
양식별 자동 정렬·검토된 템플릿 적용은 문서 선정 후 후속 작업이다.

CLI에서 Roboflow를 사용할 때는 마스킹 계획이 필수다. 개인정보가 없는 빈 양식을
직접 확인한 경우에 한해 `--allow-unmasked-blank`로 원본 전송을 명시할 수 있다.
이는 자동 안전 검사나 마스킹 완료 표시가 아니다. 두 옵션은 함께 쓸 수 없다.
`MaskedDetector`는 같은 좌표계의 검은 사본만 전달하므로 탐지 좌표를 그대로 쓴다.
로컬 OCR은 원래 정합 이미지에서 수행한다. 마스킹으로 탐지 품질이 바뀔 수 있다.

마스킹 보호는 새 CLI와 명시적으로 `MaskedDetector`를 적용한 경로에만 해당한다.
저수준 `RoboflowDetector.predict()`나 기존 `build_session()`을 직접 호출하는 코드는
이미지를 자동으로 가리지 않는다. 원본 외부 전송 경로로 사용하지 않아야 한다.

분석 JSON 기본값은 OCR 텍스트·항목 제목·선택지 문구·설명·경고 원문을 제외한다.
`--include-ocr-text` 또는 `build_report(..., include_text=True)`는 **로컬 검토 전용**
옵션이며 개인정보가 그대로 포함될 수 있다. 모든 개인정보를 정규식으로 제거했다고
간주하지 않는다. OCR 원문·검토 이미지·마스킹 계획은 `outputs/` 등 Git 제외 경로에서
관리하고 외부 공유 전에 별도 검토한다. 기존 테스트 산출물에 소급 적용되지는 않는다.
