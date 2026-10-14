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
