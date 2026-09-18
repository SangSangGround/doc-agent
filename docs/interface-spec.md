# doc-agent 인터페이스 규격 (Interface Spec)

이 문서는 Vision / PII / Agent / IO 모듈이 공유하는 **단일 계약**을 정의한다.
계약의 실체는 코드다. 이 문서와 코드가 어긋나면 **코드가 정답**이다.

| 항목 | 위치 |
|---|---|
| 데이터 계약 (dataclass · Enum · 상수) | `src/docagent/contracts.py` |
| 경계 인터페이스 (Protocol) | `src/docagent/interfaces.py` |
| 도메인 예외 계층 | `src/docagent/errors.py` |
| 계약 검증 테스트 | `tests/test_contracts.py` |

---

## 1. 좌표계 규약

| 규약 | 값 |
|---|---|
| 단위 | **밀리미터(mm)**. 픽셀 아님 |
| 원점 | 페이지 **좌상단** (0, 0) |
| x 방향 | 오른쪽이 **+** |
| y 방향 | **아래쪽이 +** (수학 좌표계와 반대. 이미지 좌표계와 동일) |
| 기준 용지 | A4 = `210.0 mm × 297.0 mm` (`A4_WIDTH_MM`, `A4_HEIGHT_MM`) |
| 픽셀 좌표 | `BoxPx`. **Vision 모듈 내부 전용.** 모듈 경계를 넘지 못한다 |

Vision 은 픽셀로 탐지하고 **내보내기 직전에 mm 로 변환**한다.
`Detection` · `OcrWord` · `Field` · `Option` 이 담는 좌표는 전부 `BoxMm` 이다.
액추에이터 목표 좌표는 `BoxMm.center()` → `Point` 로 얻는다.

```
(0,0) ──────────────► x_mm (최대 210.0)
  │
  │      ┌──────────────┐  ← BoxMm(x_mm, y_mm, w_mm, h_mm)
  │      │      ·       │     · = center() = (x_mm + w_mm/2, y_mm + h_mm/2)
  │      └──────────────┘
  ▼
y_mm (최대 297.0)
```

---

## 2. 기하 타입

| 타입 | 필드 | 메서드 | 비고 |
|---|---|---|---|
| `Point` | `x_mm`, `y_mm` | `to_tuple()` | frozen |
| `BoxMm` | `x_mm`, `y_mm`, `w_mm`, `h_mm` | `center()`, `contains(point)`, `to_tuple()`, 속성 `right_mm` / `bottom_mm` / `area_mm2` | 폭·높이 음수는 생성 시 `ValueError` |
| `BoxPx` | `x`, `y`, `w`, `h` (int) | `to_tuple()` | Vision 내부 전용 |

`contains()` 는 **경계를 포함**한다(닫힌 구간).

---

## 3. 열거형

| Enum | 값 (직렬화 문자열) | 의미 |
|---|---|---|
| `FieldType` | `signature` / `checkbox` / `choice` / `text_input` / `date` / `unknown` | 기입란 유형 |
| `FieldRole` | `applicant` / `representative` / `official` / `unknown` | 기입 주체 |
| `Sensitivity` | `public` / `private` | 공개 정보 영역 / 개인정보 영역 |

JSON 에는 항상 **문자열 `value`** 로 나간다. `from_dict()` 는 문자열과 Enum 인스턴스를
모두 받으며, 정의되지 않은 값은 한국어 메시지와 함께 `ValueError` 로 거부한다.

---

## 4. 데이터 계약

### 4.1 Vision 산출물

| 타입 | 필드 | 설명 |
|---|---|---|
| `Detection` | `type`, `box_mm`, `confidence`, `source` | 탐지기 원시 출력 1건. `source` 는 `"opencv_contour"` / `"yolo"` 등 |
| `OcrWord` | `text`, `box_mm`, `confidence` | OCR 단어 1개 |
| `Option` | `label`, `box_mm`, `checked` | 선택지 1개. `checked=None` 은 **미확인**, `False` 는 **명시적 미체크** — 구분해서 다룬다. 단, Vision 구조화 산출물에서는 언제나 `None` 이며(검증 결과를 구조에 되먹이지 않는다) `True`/`False` 는 정답 데이터에서만 쓰인다. 현재 표시 상태는 `verify_options()` 로 판정한다 |
| `Field` | `id`, `type`, `title`, `role`, `options`, `required`, `sensitivity`, `box_mm`, `clause_text`, `order`, `confidence` | 기입란 1개. `id` 만 필수, 나머지는 안전한 기본값 |
| `DocumentStructure` | `document_id`, `doc_title`, `fields`, `page_size_mm`, `source_image`, `warnings` | 문서 해석 결과 **전체 = fields JSON** |
| `VerificationResult` | `field_id`, `written`, `ink_ratio_before`, `ink_ratio_after`, `confidence`, `reason` | Verify 단계 판정. 속성 `ink_delta` |

`Field` 기본값은 **안전한 쪽**으로 잡혀 있다. `sensitivity` 기본값은 `PRIVATE`,
`type` · `role` 기본값은 `UNKNOWN` 이다. 분류에 실패한 항목이 실수로 공개 영역에
들어가는 일을 막기 위해서다.

`DocumentStructure` 는 생성 시 **항목 id 중복을 거부**한다. 에이전트가 "세 번째 항목"을
지시할 때 대상이 모호해지는 것을 원천 차단한다.

주요 메서드

| 메서드 | 반환 | 설명 |
|---|---|---|
| `required_fields()` | `tuple[Field, ...]` | 필수 항목만 `order` 순 |
| `field_by_id(field_id)` | `Field \| None` | 없으면 `None` |
| `public_payload()` | `dict` | **외부 LLM 전송용** 공개 영역 payload (§6) |
| `to_dict()` / `to_json()` | `dict` / `str` | fields JSON |
| `from_dict()` / `from_json()` | `DocumentStructure` | 무손실 복원 |

### 4.2 PII

| 타입 | 필드 | 설명 |
|---|---|---|
| `PiiSpan` | `start`, `end`, `pii_type`, `raw_len`, `confidence` | 개인정보 구간 **메타데이터** |
| `SanitizedText` | `text`, `spans`, `blocked` | 마스킹 완료 텍스트. 속성 `has_pii` |

> **`PiiSpan` 은 원문 값을 절대 담지 않는다.**
> 위치·유형·길이만 남긴다. 로그나 세션 파일에 그대로 남아도 원문을 복원할 수 없어야 한다.
> `value` · `raw` · `text` 같은 필드를 추가하는 것은 **계약 위반**이며
> `tests/test_contracts.py::test_pii_span_never_carries_raw_value` 가 이를 강제한다.
> `spans` 의 인덱스는 **마스킹 전 원문 기준**이다.

### 4.3 검색 · Agent

| 타입 | 필드 | 설명 |
|---|---|---|
| `RetrievedChunk` | `chunk_id`, `text`, `source`, `score` | RAG 근거 조각. `source` 는 사용자에게 낭독 가능한 출처 표기 |
| `ToolCall` | `name`, `arguments` | 도구 호출 요청 |
| `ToolResult` | `ok`, `speech`, `state_patch`, `error`, `handoff` | 도구 실행 결과 |
| `AgentTurn` | `user_text`, `intent`, `tool_calls`, `speech`, `confidence`, `handoff_reason` | 대화 1턴의 반환값. 메서드 `needs_handoff()`. **세션 복원 단위가 아니다** — 복원 정본은 `SessionState`, 감사 추적은 `SessionState.history` 다. `user_text` 에 원문 발화가 들어 있어 세션 JSON 에 저장하지 않는다 |

---

## 5. 신뢰도 임계값

| 상수 | 값 | 분기 |
|---|---|---|
| `EXPLAIN_THRESHOLD` | 0.85 | 이상 → **단정적으로 설명**한다 |
| `PARTIAL_THRESHOLD` | 0.70 | 이상 ~ 0.85 미만 → **부분 확신**. 재확인 질문을 덧붙인다 |
| — | — | 0.70 미만 → **사람 지원**(`HandoffRequired`)으로 넘긴다 |
| `VISION_TRUST_THRESHOLD` | 0.85 | Vision 탐지 결과를 추가 확인 없이 신뢰하는 하한 |
| `MOTION_TOLERANCE_MM` | 1.0 | 펜이 목표 지점에 도달했다고 보는 허용 오차(mm). 초과하면 사람 지원으로 넘긴다 |

`AgentTurn.needs_handoff()` 는 `handoff_reason` 이 채워진 턴에서만 `True` 를 반환한다.
신뢰도는 이 판정에 쓰지 않는다 — 위 임계값은 설명 생성 경로의 밴드 분기
(`docagent.agent.confidence.band_for`)에 쓰이고, 실제로 사람에게 넘겼는지는
오케스트레이터가 `handoff_reason` 으로만 표시한다. 두 기준을 섞으면
"알아듣지 못한 첫 턴"(재질문, `confidence=0.0`)까지 직원 호출로 오독된다.

---

## 6. 공개 정보 영역 / 개인정보 영역 분리 원칙

**원칙**: 개인정보 영역의 내용은 프로세스 경계를 넘지 않는다.
외부 LLM·네트워크·영구 로그로 나가는 것은 `Sensitivity.PUBLIC` 항목과
`PiiGate` 를 통과한 문자열뿐이다.

| 구분 | 대상 | 외부 전송 |
|---|---|---|
| `PUBLIC` — 공개 정보 영역 | 약관 문구, 안내문, 항목 라벨, 선택지 라벨 등 **문서에 인쇄된 내용** | 허용 |
| `PRIVATE` — 개인정보 영역 | 성명, 주민등록번호, 연락처, 주소, 계좌번호, 서명 등 **이용자가 기입하는 값** | 금지 |

`DocumentStructure.public_payload()` 의 동작:

* `PUBLIC` 항목 → `title` · `clause_text` · `options` 라벨까지 포함하되
  **좌표(`box_mm`)와 신뢰도(`confidence`)는 제외**한다
* `PRIVATE` 항목 → **구조 정보만** 남긴 축약형으로 강등
  * 남는 키: `id`, `type`, `role`, `required`, `order`, `sensitivity`, `redacted: true`
  * 제거되는 키: `title`, `clause_text`, `options`, `box_mm`, `confidence`
* 항목은 `order` 순으로 정렬되고, 강등된 항목 id 는 `redacted_field_ids` 에 모인다

에이전트는 "3번째 자리에 서명란이 있고 필수다"라는 **구조**는 알 수 있지만
그 항목의 문구와 좌표는 알 수 없다. 좌표가 필요한 시점(Act 단계)에는
LLM 이 아니라 로컬 코드가 원본 `DocumentStructure` 에서 직접 읽는다.
따라서 공개 payload 에는 어느 구현에서도 좌표가 실리지 않는다.

운영 경로의 정본은 `docagent.pii.policy.build_public_payload()` 다. 같은 키 집합을
만들되 항목 **내용**에 탐지기를 한 번 더 돌려, `PUBLIC` 으로 선언된 항목에 값이
새어 들어온 경우까지 강등한다. `DocumentStructure.public_payload()` 는 탐지기에
의존하지 않는 계약 기본 구현이며, 두 구현의 좌표 제거 규율은 동일하다.

추가로, 외부로 나가는 **모든** 문자열은 `PiiGate.sanitize()` 또는
`PiiGate.assert_clean()` 을 통과해야 한다. 마스킹으로도 안전을 보장할 수 없으면
`SanitizedText.blocked=True` 로 표시하고 호출자가 `PiiEgressBlocked` 를 던진다.

---

## 7. fields JSON — Vision → Agent 인터페이스

`DocumentStructure.to_dict()` 의 결과가 **fields JSON** 이며, Vision 과 Agent 사이의
**유일한 공식 인터페이스**다. Agent 는 Vision 의 내부 구현을 알지 못하고,
Vision 은 Agent 의 대화 로직을 알지 못한다.

```
[이미지] ──Vision──► fields JSON ──Agent──► 발화 · 도구 호출 ──► [액추에이터]
                          ▲                                          │
                          └────────── VerificationResult ────────────┘
```

### 7.1 fields JSON 전문 예시

```json
{
  "document_id": "doc_20260909_0001",
  "doc_title": "예금계좌 개설 신청서",
  "fields": [
    {
      "id": "consent_01",
      "type": "checkbox",
      "title": "개인정보 수집·이용 동의",
      "role": "applicant",
      "options": [
        { "label": "동의함" },
        { "label": "동의하지 않음" }
      ],
      "required": true,
      "sensitivity": "public",
      "clause_text": "수집 항목: 성명, 연락처, 주소. 이용 목적: 계좌 개설 및 본인 확인. 보유 기간: 거래 종료 후 5년.",
      "order": 1
    },
    {
      "id": "name_01",
      "type": "text_input",
      "title": "성명",
      "role": "applicant",
      "options": [],
      "required": true,
      "sensitivity": "private",
      "box_mm": { "x_mm": 45.0, "y_mm": 150.0, "w_mm": 60.0, "h_mm": 10.0 },
      "clause_text": "",
      "order": 2,
      "confidence": 0.9
    },
    {
      "id": "signature_01",
      "type": "signature",
      "title": "신청인 서명",
      "role": "applicant",
      "options": [],
      "required": true,
      "sensitivity": "private",
      "box_mm": { "x_mm": 130.0, "y_mm": 250.0, "w_mm": 50.0, "h_mm": 15.0 },
      "clause_text": "위 내용을 모두 확인하였습니다.",
      "order": 3,
      "confidence": 0.88
    }
  ],
  "page_size_mm": [210.0, 297.0],
  "source_image": "scans/20260909_0001.png",
  "warnings": ["3번 항목의 경계선이 흐릿하여 좌표 오차가 있을 수 있습니다."]
}
```

### 7.2 같은 문서의 `public_payload()` 결과

LLM 요청 본문에는 **이것만** 실린다. `name_01` · `signature_01` 의 제목이 사라지고,
`consent_01` 에서도 좌표(`box_mm`)와 신뢰도(`confidence`)가 빠진 것을 확인하라.

```json
{
  "document_id": "doc_20260909_0001",
  "doc_title": "예금계좌 개설 신청서",
  "page_size_mm": [210.0, 297.0],
  "fields": [
    {
      "id": "consent_01",
      "type": "checkbox",
      "title": "개인정보 수집·이용 동의",
      "role": "applicant",
      "options": [
        { "label": "동의함" },
        { "label": "동의하지 않음" }
      ],
      "required": true,
      "sensitivity": "public",
      "clause_text": "수집 항목: 성명, 연락처, 주소. 이용 목적: 계좌 개설 및 본인 확인. 보유 기간: 거래 종료 후 5년.",
      "order": 1
    },
    { "id": "name_01", "type": "text_input", "role": "applicant", "required": true, "order": 2, "sensitivity": "private", "redacted": true },
    { "id": "signature_01", "type": "signature", "role": "applicant", "required": true, "order": 3, "sensitivity": "private", "redacted": true }
  ],
  "redacted_field_ids": ["name_01", "signature_01"],
  "warnings": ["3번 항목의 경계선이 흐릿하여 좌표 오차가 있을 수 있습니다."]
}
```

### 7.3 직렬화 규약

* 모든 dataclass 는 `to_dict()` / `from_dict()` 를 가지며 **`from_dict(x.to_dict()) == x`** 가 성립한다.
  이것이 "세션 복원율 100%" KPI 의 기술적 근거다.
* `Enum` → 문자열 `value`. `tuple` → JSON `list` → 복원 시 다시 `tuple`.
* `to_json()` 은 `ensure_ascii=False` 로 직렬화한다. 한글이 `\uXXXX` 로 깨지지 않는다.
* 파일 입출력은 항상 `pathlib` + `encoding="utf-8"`.
* 필수 키가 없거나 Enum 값이 정의 밖이면 **한국어 메시지와 함께 `ValueError`** 를 던진다.
  조용히 기본값으로 때우지 않는다.

---

## 8. Protocol 인터페이스

전부 `typing.Protocol` + `@runtime_checkable`. 구현체는 상속할 필요가 없고
**시그니처만 맞추면** 된다.

| Protocol | 메서드 | 반환 | 단계 |
|---|---|---|---|
| `Detector` | `detect(image)` | `list[Detection]` | See |
| `OcrEngine` | `read(image)` | `list[OcrWord]` | Understand |
| `PiiGate` | `sanitize(text)` | `SanitizedText` | 전 단계 관문 |
| | `assert_clean(text)` | `None` (위반 시 `PiiEgressBlocked`) | |
| `Retriever` | `search(query, k=5)` | `list[RetrievedChunk]` | Explain |
| `LlmClient` | `complete(system, user, max_tokens=1024)` | `str` | Explain / Ask |
| `MotionController` | `move_to(x_mm, y_mm)` | `bool` | Act |
| | `home()` | `bool` | |
| | `position()` | `Point` | |
| `SpeechIO` | `speak(text)` | `None` | Explain / Ask |
| | `listen(timeout_s=10.0)` | `str` | |
| `Clock` | `now_iso()` | `str` | 전 단계 |

`ImageArray` 는 `typing.Any` 별칭이다. 실제 런타임 타입은 `numpy.ndarray`
(`(H, W, 3)` uint8 BGR 또는 `(H, W)` uint8 그레이스케일)이지만,
계약 계층이 numpy 에 의존하지 않도록 `Any` 로 둔다.

**`Clock` 이 존재하는 이유**: 결정론적 테스트. 구현체가 `datetime.now()` 를 직접
호출하는 것을 금지하고 항상 `Clock` 을 주입받는다. `random` 을 쓸 때는 seed 를 고정한다.

---

## 9. 예외 계층

```
DocAgentError
├── VisionError
│   ├── DocumentNotFoundError   (source)
│   └── LowConfidenceError      (confidence, threshold, field_id)
├── PiiEgressBlocked            (pii_types, count) — 원문 값 미포함
├── HandoffRequired             (reason, field_id)
├── InvalidTransition           (current, requested)
├── ToolExecutionError          (tool_name)
└── AdapterUnavailable          (package, feature, extra)
```

규칙

* **예외를 삼키지 않는다.** 하위 계층 예외는 `raise DomainError(...) from exc` 로 감싸 올린다.
  `None` 이나 빈 리스트로 오류를 숨기는 조용한 실패는 금지한다.
* 모든 메시지는 **한국어**로 작성한다.
* `PiiEgressBlocked` 의 메시지에는 유형과 건수만 담고 **원문 값은 절대 담지 않는다.**
* `AdapterUnavailable` 은 무엇을 어떻게 설치해야 하는지 안내 문구를 스스로 만든다.

---

## 10. 선택적 어댑터 규약

`ultralytics` · `torch` · `anthropic` · `pyserial` · `pytesseract` 등은
**설치되어 있지 않다.** API 키 없이 `pytest` 와 데모가 100% 통과해야 한다.

```python
# 금지 — 모듈 최상단 import
from ultralytics import YOLO


# 필수 — 함수·생성자 내부 지연 import + 도메인 예외로 감싸기
def _load_yolo(weights_path: Path):
    """YOLO 모델을 지연 로딩한다.

    :param weights_path: 가중치 파일 경로.
    :returns: YOLO 모델 객체.
    :raises AdapterUnavailable: ultralytics 가 설치되지 않은 경우.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise AdapterUnavailable(
            package="ultralytics",
            feature="YOLO 기반 기입란 탐지",
            extra="yolo",
        ) from exc
    return YOLO(str(weights_path))
```

optional-dependencies 그룹: `yolo` / `llm` / `serial` / `ocr` / `dev`
(`pyproject.toml` 참조).

---

## 11. 계약 변경 절차

이 계약이 흔들리면 병렬 구현이 전부 깨진다. 변경이 필요하면:

1. `contracts.py` / `interfaces.py` 를 먼저 고친다.
2. `tests/test_contracts.py` 를 갱신하고 통과시킨다.
   (`test_all_contract_dataclasses_are_covered` 가 새 dataclass 의 왕복 테스트 누락을 잡는다.)
3. 이 문서의 해당 표와 JSON 예시를 갱신한다.
4. 영향받는 모듈 담당자에게 알린다.

계약 파일을 담당하지 않는 에이전트는 이 파일들을 **읽기만** 하고,
문제를 발견하면 직접 고치지 말고 이슈로 보고한다.
