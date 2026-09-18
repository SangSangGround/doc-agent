# doc-agent

시각장애인이 공공·금융 **종이문서**를 타인의 대필 없이 스스로 **이해·선택·작성·확인**하도록 돕는
AI 음성·촉각 문서작성 에이전트.

하드웨어(펜 액추에이터)는 에이전트의 **Actuator 일 뿐**이다. 판단·설명·확인의 주체는 AI 에이전트다.

---

## 1. 시스템 골격

```
See ──► Understand ──► Explain ──► Ask ──► Act ──► Verify
 │           │            │          │       │        │
 │           │            │          │       │        └─ 기입 전후 잉크 비율 비교로
 │           │            │          │       │           실제로 써졌는지 확인
 │           │            │          │       └─ 좌표(mm)로 펜 이동 · 기입
 │           │            │          └─ 사용자의 선택을 음성으로 확인받음
 │           │            └─ 약관·항목을 근거와 함께 한국어로 낭독
 │           └─ 항목 유형·역할·필수 여부·민감도 판별 → fields JSON
 └─ 문서 이미지 정합 · 기입란 탐지
```

---

## 2. 아키텍처

```
                       ┌─────────────────────────────┐
   문서 스캔 이미지 ───►│  Vision  (See/Understand)   │
                       │  정합 · 탐지 · OCR · 구조화 │
                       └──────────────┬──────────────┘
                                      │  fields JSON
                                      │  (DocumentStructure.to_dict())
                                      ▼
                       ┌─────────────────────────────┐
                       │  Agent  (Explain/Ask/Act)   │
                       │  대화 · 도구 오케스트레이션 │
                       └───┬──────────────────────┬──┘
        public_payload()   │                      │  ToolCall
        (공개 영역만)      ▼                      ▼
                  ┌─────────────────┐   ┌──────────────────────┐
                  │  PII Gate       │   │  MotionController    │
                  │  탐지·마스킹·차단│   │  (Mock — 이번 범위)  │
                  └────────┬────────┘   └──────────┬───────────┘
                           ▼                       ▼
                  ┌─────────────────┐   ┌──────────────────────┐
                  │  LlmClient      │   │  Vision.verify       │
                  │  (선택 어댑터)  │   │  → VerificationResult│
                  └─────────────────┘   └──────────────────────┘
```

**설계의 핵심 두 가지**

1. **fields JSON 이 유일한 모듈 간 인터페이스다.** Vision 은 Agent 의 대화 로직을 모르고,
   Agent 는 Vision 의 탐지 알고리즘을 모른다.
2. **공개 정보 영역과 개인정보 영역을 구조적으로 분리한다.** 외부 LLM 에는
   `DocumentStructure.public_payload()` 의 결과만 나간다. 개인정보 항목은 구조 정보만 남고
   제목·약관·좌표가 제거된 상태로 강등된다.

---

## 3. 모듈 맵

| 경로 | 역할 | 상태 |
|---|---|---|
| `src/docagent/contracts.py` | 모든 모듈이 공유하는 데이터 계약. **외부 의존 0** | 확정 |
| `src/docagent/interfaces.py` | 모듈 경계 `Protocol` 8종. 구현 없음 | 확정 |
| `src/docagent/errors.py` | 도메인 예외 계층 9종 | 확정 |
| `src/docagent/config.py` | 조립 설정(`DocAgentConfig`) · 경로 · 어댑터 선택 | 구현 |
| `src/docagent/pipeline.py` | **단일 진입점** `build_session()` → `DocumentSession` | 구현 |
| `src/docagent/demo.py` | `python -m docagent.demo` 오프라인 통합 데모(Step 1~7) | 구현 |
| `src/docagent/kpi.py` | KPI 측정 `evaluate()` · 한국어 표 `render_table()` | 구현 |
| `src/docagent/vision/` | 정합 · 탐지 · OCR · 구조화 · 기입 검증 | 구현 |
| `src/docagent/pii/` | 개인정보 탐지 · 마스킹 · 유출 차단 게이트 · 감사 로그 | 구현 |
| `src/docagent/agent/` | 상태기 · 의도 분류 · 도구 · RAG · 가드레일 · 오케스트레이터 | 구현 |
| `src/docagent/io/speech.py` | `SpeechIO` 구현 — Console / Scripted / Google(지연 import) | 구현 |
| `src/docagent/io/motion.py` | `MotionController` 구현 — Mock / Serial(지연 import) | 구현 |
| `src/docagent/testing/` | 합성 신청서 생성기 · 결정론적 픽스처 | 구현 |
| `data/corpus/` | 근거 문서 8건(개인정보 보호법 · 서명 관행 요약) | 구현 |
| `docs/interface-spec.md` | 계약 규격 문서 (표 · JSON 예시 · 좌표계 규약) | 확정 |

### 파이프라인 한 줄 요약

```python
from docagent.pipeline import build_session

session = build_session(image)          # 정합 → 탐지 → OCR → 구조화 → PII → RAG → Agent
print(session.start().speech)           # "○○지원금 지급 신청서를 인식했습니다. …"
print(session.handle("쉽게 설명해줘").speech)
```

`build_session()` 이 하는 일(순서 그대로):

1. `normalize_document` — 촬영본을 A4 로 정립하고 mm 좌표계를 붙인다.
2. `build_detector().detect()` — 체크칸·기입선을 찾는다(규칙 기반, YOLO 는 선택).
3. `OcrEngine.read()` — 글자를 읽는다(미주입 시 빈 `StubOcr` + 경고).
4. `build_structure` — fields JSON 을 만든다.
5. `pii.policy.classify_field` — 민감도를 **fail-closed 재판정**한다(PUBLIC → PRIVATE 만).
6. `pii.policy.build_public_payload` + `LlmEgressGate.guard` — 좌표 없는 공개 payload.
7. `rag.build_index` — 근거 코퍼스 색인.
8. `Explainer` / `Guard` / `GatedLlmClient` 구성.
9. `ToolRegistry` — 좌표는 도구가 문서 구조에서 **직접** 읽는다(LLM 인자에 좌표 없음).
10. `DocumentAgent` 생성.

---

## 4. 범위

### 이번에 완성하는 것 — 핵심 3모듈

* **Vision** — 문서 이미지에서 기입란을 찾아 `fields JSON` 을 만든다. 기입 후 검증까지.
* **PII** — 외부로 나가는 모든 텍스트에서 개인정보를 탐지·마스킹하고, 위험하면 차단한다.
* **Agent** — 항목을 설명하고, 선택을 확인받고, 도구를 호출하고, 신뢰도가 낮으면 사람에게 넘긴다.

### 이번에 구현하지 않는 것 — Protocol + Mock(+ 미실행 어댑터 골격)까지만

* **음성 (STT / TTS)** — `ConsoleSpeechIO` · `ScriptedSpeechIO` 가 실제로 동작한다.
  `GoogleSpeechIO` 는 호출 코드까지 작성했으나 `google-cloud-speech` 가 설치되어 있지
  않아 **생성 시점에 `AdapterUnavailable`** 로 끝난다. 실제 음성 연동은 검증되지 않았다.
* **하드웨어 (Arduino 펜 액추에이터)** — `MockMotionController` 가 기본이다.
  `SerialMotionController` 는 `MOVE x y` / `HOME` 프로토콜과 재시도·타임아웃·범위 검사까지
  구현했고 가짜 포트로 테스트했지만, **실제 장치와 통신한 적은 없다**(`pyserial` 미설치).
* **OCR** — `pytesseract` 미설치. 데모·테스트는 합성 문서 레이아웃에서 만든
  `StubOcr` 로 대신한다(`docagent.demo.synthetic_ocr_words`).
* **YOLO 탐지기** — `ultralytics` 미설치. 규칙 기반 탐지기(`HeuristicDetector`)만 동작한다.
  학습 스크립트(`scripts/train_yolo.py`)와 데이터셋 변환기는 준비되어 있다.
* **Claude LLM** — `anthropic` 미설치. 기본값은 근거 문장만 골라 다듬는
  오프라인 구현(`OfflineTemplateLlm`)이다.
* **관리자 대시보드** — 미착수.
* **인적사항 표(성명·주민등록번호·주소·연락처) 탐지** — 규칙 기반 탐지기가 표 안쪽
  입력 칸을 찾지 못한다. 그래서 합성 신청서에서 파이프라인이 확정하는 항목은
  동의 · 신청일자 · 신청인 서명 **3개**다(정답은 8개). 자세한 내용은 아래 KPI 절 참조.

### 의도적으로 설치하지 않은 것

`ultralytics` · `torch` · `anthropic` · `openai` · `langchain` · `faiss` · `presidio` ·
`pydantic` · `pyserial` · `pytesseract` 는 **전부 선택적 어댑터**로만 다룬다.

* 모듈 최상단 import 금지. 함수·생성자 내부에서 지연 import 한다.
* `ImportError` 는 `AdapterUnavailable` 로 감싸 한국어 설치 안내를 담아 던진다.
* **API 키 없이 `pytest` 와 데모가 100% 통과해야 한다.** 네트워크 호출은 하지 않는다.

런타임 의존성은 `numpy`, `opencv-python-headless`, `Pillow` 세 개뿐이다.

---

## 5. 설치 · 실행

Windows 기준이며, **반드시 저장소 내부 venv 인터프리터**를 사용한다.

```powershell
# 저장소 루트
cd C:\Users\jkimz196\projects\doc-agent

# 의존성 확인 (numpy / opencv-python-headless / Pillow / pytest 는 이미 설치됨)
.venv\Scripts\python.exe -c "import numpy, cv2, PIL; print('ok')"
```

### 테스트

`pyproject.toml` 의 `[tool.pytest.ini_options]` 에 `pythonpath = ["src"]` 가 있어
**패키지를 설치하지 않아도** 저장소 루트에서 바로 돌아간다.

```powershell
# 전체 테스트
.venv\Scripts\python.exe -m pytest -q

# 계약 계층만
.venv\Scripts\python.exe -m pytest tests\test_contracts.py -q
```

Git Bash 를 쓴다면:

```bash
cd /c/Users/jkimz196/projects/doc-agent
/c/Users/jkimz196/projects/doc-agent/.venv/Scripts/python.exe -m pytest -q
```

### 통합 데모

```powershell
# 대화 로그 + KPI 표
$env:PYTHONPATH = "src"; .venv\Scripts\python.exe -m docagent.demo

# KPI 표만
$env:PYTHONPATH = "src"; .venv\Scripts\python.exe -m docagent.demo --kpi-only

# 기계 판독용 JSON
$env:PYTHONPATH = "src"; .venv\Scripts\python.exe -m docagent.demo --json
```

Git Bash:

```bash
cd /c/Users/jkimz196/projects/doc-agent
PYTHONPATH=src ./.venv/Scripts/python.exe -m docagent.demo
```

> **`PYTHONPATH=src` 가 필요한 이유** — venv 에 `setuptools` 가 없어 `pip install -e .`
> 를 실행할 수 없다. `pytest` 는 `pyproject.toml` 의 `pythonpath = ["src"]` 덕분에 그냥
> 돌지만, `python -m docagent.demo` 는 그 설정을 읽지 않으므로 경로를 직접 준다.
> `setuptools` 를 설치할 수 있는 환경이라면 `pip install -e .` 후 `PYTHONPATH` 없이
> 실행된다.

데모는 네트워크·API 키·하드웨어 없이 **항상 성공**하며, 필수 항목을 모두 마치고
개인정보 유출이 0 건이면 종료 코드 0, 아니면 1 을 돌려준다.

실행 흐름(로드맵 Step 1~7):

| Step | 발화 | 일어나는 일 |
|---|---|---|
| 1 | — | 정합 · 탐지 · 구조화 결과를 안내한다 |
| 2 | — | 첫 항목을 **원문 듣기 / 쉬운 설명 / 다음 항목** 과 함께 안내 |
| 3 | `쉽게 설명해줘` | 근거(법조항 출처) 포함 설명 + "원문을 대신하지 않습니다" 고지 |
| 3' | `원문 읽어줘` | 약관 원문을 가공 없이 낭독 |
| 4~5 | `동의할게` | 선택 확정 → 펜이 **'동의함' 네모 칸** 좌표로 이동 |
| 6 | `다 썼어요` | 기입 전/후 잉크 비율 비교로 체크 확인 → 다음 항목 |
| 7 | `확인` → `서명했어요` | 서명란 이동 → 서명 확인 → 완료 안내 |
| 안전 | `제가 이 지원금을 받을 수 있나요?` | 가드레일이 막고 **사람 지원**으로 넘긴다. 펜은 움직이지 않는다 |

마지막 안전 장면은 별도 세션으로 돌린다. 본 세션의 대화 기록에 섞이지 않으며
(`DemoResult.safety` 로 따로 노출), KPI 의 "직원 연결 정확도" 를 실제로 측정하기 위한
장면이다. 이 장면이 없으면 그 지표는 영원히 "측정 불가" 로 남는다.

### KPI 측정

```python
from docagent.demo import build_demo_form, run_demo
from docagent.kpi import KpiCase, evaluate, render_table

form = build_demo_form()
result = run_demo(quiet=True)
print(render_table(evaluate([KpiCase(result.session, form.truth)])))
```

측정 지표와 목표치는 `docagent.kpi.TARGETS` 에 있다. 정답 구조가 없는 세션에서는
Vision 지표를 추정하지 않고 **"측정 불가"** 로 보고한다.

현재 합성 신청서에서의 실측값은 모든 지표가 목표를 만족한다. 다만 **이 숫자를 읽을 때
반드시 알아야 할 두 가지**가 있다.

1. 재현율의 분모는 **파이프라인이 확정한 3개 항목**이 아니라 정답 구조의 체크칸·서명란이다.
   인적사항 표의 입력 칸 4개(`TEXT_INPUT`)는 애초에 재현율 지표의 대상이 아니어서
   100% 라는 값에 반영되지 않는다. 표 셀 탐지가 추가되면 `TEXT_INPUT` 재현율 지표를
   새로 넣고 목표치를 다시 잡아야 한다.
2. `handoff_probe=True` 로 표시한 세션(가드레일 검증 장면)은 완료율·턴 수 분모에서
   빠진다. 표 아래에 그 사실이 명시된다. 이 표시가 없으면 "가드레일을 더 많이 시험할수록
   완료율이 떨어지는" 지표가 되기 때문이다.

### 계약 사용 예

```python
from docagent.contracts import (
    BoxMm, DocumentStructure, Field, FieldRole, FieldType, Option, Sensitivity,
)

document = DocumentStructure(
    document_id="doc_0001",
    doc_title="예금계좌 개설 신청서",
    fields=(
        Field(
            id="consent_01",
            type=FieldType.CHECKBOX,
            title="개인정보 수집·이용 동의",
            role=FieldRole.APPLICANT,
            options=(
                Option("동의함", BoxMm(30.0, 120.0, 5.0, 5.0)),
                Option("동의하지 않음", BoxMm(62.0, 120.0, 5.0, 5.0)),
            ),
            required=True,
            sensitivity=Sensitivity.PUBLIC,
            box_mm=BoxMm(25.0, 112.0, 160.0, 18.0),
            clause_text="수집 항목: 성명, 연락처. 보유 기간: 1년.",
            order=1,
            confidence=0.93,
        ),
    ),
)

# fields JSON 왕복 (세션 복원)
assert DocumentStructure.from_json(document.to_json()) == document

# 액추에이터 목표 좌표 (mm)
target = document.field_by_id("consent_01").options[0].box_mm.center()
print(target.to_tuple())  # (32.5, 122.5)

# 외부 LLM 에 보낼 수 있는 공개 영역만
payload = document.public_payload()
```

---

## 6. 개발 규약

| 항목 | 규약 |
|---|---|
| 언어 | 주석 · 독스트링 · 로그 · 사용자 대상 문자열은 **한국어**. 식별자 · 파일명은 영어 `snake_case` |
| 좌표 | 도메인 좌표는 **A4 mm**, 원점 좌상단, y 는 아래쪽이 + . 픽셀은 Vision 내부 전용 |
| 타입 | 모든 공개 함수 · 메서드에 타입힌트. 독스트링에 입출력 규격 명시 |
| 테스트 | 담당 모듈마다 pytest 동봉. 작업 종료 전 직접 실행해 통과시킨다 |
| 결정성 | 결정론적으로 동작해야 한다. `random` 사용 시 **seed 고정**. 시각은 `Clock` 주입 |
| 예외 | 삼키지 않는다. 도메인 예외로 감싸 `raise ... from exc`. **조용한 실패 금지** |
| 경로 | `pathlib` 사용. POSIX 경로 하드코딩 금지 |
| 파일 I/O | 항상 `encoding="utf-8"` 명시 |
| 개인정보 | `PiiSpan` 은 원문 값을 담지 않는다. 스캔 이미지 · 세션 로그는 커밋하지 않는다(`.gitignore`) |

---

## 7. 통합 시 조정한 것

여섯 모듈은 서로의 코드를 보지 못한 채 병렬로 작성되었다. 연결하면서 실제로 드러난
불일치와 그 처리는 다음과 같다. **기능 재설계는 하지 않았고, 계약을 맞추는 최소 수정만
했다.**

| 파일 | 무엇을 | 왜 |
|---|---|---|
| `vision/structuring.py` | 서명란으로 탐지된 밑줄의 왼쪽 라벨이 날짜를 가리키면 `DATE` 로 되돌린다(`_is_date_line`) | 기하학적 탐지기는 "신청일자 ____" 와 "신청인 ____ (서명 또는 인)" 을 구분할 수 없다. 의미 판별은 구조화 단계 책임이라고 탐지기 스스로 문서화해 두었는데, 그 처리가 빠져 있었다. 그 결과 날짜선이 **주체 미상 서명란**으로 남아 서명란이 둘이 되고, 문서를 열 때마다 "어느 것이 신청인 서명란인지 확정 불가" 로 직원 연결이 발생했다 |
| `pii/gate.py` | 무인자 팩토리 `build_default_gate()` 추가 | `agent/llm.py` 의 `_resolve_default_gate()` 가 `build_default_gate` / `default_gate` / `DefaultPiiGate` / `PiiGateImpl` 순으로 찾도록 이미 작성되어 있었으나, `pii/gate.py` 에 그 이름이 없어 `build_llm(gate=None)` 이 항상 차단으로 끝났다 |
| `tests/agent/test_explainer.py` | `test_build_llm_without_gate_is_blocked` → 두 개의 테스트로 분리 | 위 팩토리가 생기면서 "게이트를 확보하지 못한다" 는 전제가 사실이 아니게 되었다. 단언을 약화시키지 않고 **더 강하게** 바꿨다: (1) 게이트를 주지 않으면 기본 게이트를 확보해 감싼다, (2) 팩토리 이름을 전부 제거하면 여전히 `PiiEgressBlocked` 로 차단한다 |

### 통합 시점에 확정한 정책

병렬 작업자들이 `open_issues` 로 남긴 미결 사항 중 통합이 결정해야 했던 것들이다.

* **LLM 래퍼 단일화** — 한때 `pii/gate.py` 와 `agent/llm.py` 에 이름이 같은
  `GatedLlmClient` 가 각각 있었고, 후자는 마스킹만 하고 감사 기록을 남기지 않아
  `build_llm()` 을 직접 쓰는 호출자에게서 안전 KPI(`pii_llm_transmissions`)가
  근거 없이 0 으로 보고됐다. 지금은 `pii/gate.py` 의 것 **하나뿐**이며
  (마스킹 + 마스킹 후 재검사 + 감사 기록), `build_llm()` 도 그것을 돌려준다.
  `sanitize()` 만 가진 게이트가 주입되면 `agent/llm._PiiGateEgressAdapter` 가
  나머지 규약(`prepare` / `audit`)을 채운다.
* **문서 단위 저신뢰의 처리** — 정합 신뢰도가 임계값에 미달해도 **항목별 신뢰도를
  깎지 않는다.** 항목별 신뢰도는 실측값이고, 문서 단위 계수를 곱하면 측정값의 의미가
  사라진다. 특히 여백 없는 스캔에서 발생하는 "문서 경계 미확인" 대체 경로는 좌표 오차가
  오히려 0.00mm 로 가장 정확하므로 직원 연결 사유로 쓰지 않고, 안내 문구로만 알린다.
  더 엄격한 배치가 필요하면 `build_session(..., propagate_document_confidence=True)`.
* **설명 신뢰도 미달 시 즉시 핸드오프** — 오케스트레이터 원안대로 유지했다. 원문 낭독
  폴백으로 계속 진행하는 안도 검토했으나, 이 동작을 단언하는 테스트가 이미 있어
  통합이 임의로 바꿀 사안이 아니라고 보았다. 대신 데모는 설명이 잘 되는 공개 영역
  항목(동의)에서 설명을 요청하고, 개인정보 영역 항목은 원문·안내로 진행한다.
* **`ExplanationResult` 계약 승격** — 하지 않았다. `tools.py` 가 `getattr` 덕타이핑으로
  접근하고 있고 실제 필드명이 일치함을 통합 테스트로 고정했다. 승격은 계약 파일 변경이라
  통합 범위를 넘는다.

---

## 8. 더 읽을 것

* [`docs/interface-spec.md`](docs/interface-spec.md) — 계약 전문. 좌표계 규약, 타입별 표,
  fields JSON 예시, 공개/개인정보 영역 분리 원칙, Protocol 목록, 예외 계층, 계약 변경 절차.
