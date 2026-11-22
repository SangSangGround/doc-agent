# doc-agent

시각장애인의 종이문서 작성 안내를 위한 탐지·OCR·검증된 안내문 검색 프로젝트다.
현재 실제 지원 문서는 선정 중이다. 사람이 검토한 원문 전체만 그대로 반환하며,
문서·작성 영역·신뢰도·승인 상태를 확인할 수 없으면 안내를 중단한다.

## 현재 구현 상태

| 기능 | 상태 |
|---|---|
| Roboflow RF-DETR 탐지 | 선택 어댑터 구현, 검토된 마스킹 또는 명시적 빈 양식 확인 필요 |
| 로컬 Tesseract OCR | 전체 페이지/탐지 주변 영역 OCR 연결 |
| 촬영 품질 검사 | 해상도·흐림·밝기·경계·기울기 휴리스틱 |
| 승인 원문 검색·낭독 | 항목/페이지/문서 전체 원문 보존, 모호하거나 미승인인 경우 차단 |
| 통합 안내 실행 | 검토 기록에 묶인 분석 결과, 항목 순서, 다음/이전 이동 |
| 로컬 TTS | macOS `say`, 재생·정지·반복. Windows/Linux 음성 어댑터는 미구현 |
| 이미지 마스킹 | 사람이 검토한 이미지별 박스 적용. 자동 개인정보 영역 탐지는 아님 |
| 로컬 검수 화면 | 탐지/OCR 오버레이, 마스킹 박스 편집·미리보기·초안 저장 |
| iPhone 앱 | 방향·비프음 → 수동 초점·촬영 → Vision OCR·원문 듣기. 기기 내 한국어 질문 받아쓰기·확인 후 Wi-Fi로 GPT-5.6 Luna에 질문 |
| 평가 도구 | COCO/YOLO 정답 연결, 탐지·안내·마스킹 평가, JSON/Markdown 보고서 |
| 정답 박스 검수 | 원본 위 박스 편집, 미승인 초안 저장, 사람 검수 후 COCO 연결 |
| 환경 점검·CI | 설치/언어/음성/키 설정 여부 점검, 외부 API 없는 회귀 테스트 구성 |
| 실제 문서 검증 | 지원 문서·안내문·정답 데이터 확보 후 진행 |

실제 OCR·Roboflow 연결과 모델 재학습은 다른 작업이다. 이 저장소의 최근 작업은
기존 탐지 모델과 OCR을 연결하고 안내·검수·평가 도구를 구현한 것이다.

## 설치

Python 3.10 이상. 기존 가상환경이 있으면 활성화해서 사용한다.

```sh
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows PowerShell: .\venv\Scripts\Activate.ps1
python -m pip install -e '.[dev,ocr]'
python -m docagent.doctor --profile core
python -m pytest -q
```

실제 OCR에는 OS용 Tesseract와 한국어/영어 언어 데이터가 별도로 필요하다.
Roboflow 사용 시 `python -m pip install -e '.[roboflow]'`로 선택 의존성을 설치하고
`.env.example`을 참고해 로컬 `.env`에 키를 설정한다. API 키는 커밋하지 않는다.
`doctor --profile ocr|roboflow|tts|all`로 필요한 기능을 점검한다.

## API 없는 빠른 실행

```sh
python -m docagent.testing.workflow --output-dir outputs/workflow-demo
python -m docagent.run --analysis outputs/workflow-demo/analysis.json --document-context outputs/workflow-demo/context.json --catalog outputs/workflow-demo/catalog.json --demo --interactive
python -m docagent.evaluate outputs/workflow-demo/manifest.json --output-dir outputs/workflow-evaluation --demo
```

`outputs/workflow-demo/review.html`을 브라우저로 열면 마스킹 검수 화면이 나온다.
가상 승인 기록은 테스트 전용이다. 실제 문서용 승인으로 재사용하지 않는다.

## 실제 문서 작업 흐름

### 노트북 생활 문서 OCR·음성 질의응답

등록 양식 없이 안내문·가정통신문 등 촬영한 문서의 글자를 읽는다. macOS는 Apple Vision으로 문서 경계와 글자를 인식하고 다른 OS는 OpenCV·Tesseract를 사용한다. 문서가 안정적으로 들어오면 사용자가 초점을 맞추고 직접 촬영하며, 성공 후 원문 듣기·문서 질문·체크 상태 듣기를 제공한다. 비프음은 문서 위치를 안내하고 음성 안내 중에는 작아진다. 질문에는 로컬 Ollama `qwen3.5:4b`, 마이크에는 로컬 Whisper small을 사용한다.

```sh
PYTHONPATH=src ./venv/bin/python -m docagent.home serve
```

[로컬 앱](http://127.0.0.1:8765) · [설치·사용·검증 범위](docs/home-document-assistant.md). 일반 문서 모드는 `--template`을 붙이지 않는다. 모델 없이 OCR부터 확인하려면 `--llm off --no-speech`를 추가한다. 인식문과 체크 상태에는 오류가 있을 수 있으며, 불확실한 글자는 함께 안내한다. 기존 등록 양식 모드는 명시적으로 `--template ... --demo`를 지정할 때 사용한다.

### iPhone 문서 촬영과 Wi-Fi 질문

iPhone 13 이상·iOS 17 이상에서 후면 카메라로 문서를 찾고 직접 촬영한다. 네 모서리가 들어오면 긴 비프음이 난다. iOS 17.2부터 **볼륨 위로 초점 → 볼륨 아래로 촬영**할 수 있다. 읽기용 사진은 iPhone에서 Apple Vision이 해상도와 언어 보정 설정을 달리해 두 번 읽는다. 촬영과 원문 낭독은 오프라인에서도 가능하다.

사진을 읽으면 같은 Wi-Fi의 Mac 중계 서버가 OCR 글자를 OpenAI에 보내 불확실한 줄의 문맥을 검토한다. **문서에 질문하기**에서 말하면 iPhone의 Apple Speech가 한국어를 받아쓴다. 사용자는 질문을 듣고, 고치거나 다시 말한 뒤 확인해 보낸다. 한국어 기기 내 받아쓰기를 사용할 수 없으면 직접 입력할 수 있다. GPT-5.6 Luna는 확인한 질문과 OCR 원문, 최근 문답, 등록된 사례·용어를 보고 답한다. 직업훈련 생계비 대부 신청용 개인정보 동의서의 제공 기관을 물은 뒤 **출처 듣기**로 해당 줄을 확인할 수 있다. [아이폰 설치·중계 서버 설정·시연 안내](ios/README.md)

읽기 결과의 **체크·서명 안내**에서는 체크 상태를 듣고 선택한 칸까지 검지를 안내받는다. 시연 양식은 기기에 등록된 인쇄 문구와 작성 위치를 확인한다. 일반 문서에서는 사용자가 동의한 사진 한 장으로 Gemini·OpenCV가 작성란을 분석한다. 손끝과 현재 문서 위치는 후면 카메라로 기기 안에서 추적한다. [발표자용 검증 정리](docs/ios-writing-demo-handoff.md)

### 기존 분석·검수 경로

1. `docagent.review mask`: 로컬 검수 화면에서 개인정보 영역 지정
2. `docagent.review mask-approve`: 사람이 영역 확인 후 승인 기록 저장
3. `docagent.analyze`: 촬영 품질 검사·검토된 이미지 전송·로컬 OCR 분석
4. `docagent.review`: 분석 결과, 문서 식별 정보, 안내 순서, 안내문 검토
5. `docagent.run`: 모든 확인 조건을 만족한 경우 승인된 문구만 안내
6. `docagent.evaluate`: 저장된 정답과 예측을 비교하여 보고서 생성

기존 검토 경고는 승인 도구가 자동으로 지우지 않는다. 실제 문서 식별과 문맥 검증이
완료되기 전에는 분석 결과가 있어도 안내가 차단될 수 있다.

[통합 실행·검수·평가 안내](docs/pipeline-review-evaluation.md)에 전체 명령,
데이터 형식, 팀원 설치 방법 및 후속 검토 항목을 정리했다.

## 개발용 기존 파이프라인

`pipeline.build_session()`과 `python -m docagent.demo`는 기존
See → Understand → Explain → Ask → Act → Verify 개발 데모다.
이 경로에는 생성형 설명/질의응답 및 Mock 펜 제어가 포함되며,
실제 사용자에게 승인 안내를 제공하는 경로는 `python -m docagent.run`이다.
실제 하드웨어와 클라우드 STT는 검증하지 않았다.

주요 모듈: `contracts.py`(공유 데이터), `interfaces.py`(Protocol), `pipeline.py`(분석·기존 데모),
`guidance/`(승인 검색·검토 세션), `pii/`(개인정보 보호), `vision/`(영상·OCR),
`evaluation/`(오프라인 평가), `io/`(음성·모션 어댑터).

## 개발 규약

- 주석·문서·로그·사용자 문자열은 한국어, 식별자와 파일명은 영어를 사용한다.
- 공개 함수에 타입 힌트를 작성하고 오류를 조용히 무시하지 않는다.
- 도메인 좌표는 A4 mm(좌상단 원점), 이미지 내부 및 평가 좌표는 명시된 픽셀 좌표다.
- 픽셀과 mm를 섞지 않는다. 촬영본은 정합 변환 정보와 이미지 지문을 함께 검증한다.
- 선택적 SDK는 지연 import하며, API 키·네트워크·하드웨어 없이 테스트할 수 있어야 한다.
- 합성 테스트는 결정론적으로 작성하고 테스트 통과를 실제 문서 성능으로 해석하지 않는다.
- 파일은 `pathlib`과 UTF-8을 사용한다. 실제 문서 이미지·OCR 원문·세션 로그는 커밋하지 않는다.
- 안내문은 사람이 검토하고 승인한다. 요약·임의 생성으로 대체하지 않는다.

## 관련 문서

- [계약 및 인터페이스 규격](docs/interface-spec.md)
- [Roboflow 탐지기](docs/roboflow-detector.md)
- [탐지·OCR 연동](docs/ocr-integration.md)
- [승인 안내·마스킹·촬영 검사 계약](docs/guidance-foundation.md)
- [통합 실행·검수·평가](docs/pipeline-review-evaluation.md)
- [실제 문서 후보 3종·안내 초안·로컬 검수 결과](docs/candidate-documents.md)
- [전입신고서 후보·학습 중복 확인·실제 RF-DETR 추론](docs/community-form-audit.md)
- [OCR 비교·원문 전체 낭독·정답 박스 검수](docs/performance-and-verbatim.md)
