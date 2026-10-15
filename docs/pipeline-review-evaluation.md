# 통합 안내·검수·평가 실행 안내 (#6)

지원 문서가 정해지기 전의 공통 도구다. 실제 지원 문서 목록은 여전히 비어 있다.
명령은 저장소 루트에서 실행한다. 개인정보를 담은 입력·검토 산출물은 `outputs/`
또는 `data/private/`에서 관리하며 Git에 올리지 않는다.

## 1. 설치와 환경 점검

Python 3.10 이상을 사용한다. 가상환경 이름을 강제하지 않는다. 이번 로컬 검증은
기존 `venv/`에서 수행했으며 이전 `.venv/`는 수정하지 않았다.

```sh
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows PowerShell에서는 위 줄 대신 .\venv\Scripts\Activate.ps1
python -m pip install -e '.[dev,ocr]'
python -m docagent.doctor --profile core
python -m docagent.doctor --profile ocr
```

이미 만들어진 환경은 다시 만들 필요 없이 활성화한다. OCR에는 Python 패키지와
별도로 OS용 Tesseract 실행 파일 및 `kor`, `eng` 언어 데이터가 필요하다.
`doctor`는 설치 여부를 검사하며 자동 설치하지 않는다.

Roboflow를 사용할 경우에만 `python -m pip install -e '.[roboflow]'`로 추가 설치한다.
키는 `.env.example`을 참고해 로컬 `.env`에 둔다. 코드나 이슈에 키 값을 쓰지 않는다.

```sh
python -m docagent.doctor --profile roboflow
python -m docagent.doctor --profile tts
```

프로필은 `core`, `ocr`, `roboflow`, `tts`, `all`이다. 필수 구성요소가 모두 있으면
종료 코드 0, 부족하면 1이다. 선택 기능의 미설치는 `core` 성공을 막지 않는다.
키 값은 출력하지 않고 설정 여부만 표시한다. 인증 성공을 검사하거나 API를 호출하지 않는다.
TTS는 현재 macOS `say`와 설치된 `Yuna` 음성을 사용한다. Windows/Linux에서
안내 검색·검수·평가는 가능하지만 로컬 음성 재생 어댑터는 아직 없다.

## 2. API 없는 전체 예제

```sh
python -m docagent.testing.workflow --output-dir outputs/workflow-demo
python -m docagent.run --analysis outputs/workflow-demo/analysis.json --document-context outputs/workflow-demo/context.json --catalog outputs/workflow-demo/catalog.json --demo --interactive
python -m docagent.evaluate outputs/workflow-demo/manifest.json --output-dir outputs/workflow-evaluation --demo
```

`outputs/workflow-demo/review.html`을 브라우저로 열면 탐지·OCR 오버레이와
마스킹 편집 화면을 확인할 수 있다. 서버나 추가 프론트엔드 설치가 필요 없다.
`--interactive`에서 n=다음, p=이전, r=다시 듣기, s=정지, q=종료다.
실제 소리 재생은 `--speak`를 함께 지정한다. 텍스트 실행에는 음성 장치가 필요 없다.

예제의 승인자 `synthetic-fixture-only`, 높은 신뢰도, 검사 통과 상태는 **가상 테스트
입력**이다. 실제 사람 검토·실제 모델 성능을 뜻하지 않으며 실제 자료에 복사하지 않는다.
데모 목록은 `--demo`가 없으면 안내되지 않는다. 출력 폴더는 새 경로를 사용한다.

## 3. 이미지 마스킹 검수

```sh
python -m docagent.review mask form.png --dpi 300 --output outputs/review/form.html
```

HTML에서 할 수 있는 일:

- 드래그 또는 x/y/너비/높이 입력으로 영역 추가
- 목록에서 선택 후 좌표 수정·삭제·실행 취소
- 여백 조정과 불투명 검은색 마스킹 미리보기
- 미승인 `mask-plan.draft.json` 다운로드

다운로드 위치는 브라우저 설정을 따른다. 파일을 로컬 검토 폴더로 옮긴 후 사용한다.
이미 검토한 계획을 수정하려면 `--plan`으로 불러온다. 편집 화면에서 내보내는 계획은
항상 미승인이다. 기존 승인을 재사용하지 않는다.

사람이 모든 개인정보 영역을 확인한 후 다음 명령으로 검토자·UTC 시각·변경 감지값을 기록한다.

```sh
python -m docagent.review mask-approve outputs/review/mask-plan.draft.json --image form.png --dpi 300 --reviewer "검토자 이름" --confirm-reviewed --output outputs/review/mask-plan.approved.json
```

원본 및 DPI는 준비 단계와 같아야 한다. 다른 이미지·빈 영역·유효하지 않은 좌표는
승인할 수 없다. 승인 해시는 변경 감지용이며 신원 인증이나 마스킹 누락 검출을 대신하지 않는다.
검토 HTML에는 이미지와 OCR 원문이 포함될 수 있어 **공유용 보고서가 아니다**.

## 4. 분석 결과와 오버레이

```sh
python -m docagent.analyze form.png --detector roboflow --capture-mode photo --ocr-mode page --dpi 300 --mask-plan outputs/review/mask-plan.approved.json --include-ocr-text --output outputs/review/analysis.json
python -m docagent.review mask form.png --dpi 300 --analysis outputs/review/analysis.json --plan outputs/review/mask-plan.approved.json --output outputs/review/overlay.html
```

촬영본은 `photo`, 이미 페이지로 잘린 원본은 `scan`을 사용한다. OCR 원문 옵션은
로컬 안내 연결·검토용이다. 기본 분석 보고서는 OCR 원문을 제외한다.
오버레이는 정합 이미지 지문이 분석 결과와 같을 때만 표시한다. OCR 글자가 문서의
HTML/스크립트로 실행되지 않도록 처리하며 외부 요청은 허용하지 않는다.

`RoboflowDetector.predict()` / `detect()` 직접 호출도 이제 기본 차단된다.
검토된 배열에는 `mask_plan=MaskPlan(...)`을 전달한다. 개인정보 없는 빈 양식을
직접 확인한 경우에만 `allow_unmasked_blank=True`를 사용한다. 기존 `build_session()`에
날것의 Roboflow 탐지기를 주는 경로도 이 제한을 받는다.
코드에서 연결할 때는 `MaskedDetector(detector, plan)` 또는 명시적인
`BlankDocumentDetector(detector)`를 사용한다. 빈 양식 예외는 자동 안전 검사가 아니다.
탐지기 단독 CLI는 `--allow-unmasked-blank`를 요구하며, 마스킹 실행은 분석 CLI를 사용한다.

## 5. 문서 정보·순서·안내문 승인

```sh
python -m docagent.review context-draft outputs/review/analysis.json --output outputs/review/context.draft.json
```

초안의 `document_id`, `document_version`, `page`, `confidence`, `identity_verified`,
`field_order`를 실제 근거에 따라 검토한다. `field_order`는 분석 결과의 항목 ID이며
체크박스/서명 대상 전체를 중복 없이 포함해야 한다. 분석 파일명에서 문서 종류를 추정하지 않는다.

```sh
python -m docagent.review context-approve outputs/review/context.draft.json --analysis outputs/review/analysis.json --reviewer "검토자 이름" --confirm-reviewed --output outputs/review/context.approved.json
python -m docagent.review catalog-check data/private/catalog.json
```

문서 정보 승인은 분석 결과의 `needs_review`, 신뢰도, 촬영 검사 결과를 바꾸지 않는다.
그 경고가 남아 있으면 승인된 context가 있어도 안내를 거절한다.
분석 JSON 전체 해시와 순서가 승인에 묶이므로 OCR 결과·박스·순서를 바꾸면 재검토가 필요하다.

안내 목록 형식은 [기존 안내 계약](guidance-foundation.md)을 따른다.
문구를 수정할 때는 대상 항목을 초안으로 바꾸고 수정한 뒤 사람이 재검토한다.

```sh
python -m docagent.review catalog-draft data/private/catalog.json --field-id target-field --output outputs/review/catalog.draft.json
# catalog.draft.json의 문구·키워드를 수정한 후 실제 검토
python -m docagent.review catalog-approve outputs/review/catalog.draft.json --field-id target-field --reviewer "검토자 이름" --confirm-reviewed --output outputs/review/catalog.approved.json
```

같은 field_id가 여러 문서에 있으면 문서별 파일로 나누어 지정한다. 승인 명령은
대상을 자동 추정하지 않는다. 검수·승인 명령은 기존 출력 파일을 덮어쓰지 않는다.

## 6. 통합 안내 실행

```sh
python -m docagent.run --analysis outputs/review/analysis.json --document-context outputs/review/context.approved.json --catalog outputs/review/catalog.approved.json --interactive --speak
```

저장된 분석을 사용하면 외부 추론을 반복하지 않는다. 모든 대상 항목이 승인 문구와
하나씩 대응해야 세션을 시작한다. 여러 분석 항목이 같은 안내 항목에 매칭되어도 차단한다.
문구는 변경 없이 반환·재생한다. 마지막/첫 항목에서 이동하면 현재 항목을 유지한다.
Ctrl+C·EOF·종료 시 해당 세션의 음성 프로세스를 정리한다.

이미지부터 다시 실행하는 경로도 제공한다.

```sh
python -m docagent.run --image form.png --detector roboflow --capture-mode photo --ocr-mode page --dpi 300 --mask-plan outputs/review/mask-plan.approved.json --document-context outputs/review/context.approved.json --catalog outputs/review/catalog.approved.json
```

이미지 경로는 먼저 문서 확인 기록·지원 여부·검토한 원본 픽셀 지문을 검사하고
품질→전송 검토→OCR 준비→탐지/OCR→
안내 순으로 실행한다. 재분석 결과가 기존 승인된 분석과 달라지면 안내를 중단한다.
SDK/모델 업데이트 등으로 결과가 달라지면 다시 분석·검토해야 한다.
새 문서를 처음 받는 경우는 앞 절의 분석 및 사람 검토를 먼저 진행한다.
결과에는 `status`, 단계별 `events`, 실패 `reason`을 담는다.
종료 코드: 준비 완료 0, 검토 필요/차단 2, 입력 파일·음성 실행 오류 1, 사용자 중단 130.
기존 `docagent.demo`는 생성형 설명을 포함한 개발 데모이며 승인 안내 실행 경로가 아니다.

## 7. 평가 입력 형식

평가기는 저장된 예측을 읽으며 외부 추론이나 학습을 수행하지 않는다.
상대 경로는 manifest 파일이 있는 디렉터리를 기준으로 해석한다.

```json
{
  "schema_version": 1,
  "dataset_version": "external-v1",
  "model_version": "detector-v2",
  "prediction_version": "run-001",
  "samples": [{
    "id": "sample-001",
    "image": "original.png",
    "image_size": [600, 850],
    "variant": "original",
    "split": "external",
    "form_id": "form-version-1",
    "source_id": "source-page-1",
    "training_membership": "unknown",
    "labels": {
      "format": "coco",
      "path": "labels.coco.json",
      "image_id": 1
    },
    "predictions": "analysis.json"
  }]
}
```

- `variant`: original/photo. 같은 원본을 출력·촬영한 자료는 같은 `source_id`를 쓴다.
- `form_id`: 동일 양식·버전의 그룹. 페이지가 달라도 같은 양식이면 공유한다.
- `split`: train/val/test/external/unknown.
- `training_membership`: included/excluded/unknown. 근거가 없으면 unknown을 유지한다.
- `labels: null`은 정답 없음이다. 정답 파일에 객체가 0개인 경우는 배경 정답으로 평가한다.
- `predictions`가 없으면 예측 미실행으로 제외하며, 예측 파일의 빈 목록은 미탐으로 평가한다.
- YOLO는 `{"format":"yolo","path":"labels.txt","class_names":["check_box","signature_field"]}`처럼
  **내보낸 데이터의 실제 클래스 순서**를 명시한다. 프로젝트 내부 ID 순서와 다를 수 있다.
- COCO는 클래스 이름으로 매핑하고 실수 좌표를 보존한다. crowd/ignore는 지원하지 않으며 거절한다.

예측 입력은 두 종류다.

1. 최신 분석 CLI의 `normalized_a4_mm` JSON. 원본 지문, 정합 크기, 호모그래피가 필수다.
   정합 mm 좌표를 원본 픽셀로 역변환한 뒤 네 꼭짓점의 축 정렬 박스로 비교한다.
   원근/회전이 있으면 축 정렬 박스가 실제 사각형보다 커질 수 있다.
2. 원본 이미지 좌표를 사용하는 아래 형식. 이미지 지문은
   `docagent.pii.image_mask.image_digest(cv2.imread(...))`로 계산한다.

```json
{
  "coordinate_system": "image_px",
  "image_size": [600, 850],
  "image_sha256": "원본 픽셀 지문",
  "predictions": [{"class":"checkbox","bbox":[10,20,15,15],"confidence":0.95}]
}
```

파일명이 같아도 이미지 지문이 다르면 거절한다. 예전 보고서에 변환 정보가 없으면
임의로 좌표를 맞추지 말고 분석 결과를 다시 만들어야 한다.

안내 평가는 manifest 루트에 `catalog` 경로, 각 샘플에 `guidance_cases`를 둔다.
각 사례는 `observation`과 `expected`를 갖는다. expected는
`{document_id, document_version, page, field_id}`, 올바른 거절 정답은 `null`이다.
`expected` 자체를 생략하면 정답 없는 사례이며 정확도 계산에서 제외한다.
예제 생성기가 실제 형식의 JSON을 제공한다.

마스킹 평가는 샘플의 `mask: {truth: [[x,y,w,h]], predicted: [[x,y,w,h]]}`로
원본 이미지 좌표의 **실제로 적용한 여백까지 포함한 사각형**을 제공한다.
이미지별 원본/정합 좌표를 혼합하지 않는다. 두 목록 중 하나가 없으면 측정 불가다.
중복 영역은 합집합으로 계산하며 자동 PII 탐지나 실제 검토를 대신하지 않는다.

## 8. 보고서 및 비교

```sh
python -m docagent.evaluate data/private/eval/manifest.json --output-dir outputs/eval/run-001 --iou 0.5 --confidence 0.5
python -m docagent.evaluate data/private/eval/manifest.json --output-dir outputs/eval/run-002 --baseline outputs/eval/run-001/evaluation.json
```

`evaluation.json`, `evaluation.md`를 생성한다. 탐지는 신뢰도 순 일대일 매칭이며
동일 정답에 대한 중복 예측은 FP다. 클래스별 TP/FP/FN·Precision·Recall·F1·매칭 IoU를
계산한다. 분모가 0인 지표는 null/측정 불가다. mAP 계산기는 아니다.
안내와 마스킹 지표는 탐지 지표와 구분하며 각 분모를 Markdown에 설명한다.

원본/촬영본 비교는 두 묶음의 집계다. 1:1 짝별 성능 개선 검정은 아니다.
JSON에는 분할별 결과도 포함한다. 전체 집계에는 manifest의 모든 자료가 포함되므로
최종 외부 성능은 external/test 분할 및 중복 경고를 함께 확인해야 한다.
동일 픽셀·양식·원본 그룹의 분할 간 중복, 학습 포함 여부 미상을 표시한다.
이 검사는 주어진 메타데이터를 확인할 뿐 알려지지 않은 학습 이력을 알아내지 않는다.

버전·임계값·패키지 버전·데이터/예측/안내 목록 지문을 기록한다.
기준 보고서와 자료·임계값·안내 목록이 같을 때만 탐지 지표 증감값을 제공한다.
이미지, OCR, 원본 파일명, 양식 그룹명은 공유 보고서에 복사하지 않고 샘플 별칭을 쓴다.
버전 이름에도 개인정보나 키를 넣지 않는다. 정답 자료가 없으면 수치로 성능을 주장하지 않는다.

## 9. 검증 및 후속 작업

```sh
# macOS/Linux에서 실제 소켓 연결까지 차단하는 검증
DOCAGENT_TEST_OFFLINE=1 python -m pytest -q
# Windows PowerShell: $env:DOCAGENT_TEST_OFFLINE="1"; python -m pytest -q
python -m compileall -q src
```

GitHub Actions는 Linux Python 3.10/3.12, Windows/macOS Python 3.12에서 핵심·테스트
의존성을 설치해 테스트·합성 예제를 실행한다. Ubuntu에는 `fonts-nanum`도 설치하고,
모든 OS에서 한글 폰트 존재 여부를 확인한다. 폰트가 없으면 사각형 대체 글리프가
체크박스로 오탐될 수 있으므로 탐지 정확도 테스트에는 한글 폰트가 필요하다.
로컬 Linux에서도 `sudo apt-get install fonts-nanum`으로 설치할 수 있다.
실제 API 키·OCR 엔진·스피커는 필요하지 않다. 패키지·폰트 설치에는 네트워크가
필요하지만 테스트는 가짜 어댑터를 사용한다. 한글 JSON·보고서는 OS의 기본
인코딩에 의존하지 않고 UTF-8로 읽는다. CI는 `PYTHONIOENCODING=utf-8`을 지정해
Windows의 리다이렉트된 표준 출력에서도 한글 안내문을 기록할 수 있게 한다.
Actions 버전은 [checkout](https://github.com/actions/checkout) 및
[setup-python](https://github.com/actions/setup-python)의 공식 사용 예를 따른다.
PR #7의 첫 원격 실행에서 macOS는 통과했고, Ubuntu는 한글 폰트 부재로 인한
합성 문서 오탐, Windows는 테스트 파일 읽기의 인코딩 누락으로 실패했다.
폰트 설치·사전 확인 및 UTF-8 지정을 보완했으며, 수정 후 원격 결과는 푸시한 뒤 확인한다.
수정 후 로컬 검증에서는 기본 macOS 환경과 나눔고딕·CP1252 기본 파일 읽기를
적용한 재현 조건 모두 1,456개 테스트가 통과했다. 이는 다른 OS에서 직접 실행한
결과가 아니므로 원격 CI 재실행으로 최종 확인해야 한다.

실제 지원 문서 선정, 문서별 식별 근거와 안내문 승인, 개인정보 정답 영역,
학습 이력 확인, 실제 촬영 장치에서의 임계값 보정과 접근성 검증은 이후 사람과 함께 진행한다.
문서별 자동 정렬/마스킹 템플릿, 모델 재학습, RAG, 하드웨어 제어는 이번 범위에 포함하지 않는다.

2026-10-15 로컬 검증: Python 3.12 `venv/`에서 네트워크 연결을 차단한 전체 테스트
1,456개 통과, Python 컴파일·diff 검사 통과. 합성 자료의 안내 실행 및 JSON/Markdown
보고서 생성, 브라우저 좌표 추가·수정·삭제·되돌리기·마스킹 미리보기와 초안 다운로드를
확인했다. wheel 빌드에서 검수 HTML 포함 및 비밀 파일/산출물 미포함도 확인했다.
현재 로컬 Tesseract kor/eng와 macOS Yuna는 사용 가능하며, 이 `venv/`에는
Roboflow SDK가 설치되지 않았다. 이번 작업에서 실제 외부 추론은 수행하지 않았다.
