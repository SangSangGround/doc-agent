# Roboflow RF-DETR 탐지기

Roboflow 에서 학습한 RF-DETR 모델을 서버리스 Workflow 로 호출해
체크박스(`check_box`)와 서명란(`signature_field`)을 탐지하는 **선택적 어댑터**다.

| 항목 | 값 |
|---|---|
| 모듈 | `src/docagent/detector/roboflow_detector.py` |
| 클래스 | `RoboflowDetector` — `docagent.interfaces.Detector` 프로토콜 구현 |
| 선택 의존 | `inference-sdk`, `python-dotenv` (`pip install "docagent[roboflow]"`) |
| 출력 | `detect()` → A4 mm `Detection` 목록 / `predict()` → 픽셀 `RawPrediction` 목록 |
| `Detection.source` | `"roboflow_rfdetr"` |

이번 범위는 **탐지까지만**이다. OCR · LLM · TTS 연결, `build_detector()` 팩토리 연동은 하지 않았다.

## 설치

```powershell
.venv\Scripts\python.exe -m pip install "inference-sdk>=1.7" "python-dotenv>=1.0"
```

## API 키

저장소 루트에 `.env` 를 만든다(`.env.example` 참고). `.env` 는 `.gitignore` 에 등록되어 있다.

```
ROBOFLOW_API_KEY=<팀에서 공유받은 키>
```

키는 GitHub 에 올리지 말고 비공개 채널로 주고받는다.

## CLI

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m docagent.detector.roboflow_detector <이미지 경로>
.venv\Scripts\python.exe -m docagent.detector.roboflow_detector <이미지 경로> --conf 0.3 --output outputs/result.jpg
.venv\Scripts\python.exe -m docagent.detector.roboflow_detector <이미지 경로> --no-draw
```

출력 예:

```
=== Detection Results ===
(bbox x, y = 박스 중심 좌표, 단위 px)

[1]
class      : check_box
confidence : 76.3%
bbox       : x=717.5, y=2007.5, width=67.0, height=79.0

Total objects: 5

결과 이미지 저장: outputs\result.jpg
```

기본 confidence 임계값은 0.5 이다. `outputs/` 는 커밋하지 않는다.

## 코드에서 쓰기

```python
import cv2
from docagent.detector.roboflow_detector import RoboflowDetector

detector = RoboflowDetector(conf=0.5)        # ROBOFLOW_API_KEY 환경변수 사용
detections = detector.detect(cv2.imread("page.png"))   # A4 mm Detection 목록
raw = detector.predict("page.png")           # 픽셀 좌표(중심 기준) RawPrediction 목록
```

## Workflow 응답 구조 (실제 호출로 확인)

```json
[
  {
    "predictions": {
      "image": {"width": 2067, "height": 2924},
      "predictions": [
        {"x": 717.5, "y": 2007.5, "width": 67.0, "height": 79.0,
         "confidence": 0.76, "class": "check_box", "class_id": 0}
      ]
    },
    "inference_id": "...",
    "model_id": "..."
  }
]
```

* `x`, `y` 는 상자 **중심** 픽셀 좌표다. `detect()` 는 좌상단 기준으로 바꿔 mm 로 환산한다.
* 탐지가 0건이면 `image.width/height` 가 `null` 이다(오류 아님).
* 클래스 이름은 `check_box` 다. `vision/dataset.py` 의 YOLO 클래스 이름(`checkbox`)과
  다르므로 `CLASS_NAME_TO_FIELD_TYPE` 에서 둘 다 `FieldType.CHECKBOX` 로 매핑한다.
  클래스 id 순서도 YOLO 매핑(`0=signature_field`)과 다르므로 **id 가 아니라 이름으로** 매핑한다.

## 알아둘 점

* Workflow `my-first-project-lkpc5` 의 기본 `model_id`(`...--9602f1`)는 실측 결과 탐지가
  0건이었다. 그래서 학습된 모델 `-0qtd3/my-first-project-lkpc5-3-rfdetr-nas-t1--f98b3b` 를
  workflow 파라미터로 넘긴다. 바꾸려면 `.env` 에 `ROBOFLOW_MODEL_ID=...` 를 둔다.
* `detect()` 의 mm 환산은 다른 탐지기와 같은 `coordinate_system_from_image()` 를 쓴다.
  즉 입력은 **A4 로 정립된 이미지**여야 한다.

## 오류 처리

| 상황 | 결과 |
|---|---|
| 이미지 파일 없음 (CLI) | `[ERROR] 이미지 파일이 없습니다` · 종료 코드 1 |
| API 키 없음 | `ValueError` (설정 안내 포함) |
| `inference-sdk` 미설치 | `AdapterUnavailable` (설치 안내 포함) |
| API 호출 실패(401 등) | `VisionError("Roboflow API 호출에 실패했습니다: ...")` |
| 예상하지 못한 응답 구조 | `VisionError` |
| 탐지 0건 | 빈 리스트 · CLI 는 `Total objects: 0` |

## 테스트

`tests/detector/test_roboflow_detector.py` — 실제 응답 구조를 본뜬 가짜 클라이언트를 주입하므로
**네트워크 · API 키 · inference-sdk 없이** 통과한다.

```powershell
.venv\Scripts\python.exe -m pytest tests\detector -q
```
