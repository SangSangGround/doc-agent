# 작성 영역 탐지와 주변 텍스트 OCR (Step 2)

`python -m docagent.analyze`는 빈 A4 양식을 정합하고, 작성 영역을 탐지한 뒤
로컬 Tesseract로 주변 문구를 읽어 기존 `DocumentStructure`로 연결한다.
LLM 호출, 음성 안내 및 펜 이동은 실행하지 않는다. 기존 오프라인 데모
`python -m docagent.demo`의 실행 방법은 그대로다.

## 설치

Python 3.10 이상 가상환경에서 저장소 루트를 작업 디렉터리로 사용한다.

```sh
python -m pip install -e '.[roboflow,ocr,dev]'
```

`pytesseract`는 Python 래퍼이므로 Tesseract 실행 파일과 한국어 데이터도 필요하다.
macOS(Homebrew):

```sh
brew install tesseract tesseract-lang
tesseract --list-langs
```

목록에 `kor`, `eng`가 있어야 한다. Windows는 [Tesseract 공식 설치 안내](https://tesseract-ocr.github.io/tessdoc/Installation.html)의
설치 경로를 참고하고, 한국어 데이터를 설치한 뒤 실행 파일을 PATH에 추가한다.
`.venv/Scripts/python.exe`가 있는 Windows 환경을 Mac에 복사했다면 재사용하지 말고
Mac에서 새 가상환경을 만든다.

## 환경변수

저장소 루트의 `.env.example`을 `.env`로 복사하고 다음 값을 채운다.
기존 `.env`가 있으면 덮어쓰지 않고 필요한 설정만 추가한다.

```dotenv
ROBOFLOW_API_KEY=팀에서_공유받은_키
# 모델을 변경할 때만 지정
# ROBOFLOW_MODEL_ID=학습된_모델_ID
```

`analyze` 실행 시 현재 디렉터리 또는 저장소 루트의 `.env`를 읽는다.
이미 설정된 환경변수가 파일 값보다 우선한다. 키는 결과 JSON에 포함하지 않는다.
`.env`, 원본 문서와 결과 파일은 커밋하지 않는다.

## 실행

개인정보가 없는 **빈 양식**의 PNG/JPEG를 사용한다. Roboflow를 선택하면 정합된
전체 이미지가 외부 서버에 전송된다. 이후의 PII 텍스트 게이트가 이 이미지
전송을 보호하지 않으므로 작성된 문서에는 사용하지 않는다.

```sh
python -m docagent.analyze /path/to/blank-form.png --detector roboflow --output outputs/analysis.json
```

기본 OCR 모드는 `regions`다. 문서 제목과 약관 전체가 필요하면 `page`로 실행한다.

```sh
python -m docagent.analyze /path/to/blank-form.png --detector roboflow --ocr-mode page --output outputs/page-analysis.json
```

API 없이 로컬 탐지와 OCR만 확인할 수도 있다.

```sh
python -m docagent.analyze /path/to/blank-form.png --detector heuristic --output outputs/local-analysis.json
```

- `--conf 0.3`: 탐지 결과에 남길 최소 신뢰도. 미지정 시 Roboflow 0.5, YOLO 0.25.
- `--lang kor+eng`: OCR 언어.
- `--dpi 300`: 정합 해상도.
- `--output` 생략: 결과 JSON을 터미널에 출력.
- 탐지기 미지정: `DOCAGENT_DETECTOR_PREFER` 또는 기존 `auto` 선택 규칙 사용.

설정 객체에서도 `detector_prefer="roboflow"`, `ocr_kind="tesseract"`,
`ocr_mode="regions"`, `ocr_lang="kor+eng"`을 사용할 수 있다.
각각 `DOCAGENT_DETECTOR_PREFER`, `DOCAGENT_OCR_KIND`, `DOCAGENT_OCR_MODE`,
`DOCAGENT_OCR_LANG` 환경변수와 대응한다. 일반 `build_session()`의 기본값은
여전히 오프라인 탐지 + 빈 StubOcr다. 설정 객체는 `.env`를 자동으로 읽지 않는다.

```python
import cv2
from dotenv import load_dotenv
from docagent.config import DocAgentConfig
from docagent.pipeline import analyze_document

load_dotenv()
config = DocAgentConfig(
    detector_prefer="roboflow", ocr_kind="tesseract", ocr_mode="regions",
)
analysis = analyze_document(cv2.imread("blank-form.png"), config=config)
print(analysis.structure.to_dict())
```

`analyze_document()`는 로컬 분석만 수행한다. `build_session()`은 그 결과로
LLM 전송용 payload·RAG·대화 에이전트를 추가로 만들며 기존 개인정보 게이트를
계속 적용한다. 로컬 분석 JSON은 외부 전송 승인을 받은 데이터가 아니다.

## 좌표와 매칭

1. 문서 정합 후 탐지한다. 탐지 박스는 A4 mm 좌표다.
2. 탐지 박스에 왼쪽 45 / 위 15 / 오른쪽 45 / 아래 8 mm 여백을 더한다.
3. 페이지 경계를 넘지 않게 자르고, 겹치는 Crop은 병합해 문구의 중복 인식을 줄인다.
4. 각 Crop의 실제 물리 크기로 OCR 좌표를 환산하고 원래 페이지 오프셋을 더한다.
5. 같은 행의 체크박스와 문구 배치를 비교해 좌/우 라벨을 연결한다.
   양쪽 매칭이 비슷하거나 문구가 누락되면 임의로 채우지 않고 확인 경고를 남긴다.
6. 서명란은 왼쪽 라벨을 먼저 확인하고, 없으면 8 mm 이내 위쪽 라벨을 확인한다.
   기입 주체를 확정하지 못하면 기존 UNKNOWN 및 재확인 규칙을 적용한다.

JSON은 `detections`와 `ocr_words`의 신뢰도를 따로 보존한다.
`fields`는 기존 구조화 결과이며, 해석 신뢰도는 매칭에 사용한 OCR 신뢰도보다
높아지지 않는다. `fields[].needs_review`, 문서의 `needs_review`, `warnings`,
`stages`로 확인이 필요한 결과를 구분한다. 신뢰도는 보정된 정답 확률이 아니다.

`regions` 모드는 제목·약관·문서 상단 필수 안내를 생략할 수 있어 문서 단위
`needs_review`가 항상 true다. 탐지되지 않은 서명란의 존재를 알아내거나 문서의
모든 필수 항목을 확인하는 기능은 이번 단계에 포함되지 않는다.

## 실패 처리

- 로컬 OCR 설치·언어 검사를 API 요청 전에 수행한다(`analyze` CLI).
- Roboflow 실패 시 로컬 탐지기로 조용히 바꾸지 않는다.
- `regions` 모드에서 탐지가 0건이면 빈 문서 결과를 성공으로 출력하지 않고 실패한다.
- 탐지는 있으나 OCR 문구가 없으면 결과에 확인 경고를 남긴다.
- OCR 실행 오류는 빈 결과로 숨기지 않고 오류로 반환한다.
- CLI 성공은 종료 코드 0, 실행 실패는 1이다. 성공이어도 `needs_review`를 확인한다.

## 검증

```sh
python -m pytest tests/vision/test_region_ocr.py tests/test_analyze.py -q
python -m pytest -q
```

자동 테스트는 가짜 Workflow 응답과 OCR 엔진으로 경계 Crop, 겹침 병합,
픽셀 반올림 후 좌표 복원, 좌/우 문구 매칭, 저신뢰 및 실패 처리를 검증한다.
외부 통신이나 API 키가 필요 없다. 실제 Roboflow 정확도와 한국어 OCR 품질은
별도의 빈 문서 샘플로 검증해야 한다.

개발 시 로컬 검증에서는 개인정보를 채우지 않은 합성 신청서에 실제 Tesseract
한국어 OCR을 적용해 체크박스 2개, 날짜란 1개, 신청인·대리인 서명란 2개를
구조화했다. `신 청 일 자`처럼 OCR이 음절 사이에 넣은 공백은 분류 시에만
무시하고 결과 원문에는 보존한다. 이 합성 검증만으로 모델 일반화 성능을 판단하지 않는다.


### 실제 문서 검증 (2026-10-06)

공유된 빈 양식 219파일에서 동일 파일 46개를 해시로 제외하고, PDF를 페이지별로
렌더링해 299장을 실제 Roboflow + 로컬 Tesseract로 검증했다. 원본 162페이지,
촬영본 137장이며, 촬영본은 원본 85페이지에 대응시켰다. 촬영본을 새로운 양식이나
독립 외부 테스트로 집계하지 않았다. 학습 데이터와 겹치는지 확인되지 않았으므로
일반화 성능 평가로 보지 않는다.

| 입력 | 페이지 | 탐지·OCR 결과 반환 | 탐지 후보 없음 | 최종 API/실행 오류 |
| --- | ---: | ---: | ---: | ---: |
| 원본 | 162 | 119 | 43 | 0 |
| 촬영본 | 137 | 118 | 19 | 0 |

조건은 confidence ≥ 0.5, 정합 300dpi, Tesseract `kor+eng`, PSM 11,
`regions` 모드다. 안내문·법령 본문도 포함되어 있어 탐지 0건을 모두 미탐으로
계산할 수 없다. 정답 박스·정답 텍스트가 없어 Precision/Recall/mAP/CER는
산출하지 않았다. API 결과를 로컬에 캐시해 코드 재검증 시 같은 입력을 재전송하지
않았다. 중간 API 오류 1건은 한 번 재시도해 해소했다.

실제 입력으로 확인해 수정한 문제:

- A4 원본의 큰 표를 종이 경계로 오인해 제목과 서명란이 잘리거나 페이지가
  기울어졌다. A4 비율 오차 2% 이내이고 네 변의 1% 폭 여백 각각에서 99% 이상이
  밝은 픽셀인 경우 전체 페이지를 유지한다. 원본 154페이지가 이 판단에 해당했고
  촬영본은 0장이었다. 문서 경계를 직접 검출한 것은 아니므로 대체 경로의 낮은
  신뢰도와 검토 경고는 유지한다.
- 로컬 분석 CLI가 대화 세션까지 생성하면서 LLM 전송 게이트에서 항목 ID·순서와
  OCR 숫자 조합을 전화번호로 오인해 중단됐다. 공통 Vision 분석을
  `analyze_document()`로 분리해 로컬 분석에는 LLM payload·RAG·장치 초기화가
  필요 없도록 했다. `build_session()`의 실제 외부 전송 게이트는 유지한다.

남은 한계:

- 학교 동의서는 실제 체크박스 4개·서명 표시 1개를 찾았지만, ‘아니요’의 ‘요’를
  추가 체크박스로 오인했다. 표준근로계약서에서는 ‘(전화:’를 서명란으로 오인했다.
  탐지 수 증가가 그대로 정확도 향상은 아니다.
- 탐지 후보 중 선택지 라벨이 비어 있는 것은 원본 263/1,088개, 촬영본
  445/1,235개다. 라벨이 채워진 나머지도 정확성을 보장하지 않는다. 두 입력군의
  양식 구성이 달라 이 비율 차이를 촬영에 따른 성능 하락률로 해석하지 않는다.
- 체크박스와 글자가 붙어 OCR 단어 박스가 겹치는 경우, 작은 글자와 표 선,
  그림자·원근 왜곡, 미탐 서명란, 서명 역할 판정 개선이 필요하다.
- 가로형·여러 면을 한 장에 넣은 문서는 세로 A4 좌표 가정의 한계가 있다.

로컬 보고서와 페이지별 JSON·탐지 이미지는 `outputs/drive-eval/`에 저장했으며
원본과 함께 Git에서 제외한다. 자동 테스트는 1,346개 통과했다. 기존 RAG의
NumPy 행렬 연산 경고 90개는 별도 확인 대상이며, 이번 Vision 경로에는 RAG가
포함되지 않는다.
