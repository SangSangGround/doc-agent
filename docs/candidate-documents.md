# 문서 후보 3종 선정과 로컬 검수 결과

확인일: 2026-10-17. 현재 상태는 **후보 선정·안내 초안·검수 자료 준비**다.

후속 작업: 2026-10-25에 주민센터 문서 후보를 별도로 선정하고 Roboflow 학습 이력을
추가 대조했다. [전입신고서 후보 확인 기록](community-form-audit.md)에 범위와 실제
RF-DETR 추론 결과를 정리했다. 아래 내용은 기존 3종 후보의 당시 기록이다.
서비스 지원 확정, 사람 승인, 새로운 모델 학습, Roboflow 재추론은 수행하지 않았다.
기존 운영 안내 목록 `examples/guidance/catalog.production.json`은 비어 있다.

## 선정한 후보

| 후보 | 로컬 자료 ID | v3 분할 | Roboflow 라벨 | 안내 초안 |
|---|---|---|---|---|
| 개인정보 수집·이용동의 및 확인서(예술인) | d117p01 | test | 체크박스 4개 | 개인정보 동의 여부, 기존 융자 여부 2개 |
| 진료기록 열람 및 사본발급 동의서 | d092p01 | test | 서명란 1개 | 본인 또는 법정대리인 서명 1개 |
| 진료기록 열람 및 사본발급 위임장 | d100p01 | test | 서명란 1개 | 위임인 서명 1개 |

현재 학습 분할과 이름으로 직접 중복되지 않고, 이전에 확보한 로컬 원본이 있으며,
한 페이지에서 체크박스·서명 유형을 검토할 수 있는 자료를 우선 골랐다.
의료 관련 서식 2종도 포함되므로, 팀이 계획한 관공서 MVP의 최종 지원 범위와
일치하는지는 별도로 결정해야 한다. 선정 후보를 자동으로 서비스에 등록하지 않는다.

동의서 원본에는 2018-09-28, 위임장에는 2017-06-21 개정 표시가 있다.
이번 작업은 이 로컬 파일을 분석한 것이며 현행 공식 서식 여부를 확인한 것은 아니다.
명세의 `local-...` 버전은 파일 지문으로 구분한 내부 버전이지 법령·발행 버전이 아니다.

## 학습과 테스트 분리에서 확인한 점

[Roboflow v3](https://app.roboflow.com/-0qtd3/my-first-project-lkpc5/3)의
이미지 상세 화면에 표시된 Source Data와 Annotation Data를 읽어 106개의 서로 다른
이미지 ID를 대조했다. Train 76 / Valid 15 / Test 15였다.
이 가운데 v2 분할 기록이 있는 74장은 Train 51 / Valid 15 / Test 8이었다.

후보별 직접 근거:

- [예술인 확인서](https://app.roboflow.com/-0qtd3/my-first-project-lkpc5/3/images/34a92179b32bce541567e7bcdbc3fc4a?split=test): 화면의 v2·v3 분할 모두 test.
- [진료기록 동의서](https://app.roboflow.com/-0qtd3/my-first-project-lkpc5/3/images/fdc5704f342fb9ee098bfc8c3ca70c52?split=test): 화면의 v2·v3 분할 모두 test.
- [진료기록 위임장](https://app.roboflow.com/-0qtd3/my-first-project-lkpc5/3/images/10502f970f2be3cb1b239158e7830952?split=test): 화면의 v3 분할 test. v2 기록 없음.

**test에 있다는 사실만으로 처음 보는 문서 양식이라고 결론 내릴 수 없다.**
파일 이름에서 같은 원본 문서 묶음의 페이지가 학습·테스트로 나뉜 사례를 확인했다.

| 원본 묶음 | Train 페이지 | Test 페이지 |
|---|---|---|
| 직업훈련생계비대부 신청서 | 1, 2, 5, 7 | 4 |
| 주택임대차표준계약서 | 1, 2 | 3, 5 |
| 노령연금지급청구서 | 1 | 2 |
| AIA생명 보험금청구서 | 1, 3, 9 | 4 |
| `인쇄 _ 국가법령정보센터` 파일 묶음 | 10, 11, 14, 15, 21, 24, 30 | 2, 12, 22 |

국가법령정보센터 묶음처럼 여러 서식이 들어 있을 수 있는 파일은 동일 파일 묶음이라는
근거만 확보했다. 모든 페이지가 동일 양식이라는 판정은 아니다.
멘토 신청서/멘토 신청서2처럼 이름이 다른 유사 양식도 별도 대조가 필요하다.

v1 전체 학습 자료, 이름만으로 양식을 식별할 수 없는 스크린샷·`1`·`canada` 등의
학습 이미지 11개, 인쇄 후 촬영한 복제본과의 시각적 중복은 아직 확정하지 않았다.
세 후보는 **v3의 학습 분할에서 제외된 이미지**지만, 모든 학습 이력에서 양식 전체가
제외됐다는 증거는 아직 부족하다. 따라서 로컬 평가 명세의 전체 학습 이력은
`training_membership: unknown`으로 유지한다.

또한 v3 test는 이미 모델 비교에 활용된 자료다. 이번에 안내문과 OCR 설정을 개발하는 데
사용하면 이후의 최종 외부 평가에는 별도의 고정 자료가 필요하다.
인쇄·촬영한 사본은 원본과 같은 양식·원본 그룹으로 묶는다.

학습에 사용한 문서도 실제 지원 문서로 등록할 수 있다. 다만 그 문서를 평가한 결과를
미학습 양식에 대한 일반화 성능으로 발표해서는 안 된다.

## 이번에 구현한 것

- 후보 명세: `examples/guidance/candidates/three-documents.draft.json`.
  원본 파일 SHA-256, 문서 ID·내부 버전·페이지, 분할 근거, 참조 라벨 6개,
  안내 초안 4개, OCR 검토 영역, 마스킹 영역 초안을 포함한다.
- `python -m docagent.candidate_review`: 기존 로컬 분석과 후보 명세를 결합해
  문서별 원문·참조 박스·안내 초안·주변 OCR 비교 화면과 마스킹 검수 화면을 만든다.
- 원본/정합 이미지 지문, DPI에 따른 변환 행렬, 분석 이미지 크기를 대조한다.
  다른 원본이나 다른 분석을 연결하면 출력 전에 실패한다.
- 안내·문서 식별·마스킹 상태는 모두 draft다. 문서 식별 신뢰도 0,
  `identity_verified: false`를 유지하고 기존 분석의 실패 표시를 변경하지 않는다.
- 참조 라벨을 로컬 원본 해상도에 비례 표시하되, 정답·탐지 결과로 주입하지 않는다.
  원본 동일성과 정렬을 검토하기 전까지 평가 명세의 `labels`는 null이다.
- 기존 평가 도구에서 읽을 수 있는 명세와 보고서를 생성했다. 정답이 없는
  Precision/Recall/F1·안내 정확도·마스킹 누락률은 측정 불가로 남긴다.
  이 평가 명세의 중복 검사는 로컬 후보 3개 사이의 비교다. 여기서 중복 0건이 나와도
  Roboflow 전체 학습 자료와의 중복이 없다는 뜻은 아니다.
- 경로 이탈, 파일 변경, 분석 불일치, 승인된 안내의 후보 입력, 좌표 오류,
  HTML 삽입, 결과 덮어쓰기를 검사하는 회귀 테스트를 추가했다.

웹 다운로드를 다시 시도하거나 권한 요청을 허용하지 않았다. 이전에 확보한 로컬
원본만 사용했고, 문서를 외부 API로 전송하지 않았다. API 키나 로그인 정보는
이번에 만든 명세·문서·커밋에 포함하지 않는다.

## 실제 로컬 실행 결과

탐지기는 **OpenCV 휴리스틱**, OCR은 **Tesseract kor+eng**이다.
다음 수치는 RF-DETR/Roboflow 모델의 재평가 결과가 아니다.

| 후보 | 로컬 탐지 수 | 참조 라벨 수 | 전체 페이지 OCR 단어 수 | 분석 결과 |
|---|---:|---:|---:|---|
| 예술인 확인서 | 4 | 4 | 30 | 체크박스 2개 그룹 구성, 제목/선택지 OCR 부족 |
| 진료기록 동의서 | 0 | 1 | 300 | 기본 탐지기가 서명란을 찾지 못함 |
| 진료기록 위임장 | 0 | 1 | 216 | 기본 탐지기가 서명란을 찾지 못함 |

개수가 같다고 박스가 정확히 일치하거나 의미가 맞는 것은 아니다.
세 자료 모두 scan 품질 검사는 pass지만, 정합 신뢰도는 0.70이고
분석의 `needs_review`는 true다. 이 상태에서 사용자 안내를 강행하지 않는다.

사전에 지정한 주변 영역을 `--psm 6`으로 잘라 읽은 진단:

| 항목 | 필수 키워드 결과 | OCR 단어 신뢰도 최솟값 |
|---|---|---:|
| 예술인 개인정보 동의 | `동의함`, `동의하지 않음` 누락 | 0.21 |
| 예술인 기존 융자 여부 | `있음` 누락 | 0.32 |
| 본인 또는 법정대리인 서명 | 필수 키워드 모두 인식 | 0.86 |
| 위임인 서명 | 필수 키워드 모두 인식 | 0.90 |

이는 미리 지정한 영역을 읽은 결과이며 자동 탐지→정확한 항목 매칭 성공을 뜻하지 않는다.
OCR 신뢰도 수치는 보정된 정확도나 전체 문서의 정확도도 아니다.
이 진단 결과로 기존 분석·문서 식별·안내 승인 상태를 대체하지 않았다.

참조 박스를 화면에 겹쳐 확인했을 때, 두 서명 라벨은 인쇄된 `(자필서명)` 문구를
감싸고 있다. 실제 서명을 넣을 빈 공간의 범위와 펜이 이동할 목표점까지 정의된
것은 아니므로, 위치 안내/필기 기능을 연결하기 전에 라벨 기준을 맞춰야 한다.

검증: 신규 테스트 19개, 전체 **1,475 passed**. 실제 세 후보의 검수 HTML을
브라우저에서 열어 체크박스·서명 문구와 참조 박스를 확인했다.

## 다시 실행하기

실제 이미지와 OCR 포함 산출물은 `outputs/` 아래에만 저장하며 Git에서 제외한다.
다른 개발자는 같은 Drive 원본을 로컬에 준비해야 한다. 이미지 파일명만 같은 경우는
허용하지 않고 명세에 기록한 파일 지문까지 일치해야 한다.

```sh
# 각 원본의 로컬 분석. 새 API 키나 모델 다운로드를 사용하지 않는다.
PYTHONPATH=src venv/bin/python -m docagent.analyze outputs/drive-eval/images/d117-1.jpg \
  --detector heuristic --ocr-mode page --capture-mode scan --include-ocr-text \
  --output outputs/candidate-review-20261017/d117/analysis.json
PYTHONPATH=src venv/bin/python -m docagent.analyze outputs/drive-eval/images/d092-1.jpg \
  --detector heuristic --ocr-mode page --capture-mode scan --include-ocr-text \
  --output outputs/candidate-review-20261017/d092/analysis.json
PYTHONPATH=src venv/bin/python -m docagent.analyze outputs/drive-eval/images/d100-1.jpg \
  --detector heuristic --ocr-mode page --capture-mode scan --include-ocr-text \
  --output outputs/candidate-review-20261017/d100/analysis.json

# output은 아직 존재하지 않는 새 디렉터리로 지정한다.
PYTHONPATH=src venv/bin/python -m docagent.candidate_review \
  examples/guidance/candidates/three-documents.draft.json \
  --source-root outputs/drive-eval --analysis-root outputs/candidate-review-20261017 \
  --output outputs/candidate-review-20261017/review-pack --crop-ocr

PYTHONPATH=src venv/bin/python -m docagent.evaluate \
  outputs/candidate-review-20261017/review-pack/evaluation.manifest.json \
  --output-dir outputs/candidate-review-20261017/review-pack/evaluation
```

`review-pack/index.html`을 브라우저로 열면 후보 목록이 나온다. 서버 없이 파일로
열어도 이미지와 문구를 볼 수 있다. `reference.html`은 원본/라벨/안내 비교,
`review.html`은 정합 이미지의 탐지/OCR 표시 및 마스킹 영역 편집 화면이다.
의료 서식의 마스킹 초안과 `masked.preview.png`는 검토용이다. 누락·과도한 가림을
사람이 확인해야 하며 다른 촬영본에 재사용할 수 없다. 예술인 양식의 마스킹 초안은 비어 있다.

공유용으로는 이 Markdown 문서를 사용한다. HTML에는 원본과 OCR 텍스트가 포함되므로
원본 없는 공유 보고서로 취급하지 않는다.

## 다음으로 남는 작업

1. 팀이 실제 지원할 문서 종류와 공식 버전을 확정한다. 이번 3종은 검토 후보다.
2. 실제 사용자에게 읽을 문구 4개를 원문과 비교하여 사람이 검토하고 승인한다.
   동의/비동의나 대리권·자격 판단을 시스템이 대신 결정하지 않는다.
3. 라벨 기준과 이미지 정렬을 확인해 평가용 정답을 확정하고, 이름이 불명확한
   학습 이미지 및 과거 버전과의 중복을 확인한다.
4. 검토된 빈 양식 또는 승인된 마스킹 사본으로 Roboflow 실제 추론을 수행한다.
   이번 OpenCV 결과와 구분해서 기록하고 서명 미탐·체크박스 OCR 오류를 확인한다.
5. 정합·문서 식별·필드 문맥을 검증한 뒤 고정 안내/TTS를 연결하고 실제 촬영본과
   사용자 환경에서 검증한다. JSON의 신뢰도나 검토 표시를 임의로 높여 통과시키지 않는다.

사람 승인 명령과 통합 실행 규약은 [검수·평가 실행 안내](pipeline-review-evaluation.md)를 따른다.
