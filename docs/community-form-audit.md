# 주민센터 문서 후보와 학습 중복 확인

확인일: 2026-10-25. **전입신고서(세대 모두 이동)**를 외부 문서 개발·검수 후보로 선정했다.
아래 Roboflow 프로젝트에서 확인한 학습 자료에는 이 서식을 발견하지 못했다.
서비스 지원 승인, 정답 라벨 승인, 최종 성능 평가 완료를 뜻하지 않는다.

## 확인 범위

로그인된 Roboflow의 Versions → Images → Raw Data 화면에서 원본 이름인
`Annotation Data.key`, 이미지 ID, 버전별 split을 직접 대조했다.
브라우저에서 데이터셋 ZIP 다운로드 권한이 거부되어 파일 내보내기는 중단했고,
화면 열람으로 확인했다. 라벨·분할·모델을 변경하거나 재학습하지 않았다.

| Roboflow 버전 | 전체 | Train | Valid | Test | 모델 |
|---|---:|---:|---:|---:|---|
| [v1](https://app.roboflow.com/-0qtd3/my-first-project-lkpc5/1) | 15 | 10 | 3 | 2 | YOLO26 Small |
| [v2](https://app.roboflow.com/-0qtd3/my-first-project-lkpc5/2) | 74 | 51 | 15 | 8 | RF-DETR NAS |
| [v3](https://app.roboflow.com/-0qtd3/my-first-project-lkpc5/3) | 106 | 76 | 15 | 15 | RF-DETR NAS |

- v3의 서로 다른 이미지 106장 전체의 원본 이름·분할 이력을 읽었다.
- 그중 v2 이력이 있는 74장의 분할은 Train 51 / Valid 15 / Test 8이었다.
- v3 학습 76장 중 51장은 v2에서도 학습 자료였다.
- v1 학습 10장을 별도로 읽었고, 원본 이름은 모두 v3 학습 목록에도 있었다.
  v1은 640×640 변환본이므로 이름 대조를 픽셀 동일성 검사로 표현하지 않는다.
- v3 학습 중 이름이 불명확한 18장은 화면의 실제 서식을 열어 확인했다.
- v3 Valid/Test의 불명확한 법령 묶음·스크린샷 8장도 화면에서 확인했다.

팀 설명의 ‘Ver.1 74장 / Ver.2 106장’은 Roboflow의 **v2 / v3**에 해당한다.
Roboflow v1은 그보다 앞선 15장짜리 실험이다.

확인은 이 프로젝트에 보이는 버전에 한정한다. 기반 모델의 사전학습 자료,
다른 프로젝트나 삭제된 학습 이력, 외부에서 수행한 학습은 확인하지 않았다.
`training_membership: excluded`도 이 프로젝트의 확인 범위에 대한 기록이다.
PDF의 제목·개정일·레이아웃을 함께 비교하며, 파일명·파일 해시만으로 새 양식이라고 판단하지 않는다.

## 파일명만으로는 알 수 없었던 학습 자료

아래 ID는 [v3 이미지 목록](https://app.roboflow.com/-0qtd3/my-first-project-lkpc5/3/images)의 상세 화면에서 확인한 값이다.
확인 주소는 `https://app.roboflow.com/-0qtd3/my-first-project-lkpc5/3/images/<ID>?split=train`이다.

| 원본 파일명/페이지 | 화면에서 식별한 내용 | 이미지 ID |
|---|---|---|
| 국가법령정보센터 10 | 외국인 및 외국국적동포 세대별 주민등록표 기록·정정·변경·삭제 신청서 | dd5ab704d12b21dfed06fabc39bac4fc |
| 국가법령정보센터 11 | 외국인·외국국적동포 관련 담당 공무원 확인사항 및 유의사항 | 1392924f1c94d4828473d4aaf9f9ce44 |
| 국가법령정보센터 14 | 전입신고 등 사실 통보 서비스 신청서 | 9614410d3d7ea85c3453e72a454f2c23 |
| 국가법령정보센터 15 | 주민등록표 열람 또는 등·초본 교부 신청서 | 7617fc32e2b0d6557fbb4e262d1dda5d |
| 국가법령정보센터 21 | 주민등록표 열람 또는 등·초본 교부 신청서(위임용) | 8cda845e9e339790b6d1f4261252b721 |
| 국가법령정보센터 24 | 전입세대확인서 신청서의 뒤쪽 구비서류·동의·유의사항 | d8dbbd3ba0565ba862fac71b9bc15555 |
| 국가법령정보센터 30 | 외국인·외국국적동포 관련 담당 공무원 확인사항 및 유의사항 | fe0c3840e1ee9f9e1c8f004dfe232f83 |
| 2026-07-29 212342 | 심뇌혈관질환 예방관리 사업 개인정보 동의서 | b784c2cdec23fcfbebe75ed681ab063d |
| 2026-07-29 212626 | 저수조청소업 개설·변경 신고서 | 7fc36ab6e068f8db29cc5bdb8b9c712a |
| 2026-07-29 212734 | 전용상수도·공업용수도 설치 인가·변경 신청서 | 64f74e0cd1ef750965ef797805b67164 |
| 화면 캡처 2026-07-29 212526 | 재활기구 대여 신청서 | 5c81740973d69e732a5930601f0d181c |
| 2026-09-29 221929 | 대학 국제교류 관련 개인정보 수집·이용 동의서 | c6403cb479f80d4c0aad1cd20aad678f |
| 2026-10-05 160452 | 대학 국제교류 관련 개인정보 수집·이용 동의서 | 117346fa6f9fd44b31c7e70e650b88ed |
| 2026-10-05 160702 | 교환(방문)학생 파견 전 휴학예정 확인서 | b5babc36295886cf222b15d41ad7f1f5 |
| 2026-10-05 160828 | 교환·방문학생 포기서 | 9730a197818b0b0ead8c58400088f0bb |
| 2026-10-05 160929 | 교환학생 과목인정서 | 9f03b8d28ee90ea2cef5e78288b30a3d |
| 1_page_001 | 학부연구생 신청 동기·첨부서류·서약 부분 | f5a028132cf1ce8f222d36e941685ff9 |
| canada_page_001 | 캐나다 연구인턴십 선발·지원 개인정보 동의서 | 1fbbc035133a6153c228d21ed7b20b61 |

‘전입신고 등 사실 통보 서비스 신청서’와 ‘전입세대확인서 신청서’는
이번 후보인 ‘전입신고서(세대 모두 이동)’와 서로 다른 서식이다.
이름에 ‘전입’이 있다는 이유만으로 같은 서식으로 합치지도 않았다.

## 후보에서 제외한 주민센터 서류

- 모바일 주민등록증 발급·재발급 신청서: v1/v2/v3 학습에서 확인.
- 주민등록표 열람·등초본 교부 신청서 및 위임용: 법령 묶음의 학습 페이지에서 확인.
- 인감증명서 발급 위임장 또는 미성년자 동의서: v2/v3 학습에서 확인.
- 확정일자 신청서: v2/v3 학습에서 확인.
- 혼인신고서: v2/v3 학습에서 확인.
- 주민등록증 분실 신고서: v2/v3 검증 자료이므로 새 외부 평가 후보에서는 제외.

## 선정 서식과 개발·평가 구분

출처: [국가법령정보센터 전입신고서 PDF](https://law.go.kr/LSW/flDownload.do?bylClsCd=110202&flSeq=163702291&gubun=).
PDF 표기는 주민등록법 시행령 별지 제15호서식, 개정일 2026-04-28이다.
정확한 원본 PDF와 렌더링 이미지 SHA-256은
`examples/guidance/candidates/community-form.draft.json`에 기록했다.
이번 구현은 이 버전의 빈 서식 1쪽을 대상으로 하며 모든 전입신고 유형을 지원하지 않는다.

- 체크박스 참조 초안 8개: 전입 사유 7개, 우편물 전송 관련 정보 제공 동의 1개.
- 서명 참조 초안 3개: 신고인, 세대주, 우편물 전송 서비스 신청인.
- 고정 안내 초안 5개: 위 세 서명 영역, 전입 사유 묶음, 우편물 전송 동의.
- 참조 박스는 PDF의 문자 좌표로 제안했다. 서명 박스는 인쇄된 `(서명 또는 인)`
  표시 영역이며 실제 펜을 놓을 빈 공간 좌표가 아니다. 사람 검수 전에는 정답으로 사용하지 않는다.
- 원문 해석·조건·선택지·라벨 경계와 안내문을 검토한 뒤 승인해야 한다.
  `catalog.production.json`은 그대로 비어 있고 안내 초안은 모두 draft다.

**모델 학습 제외**와 **최종 테스트로 한 번도 사용하지 않음**은 서로 다른 조건이다.
이 빈 원본은 모델의 학습에는 없지만 안내 작성·OCR 진단·통합 개발에 쓰는 자료다.
따라서 검수 팩의 `independent_external_test`는 false로 유지한다.

지원 문서의 원문을 근거로 안내 DB를 만드는 것은 서비스의 사전 지식 구축이다.
최종 평가는 설정을 고정한 뒤 개발에 사용하지 않은 별도 촬영본으로 수행할 수 있다.
촬영본은 이 원본과 같은 `form_id`/`source_id`로 묶고, 이 서식이나 그 촬영본을
RF-DETR 재학습에 넣었다면 미학습 양식 평가 대상으로 사용하지 않는다.
한 서식의 여러 촬영본은 여러 독립 문서 양식으로 세지 않는다.

학습·추론 분리만으로 Precision/Recall이 나오는 것은 아니다.
동일 이미지의 검수된 정답 박스와 저장된 예측을 연결해야 하며, 1종 결과는 이 서식의
시험 결과로만 보고한다. 모든 주민센터 서류의 성능으로 일반화하지 않는다.

## 실행

이미지·OCR·검수 HTML은 Git에서 제외되는 `outputs/community-forms-20261025/`에 있다.
후보 설정에 등록된 이미지는 **300dpi, 2480×3505**다. 같은 원본을 Poppler로 렌더링할 때:

```sh
pdftoppm -r 300 -png -singlefile \
  outputs/community-forms-20261025/source/move-in-all-20260428.pdf \
  outputs/community-forms-20261025/images/move-in-all-300dpi
```

분석·후보 검수 생성 명령은 아래와 같다. 실제 빈 양식임을 확인한 파일만 전송한다.
로컬 `.env`의 Roboflow API 키와 `inference-sdk`, Tesseract의 `kor+eng` 언어 데이터가 필요하다.
사용한 모델을 환경 변수로 명시한다. 기존 분석 결과가 있으면 재호출할 필요가 없다.
다른 렌더러로 PNG가 달라지면 내용을 확인한 뒤 명세의 이미지 지문을 갱신해야 한다.

```sh
ROBOFLOW_MODEL_ID='-0qtd3/my-first-project-lkpc5-3-rfdetr-nas-t1--f98b3b' \
  PYTHONPATH=src venv/bin/python -m docagent.analyze \
  outputs/community-forms-20261025/images/move-in-all-300dpi.png \
  --detector roboflow --conf 0.5 --ocr-mode page --lang kor+eng --dpi 300 \
  --capture-mode scan --allow-unmasked-blank --include-ocr-text \
  --output outputs/community-forms-20261025/analysis-300dpi/move-in-all/analysis.json

PYTHONPATH=src venv/bin/python -m docagent.candidate_review \
  examples/guidance/candidates/community-form.draft.json \
  --source-root outputs/community-forms-20261025 \
  --analysis-root outputs/community-forms-20261025/analysis-300dpi \
  --output outputs/community-forms-20261025/review-pack-300dpi --crop-ocr

PYTHONPATH=src venv/bin/python -m docagent.evaluate \
  outputs/community-forms-20261025/review-pack-300dpi/evaluation.manifest.json \
  --output-dir outputs/community-forms-20261025/review-pack-300dpi/evaluation
```

출력 폴더는 새 경로여야 한다. `reference.html`에서 빨간 참조 박스·원문·안내 초안을
확인하고, `review.html`에서 실제 탐지·OCR 및 재확인 상태를 확인한다.
참조 초안을 검수·확정하기 전에는 생성된 평가 명세의 `labels: null`을 유지한다.

`docagent.evaluate ... --require-unseen`은 학습 중복·이력 미상·양식 그룹 미상,
학습/검증 split, 정답·예측 누락이 있으면 평가를 거절한다.
이 옵션은 입력 명세를 검사하며 전체 학습 이력과 라벨의 타당성을 대신 검수하지 않는다.
초안 상태에서 숫자가 나오도록 승인·신뢰도 값을 올리지 않는다.

## 실제 실행 결과와 남은 검수

RF-DETR 모델은 `-0qtd3/my-first-project-lkpc5-3-rfdetr-nas-t1--f98b3b`,
탐지 임계값은 0.5, OCR은 Tesseract `kor+eng`의 페이지 모드다.
실행 전 모델 ID·입력 지문·설정·탐지기 코드 지문·패키지 버전을
`baseline-300dpi-settings.json`에 기록했다. 새 모델 학습은 수행하지 않았다.

| 입력 | 체크박스 탐지 | 서명란 탐지 | OCR 단어 수 | 용도 |
|---|---:|---:|---:|---|
| 1557×2200, 약 188dpi | 5 | 1 | 933 | 최초 진단 |
| 2480×3505, 300dpi | 6 | 2 | 917 | 현재 검수 기준 |

API 호출에 성공한 실행은 위 두 건이다. 최초 저해상도 입력도 CLI에는 `--dpi 300`을
지정했으나 원본의 실제 픽셀 밀도는 약 188dpi였다. 이후 원본 PDF를 300dpi로
다시 렌더링했다. 모델과 탐지 임계값은 두 실행에서 동일하다.
해상도를 조정하며 결과를 확인했으므로 이 자료는 개발 자료다.

**이 표는 탐지 개수이며 Precision·Recall·F1이 아니다.** 참조 초안 11개와의
일대일 일치 여부·경계·누락·오탐을 검수하지 않았으므로 8/11을 정확도로 계산하지 않는다.
OCR 단어 수도 인식 정확도가 아니다. 생성한 평가 보고서는 정답 없는 지표를
모두 ‘측정 불가’로 표시하고, `--require-unseen`은 정답 누락 때문에 실행을 거절한다.

300dpi 분석의 `needs_review`는 true이고, 정합·탐지·OCR·구조화 단계 모두
자동 통과하지 못했다. 문서 식별과 안내 5개도 미승인 상태여서 실제 서비스 안내를
활성화하지 않았다. 수동 지정한 주변 영역 OCR에서도 신고인·전입 사유·우편물 동의
영역의 필수 키워드가 일부 누락됐다. 이 진단을 자동 필드 식별 성공으로 세지 않는다.

다음 작업은 다음과 같다.

1. `review-pack-300dpi/move-in-all/reference.html`에서 원문·박스 11개·안내 5개를
   사람이 대조한다. 특히 서명 표시와 실제 작성 공간 중 무엇을 라벨링할지 기준을 맞춘다.
2. 정답 박스를 확정해 COCO/YOLO 라벨로 연결하고, 이 개발 입력의 탐지 지표를 계산한다.
   안내문 승인·문서/필드 식별 검수도 별도로 진행한다.
3. 정합·OCR·필드 연결을 개선한 뒤 모델·임계값·안내 DB·처리 설정을 고정한다.
4. 개발에 쓰지 않은 별도 촬영본과 검수된 정답으로 최종 평가한다.
   새 촬영본도 같은 양식·원본 그룹으로 기록하고, 문서 1종의 결과라는 범위를 명시한다.

## 코드 변경과 검증

- 공식 PDF 출처·원본 지문·학습 대조 근거를 후보 검수 팩에 보존하도록 확장했다.
- 미학습 탐지 평가 옵션을 추가해 학습 포함/미상, 그룹 미상, 학습·검증 분할,
  정답·예측 누락 시 점수를 내지 않도록 했다. 일반 진단 모드는 계속 사용할 수 있다.
- 테스트 **1,492개 통과**. RAG 행렬 연산 경로에서 런타임 경고 90건이 관찰됐으며,
  이를 실제 서비스 성능이나 품질 검증 통과로 해석하지 않는다.
- 현재 가상환경에 `inference-sdk 1.8.2`를 설치했다. 실행 환경은 NumPy 2.2.6,
  OpenCV 패키지 `opencv-python 4.12.0.88`, `pytesseract 0.3.13`이며 `pip check`를 통과했다.
- 문서 이미지, OCR 결과, 검수 HTML, API 키는 커밋하지 않는다. 이 Markdown은
  원본 이미지를 포함하지 않은 작업 공유용 기록이다.
