# 인식 개선 검증과 원문 전체 낭독

작성일: 2026-10-27

## 이번에 바꾼 것

- 실제 문서의 안내는 `source_text`에 등록한 원문 전체를 그대로 반환한다. 설명을 새로 만들거나 요약한 문장은 운영 안내로 승인할 수 없다.
- 항목별 낭독에 더해 페이지 전체와 문서 전체 낭독을 추가했다. 모든 페이지가 등록·승인되지 않으면 전체 읽기를 거절한다.
- 원문·출처·페이지 수·매칭 조건을 바꾸면 기존 승인 기록을 사용할 수 없다. 페이지 원문 승인 시 원본 파일 SHA-256도 대조한다.
- 전입신고서의 공식 PDF에서 페이지 원문과 항목별 원문을 추출해 미승인 초안으로 저장했다. 우편물 동의 영역은 조건·제공 항목·연락처·동의 문구를 포함한 전체 부분을 보존했다.
- 같은 행의 OCR 단어가 작은 y 좌표 차이 때문에 뒤섞이던 문제를 수정했다. 원래 단어와 신뢰도는 변경하지 않는다.
- 선택 가능한 OCR 표 선 제거와 PSM 설정을 추가했다. 원본·처리본의 문구·좌표·신뢰도를 함께 보존한다.
- RAG 내적 연산에서 발생하던 수치 경고를 해결했다.
- 정답 박스를 추가·수정·삭제하고 클래스를 바꾸는 로컬 검수 화면을 추가했다. 검수 후 승인한 COCO를 평가에 연결할 수 있다.
- 미적용 박스 수정의 소실을 막고, 승인한 정답을 변경하거나 다른 원본과 연결하면 평가를 거절하도록 했다.

## 인식 성능은 어떻게 확인했나

저장된 빈 양식과 탐지 결과를 재사용했다. 이번 작업에서 외부 API 재호출, 모델 재학습은 하지 않았다.

| 개발용 비교 | 기존 결과 | 비교 결과 |
|---|---:|---:|
| 전입신고서의 수동 지정 영역, PSM6, 키워드 17개 | 원본 13개 | 표 선 제거 후 16개 |
| 예술인 동의서의 수동 지정 영역, PSM6, 키워드 6개 | 원본 4개 | 표 선 제거 후 4개 |
| 전입신고서 실제 자동 Crop 경로, 키워드 15개 | 기본 PSM11에서 11개 | PSM6과 표 선 제거 조합에서 7개 |

수동 지정 영역의 개선이 자동 처리 경로의 개선으로 이어지지는 않았다. 따라서 **기본값은 기존 PSM11·전처리 없음으로 유지**했다. `form_lines`는 주변 영역 `regions` 모드에서 직접 선택하는 검수용 옵션이다. 전체 페이지에는 사용하지 못하도록 제한했다.

위 숫자는 지정한 키워드의 포함 여부이며 OCR 문자 정확도, 최종 안내 정확도, 탐지 Recall이 아니다. 두 비교의 Crop 방식·키워드 수가 달라 서로 직접 비교하지 않는다. 실제 분석의 `needs_review` 상태도 유지했다.

RAG는 같은 입력에서 기존 연산의 경고 3건이 0건으로 줄었고, 최대 수치 차이는 약 `2.78e-17`이었다. 비교한 검색 순위와 소수점 6자리 점수는 같았다.

전체 자동 테스트는 **1,625개 통과**, 이번 실행의 경고는 0건이다. 가상 원문 전체로 macOS 음성 파일도 생성했다. 파일 길이는 약 18.8초이며, 테스트에서는 마지막 문장·공백·줄바꿈이 음성 엔진 입력까지 보존되는지 확인했다. 실제 사용자 청취 평가는 별도로 필요하다.

## 원문 낭독 사용 방법

키워드 비교용 `surrounding_text`와 낭독용 `source_text`를 구분한다. 실제 문서 항목은 `reading_mode: "verbatim"`, `source_text`, `source.reference`, `source.sha256`를 갖고, `instruction`이 원문과 정확히 같아야 승인할 수 있다.

페이지 원문은 안내 목록의 `pages`에 별도로 등록한다. 문서 전체 읽기는 `page_count`와 1번부터 마지막까지의 페이지·승인을 모두 확인한다. 페이지 내부 문구는 그대로 두고 페이지 경계에만 빈 줄을 추가한다. 촬영물을 자동으로 식별한 결과가 아니라 **사용자가 명시적으로 선택한 승인 원문**을 읽는 명령이다.

원문 낭독 기능은 아래 가상 문서로 확인할 수 있다.

```sh
PYTHONPATH=src venv/bin/python -m docagent.guide \
  --catalog examples/guidance/catalog.verbatim.demo.json \
  --read-document --document-id verbatim-fixture --document-version demo-1 \
  --demo --speak
```

한 페이지만 읽으려면 `--read-document` 대신 `--read-page --page 1`을 사용한다. 기존 항목별 `docagent.run`도 승인된 운영 목록의 원문을 그대로 읽는다.

전입신고서 원문은 PDF의 텍스트를 추출한 **초안**이다. 사람의 원본 대조가 끝난 뒤에만 다음 명령을 실행한다. 특히 표의 읽기 순서와 조건·유의사항의 누락을 확인해야 한다.

```sh
PYTHONPATH=src venv/bin/python -m docagent.review page-approve \
  outputs/community-forms-20261027/verbatim-review/catalog.draft.json \
  --document-id kr-move-in-all --document-version decree-form15-20260428 --page 1 \
  --source-file outputs/community-forms-20261025/source/move-in-all-20260428.pdf \
  --reviewer "실제 검토자" --confirm-reviewed \
  --output outputs/community-forms-20261027/catalog.page-approved.json
```

이 페이지 승인은 항목 안내·탐지 결과·정답 박스를 함께 승인하지 않는다. 기존 3종 문서의 설명 초안은 원문 전사 전 상태로 유지하며 운영 낭독에 사용할 수 없다.

## 정답 박스 검수와 평가

현재 로컬 검수 파일:

- `outputs/community-forms-20261027/verbatim-review/move-in-all/reference.html`: 원본·항목 원문·페이지 원문 전체
- `outputs/community-forms-20261027/label-review-v2/labels.html`: 정답 박스 편집

박스 화면에서 추가·선택·좌표 수정·클래스 변경·삭제·확대를 할 수 있다. 숫자나 클래스를 변경했으면 ‘수정 적용’ 후 저장한다. 저장 결과는 항상 미승인 초안이다. 저장한 초안은 `label_review prepare ... --draft 파일`로 다시 열 수 있다.

사람이 페이지 전체의 누락·오탐·클래스·경계를 확인한 뒤 평가용 COCO를 만든다.

```sh
PYTHONPATH=src venv/bin/python -m docagent.label_review approve \
  검수후저장한초안.json \
  --image outputs/community-forms-20261025/images/move-in-all-300dpi.png \
  --reviewer "실제 검토자" --confirm-reviewed \
  --output outputs/community-forms-20261027/labels.approved.coco.json
```

평가 명세의 `labels`를 `{"format":"coco","path":"승인한파일경로","image_id":1}`로 연결하면 기존 평가 도구를 사용할 수 있다. 직접 출력한 승인 표식이 있는 COCO는 내용 해시·원본 파일 지문·크기를 검사한다. 외부의 일반 COCO 입력도 계속 지원한다.

**실제 문서에 승인 명령을 대신 실행하지 않았다.** 검수 팩의 정답은 여전히 `labels: null`이고, 현재 Precision·Recall·F1은 미측정이다. 검수 기록의 해시는 변경 감지용이며 검토자의 신원이나 라벨 품질을 자동으로 보증하지 않는다.

## 남은 일

전입신고서의 원문 읽기 순서·항목 범위와 정답 박스를 사람이 검수해야 한다. 인쇄된 서명 표시와 실제 필기 공간 중 어느 영역을 정답으로 삼을지도 맞춰야 한다. 승인 후 실제 문서의 문서 식별·항목 매칭·음성 안내를 검증하고, 개발에 쓰지 않은 별도 촬영본으로 최종 평가해야 한다.

자동 문서 식별, 실제 펜 장치 연결, 최종 촬영본 평가에 필요한 데이터 확보는 아직 남아 있다. 정확도를 확보했다고 표현하거나 검토 플래그를 올려 실행을 강행하지 않았다.

로컬 비교 결과·재현 스크립트는 `outputs/performance-20261027/ocr/`에 있다. 원본 이미지·OCR 증거·검수 HTML·오디오는 Git에 포함하지 않는다.
