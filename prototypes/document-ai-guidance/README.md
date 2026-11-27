# 문서 AI 안내 프로토타입 (Windows · Python)

종이 양식을 촬영해 Gemini로 작성란과 문서 흐름을 분석하고, 사람이 검토한 위치로만
손끝을 비프음·음성으로 안내하는 Windows 노트북용 프로토타입이다.
`hand-tracking` 로컬 프로젝트를 그대로 옮겨 왔으며 `src/docagent` 패키지와는 아직 통합하지 않았다.

> 실험용 코드다. 실제 종이 위 안내(`--experimental-live`)는 정안인 보조자가 함께 있을 때만 실행한다.

## 구성

| 단계 | 주요 파일 | 내용 |
|---|---|---|
| 문서 스캔 | `detection/auto_document_scan.py`, `auto_paper_corners_safe.py` | DocAligner 모서리 검출·투시 보정, 종이 경계 후보와 검토용 미리보기 |
| Gemini 문서 분석 | `detection/gemini_detection.py` | 작성란·체크박스·서명란 탐지 → `gemini_output/<문서>/detection_result.json` |
| 탐지 정제 | `detection/refine_detections.py`, `detection/review_refinements.py` | OpenCV로 박스 보정, 검토 보고서 CSV |
| 작성란 위치 | `detection/writable_targets.py`, `detection/navigation_adapter.py`, `detection/field_review_v7_6.py`, `detection/rebuild_manifest_v7_6.py` | 빈칸 후보 → 사람 검토(`navigation_review.json`) → A4 mm 좌표 `navigation_manifest.json` |
| Workflow | `detection/document_workflow_v8.py`, `detection/workflow_link_v8.py`, `detection/workflow_dialogue_v8.py`, `detection/preview_workflow_v8.py` | Gemini가 의미(신청 종류·순서·조건·역할)만 만들고, 위치는 검토된 manifest와 같은 사진 SHA-256으로만 연결 |
| 손가락 추적 | `detection/multi_field_voice_tracking_v8.py`, `detection/target_guidance_v8.py`, `detection/document_tracker_v7*.py`, `detection/proximity_audio_v7_4.py`, `detection/camera_recovery_v8.py` | MediaPipe 손 랜드마크 + 문서 평면 추적, 방향 음성·거리 비프음·도착 판정, 카메라 끊김 복구 |
| TTS/STT | `detection/narration_v7_1.py`, `detection/workflow_speech_v8.py`, `detection/voice_intents_v8.py`, `detection/diagnose_stt_v8.py`, `diagnose_deepgram_*.py` | Windows SAPI(PowerShell/pyttsx3) 한국어 낭독, Deepgram nova-3 받아쓰기, 음성 명령 해석 |
| 검증 | `step3_*.py`, `step4_*.py`, `step5_*.py` | Workflow 안전 감사, 작성 전후 잉크 흔적 비교 MVP, 다른 문서 일반화 점검 |
| 이전 버전 | `detection/multi_field_voice_tracking_v6~v7_5.py`, `archive/` | v8이 v6·v7·v7.5 모듈을 import하므로 함께 보존. `archive/`는 초기 실험 |
| OCR·RF-DETR 실험 | `ocr/`, `detection/rfdetr_test.py` | Tesseract·PaddleOCR·Roboflow 비교용 |

`patch_v8_*.py`, `fix_v8_*.py`, `repair_v8_*.py`, `step3_fix_review_block.py`는 v8 파일에 **이미 적용된**
일회성 패치 기록이다. 중복 적용 방지 표식이 있지만 다시 실행할 필요는 없다.

### 포함한 데이터

- `examples/`: 테스트용 가상 양식 A/B
- `detection/gemini_output/*/detection_result.json`: 빈 양식 5종 + 시험 양식의 Gemini 탐지 결과
- `detection/refinement_output/*/`: 정제 결과, Workflow, 사람이 검토한 `navigation_*.json`, 검토 CSV

모두 빈 양식의 항목명·안내문·좌표만 담고 있으며 기입된 개인정보는 없다(주민등록번호·전화번호·이메일·계좌 패턴 검사 통과).
JSON 안의 `image`·`source_image` 등은 원작성자 PC의 절대 경로다. 검토 기록이 이 JSON 파일들의
SHA-256으로 연결돼 있어 경로를 고치면 승인 연결이 깨지므로 원본 그대로 두었다.

### 커밋하지 않은 것

- 문서 촬영본·스캔·시각화 이미지(`detection/test_images/` 등): 개인정보 가능. 실시간 안내에는 manifest를 만든 **같은 원본 사진**이 필요하므로 로컬에서 따로 받는다.
- `hand_landmarker.task`: 저장소의 `ios/DocAgentApp/hand_landmarker.task`와 동일한 파일(SHA-256 일치)이다.
- Gemini 원본 응답(`gemini_raw.json`, `workflow_raw_response.json`), 스캔·검증 실행 결과(`scan_output/`, `step*_output/`), 패치 백업, 가상환경, API 키

## 설치 (Windows, Python 3.12)

```powershell
cd prototypes\document-ai-guidance
python -m venv .venv_scan
.\.venv_scan\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item ..\..\ios\DocAgentApp\hand_landmarker.task .
```

API 키는 `.env.example`을 참고해 PowerShell 환경 변수로 설정한다. 코드는 `.env` 파일을 자동으로 읽지 않는다.

```powershell
$env:GEMINI_API_KEY = "..."
$env:DEEPGRAM_API_KEY = "..."
```

## 실행

카메라·마이크·API 없이 확인할 수 있는 것부터 실행한다. 모든 명령은 이 폴더에서 실행한다.

```powershell
# 단위 테스트 (38개)
python -m unittest discover -s tests -p "test_*.py" -v

# 실제 양식 Workflow 안전 감사 (문서1, 읽기 전용)
python -u .\step3_audit_workflow.py
python -u .\step3_3_dialogue_safety_test.py

# 다른 문서 일반화 점검
python -u .\step5_check_generalization_v2.py --doc-id document2

# 대화 흐름 미리보기 (카메라 없음, 도착은 모의 입력)
python .\detection\preview_workflow_v8.py --workflow detection\refinement_output\document1\workflow\document_workflow.json --manifest detection\refinement_output\document1\writable_output\navigation_manifest.json
```

실시간 안내는 원본 사진을 `detection\test_images\document1.jpg`에 둔 뒤 설정 점검부터 한다.

```powershell
$base = "detection\refinement_output\document1"
python .\detection\multi_field_voice_tracking_v8.py --check-setup --workflow $base\workflow\document_workflow.json --manifest $base\writable_output\navigation_manifest.json --targets $base\writable_output\navigation_targets.json --reference detection\test_images\document1.jpg
# 통과하면 --check-setup 대신 --experimental-live --beep --tts --stt 를 붙여 실행
```

## 안전 게이트

- 사람이 검토한 manifest에 있는 항목만 손끝 안내를 시작한다. AI가 추정한 위치로는 안내하지 않는다.
- Workflow와 manifest가 같은 사진(SHA-256)에서 만들어졌는지 확인한다.
- 공무원 기재란(`official`)과 제3자 기재란은 안내하지 않는다.
- 작성 후 검증은 잉크 흔적 변화만 본다. 글자 내용·서명 진위는 판단하지 않는다.

## 알려진 문제

- 실시간 카메라 문서 자동 재탐색의 안정성 검증 미완료
- 긴 도착 비프음의 실시간 재생 안정성 검증 미완료
- 일부 작성란은 자동 위치 검증 미완료 (문서1은 42개 중 18개 안내 보류)
- 작성 후 자동 검증(`step4_*`)은 문서5 기준 범용화 미완료
- 실제 iPhone 앱(`ios/`) 통합 미완료, `src/docagent` 패키지 통합 미완료
- TTS(`winsound`, SAPI)는 Windows 전용
