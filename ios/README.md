# iPhone 문서 도우미

SwiftUI 네이티브 앱과 개발용 PC의 Python 분석 서버를 연결한다. 지원 범위는 **iPhone 13 이상, iOS 17 이상**이다. 13 mini·Pro·Pro Max를 포함하고 iPhone SE·iPad·iPhone 12 및 이전 기종은 앱 진입 시 지원 대상이 아니라고 안내한다. 시뮬레이터는 개발 검증용으로 허용한다.

카메라·사진 선택, 원문 낭독, 선택 제안/확정, 작성 후 체크 결과 대조, 마스킹 후보 미리보기를 앱에서 제공한다. 문서 분석은 기존 Python 코드를 재사용하며 Jev나 클라우드 AI를 호출하지 않는다.

## 지금 확인된 범위

2026-11-04 기준:

- Python 전체 회귀 테스트 1,836개 통과. 모바일 HTTPS·토큰·Origin·공개 주소 제한 검사 포함.
- Swift 공통 통신 모듈과 연결 확인 실행 파일의 macOS 빌드 성공.
- Swift 연결 설정·지원 기종·Python JSON 응답 계약 검사 16건 확인.
- 실제 로컬 HTTPS 서버와 Swift 클라이언트의 연결·인증서 확인·응답 해석 성공.
- 동일한 HTTPS 연결로 합성 원문 읽기·선택 확정·재촬영·일치·중복·흐림·초기화 흐름 통과.
- 잘못된 인증서 지문과 토큰의 연결 차단 확인.
- 앱 Swift 소스 구문 검사, Xcode 프로젝트·Info.plist 형식 검사 완료.

**이 PC에는 Xcode 전체 앱, iOS SDK, XCTest가 없어 iPhone 앱 타깃 전체 빌드·시뮬레이터·실기기 검증은 아직 하지 못했다.** Swift 구문 검사나 공통 모듈의 macOS 빌드는 iOS 앱 빌드 성공을 의미하지 않는다. XCTest와 iPhone Simulator 빌드를 실행하는 GitHub Actions를 추가했지만 아직 원격 실행하지 않았다.

## 1. PC 서버 실행

저장소 루트에서 기존 가상환경을 사용한다.

```sh
source venv/bin/activate
export PYTHONPATH=src
python -m docagent.home demo --output outputs/iphone-demo
```

데모를 이미 생성했다면 마지막 명령은 생략한다. 아이폰과 PC를 같은 Wi-Fi에 연결하고 PC의 IPv4 주소를 확인한다. macOS Wi-Fi 기본 인터페이스에서는 `ipconfig getifaddr en0`으로 확인할 수 있다. 아래 `192.168.0.10`은 PC의 실제 주소로 바꾼다.

```sh
python -m docagent.home serve-mobile \
  --host 192.168.0.10 --port 8766 \
  --template outputs/iphone-demo/template.json --demo \
  --connection-dir outputs/iphone-connection
```

- OpenSSL로 서버용 개발 인증서와 개인키를 만들고 HTTPS로 실행한다. OpenSSL이 PATH에 있어야 한다.
- `outputs/iphone-connection/pairing.json`에 서버 주소·인증서 SHA-256 지문·접속 토큰이 저장된다.
- **서버를 다시 켤 때마다 토큰이 바뀐다.** 앱에는 새 연결 정보를 다시 입력한다.
- 인증서 유효기간은 30일이다. PC IP가 달라지거나 인증서가 만료되면 새로운 `--connection-dir` 폴더로 실행하고 새 연결 정보를 사용한다.
- `0.0.0.0`이나 공개 IP로 실행하지 않는다. 아이폰에는 `127.0.0.1`을 입력하면 안 된다. 그것은 아이폰 자신을 가리킨다.
- 선택 상태 이어하기가 필요할 때만 `--session outputs/iphone-connection/session.json`을 추가한다.
- macOS 방화벽이 물으면 사용자가 Python의 로컬 연결 허용 여부를 확인한다. 공유기의 기기 간 통신 차단/게스트 Wi-Fi에서는 연결되지 않을 수 있다.

연결 파일과 개인키는 저장소에 커밋하지 않는다. `pairing.json`을 가진 기기는 같은 네트워크에서 현재 세션에 접근할 수 있다. 이 서버는 **한 사용자의 개발 시연용**이며 여러 사람의 세션을 분리하는 서비스가 아니다.

## 2. Xcode에서 아이폰에 설치

1. Mac에 Xcode를 설치하고 한 번 실행해 필요한 iOS 플랫폼 구성요소를 설치한다.
2. `ios/DocAgent.xcodeproj`를 연다. 외부 Swift 패키지나 XcodeGen 설치는 필요 없다.
3. `DocAgent` 타깃 → `Signing & Capabilities` → 개인 Apple Account의 Team을 선택한다. Bundle Identifier 충돌이 나면 본인만의 값으로 바꾼다.
4. iPhone 13 이상을 연결하고 기기 신뢰·Developer Mode 안내를 따른다.
5. 실행 대상을 연결한 아이폰으로 정하고 Run을 누른다. 앱의 카메라와 로컬 네트워크 권한을 허용한다.

Apple의 [실제 기기에서 앱 실행 안내](https://developer.apple.com/documentation/xcode/running-your-app-on-simulated-or-physical-devices)와 [서명 설정 안내](https://help.apple.com/xcode/mac/current/en.lproj/dev60b6fbbc7.html)를 따른다. 이 작업에서 Apple 계정 로그인·서명·앱 설치는 수행하지 않았다.

## 3. 앱 연결과 시연

1. PC의 `pairing.json`을 개인 아이폰으로 옮겨 앱에서 **연결 정보 파일 불러오기**로 선택하거나, 파일 내용을 연결 입력란에 붙여넣는다.
2. 화면의 PC 주소를 확인하고 사진을 PC로 보내 분석한다는 항목을 켠 뒤 연결한다.
3. 합성 `reference.png`를 사진으로 불러온다. 생성 이미지는 **페이지 전체 스캔**을 켠다. 실제 카메라 촬영은 자동으로 사진 모드다.
4. 전체 원문/현재 항목을 듣고 각 항목에서 ‘동의하지 않음’ → **선택 내용 재확인 → 이 선택으로 확정**을 누른다.
5. `correct.png`를 불러오고 체크 결과를 대조한다. `duplicate.png`, `missing.png`, `opposite.png`, `blurred.png`로 예외 상황을 확인한다.
6. 실제 촬영은 데모 양식을 인쇄해 사용한다. 카메라 프레임을 약 1.2초 간격으로 보내 촬영 상태를 안내하며, 버튼을 누르면 화면 프레임을 분석한다. 기기 사진 앱에 자동 저장하지 않는다.

개인정보가 있는 실제 문서보다 합성 연습지로 먼저 촬영 조건·VoiceOver·읽기 속도·누락 및 중복 처리를 확인한다. 실제 동의서 선정과 정답 라벨링은 별도 작업이다.

## 4. 연결·개인정보 처리 방식

- 앱은 사설 IPv4의 HTTPS 연결만 허용한다. 서버 실행 시 생성한 토큰으로 API를 인증한다.
- 앱이 `pairing.json`의 인증서 지문과 실제 서버 인증서를 비교한다. 지정한 인증서에 한해 신뢰를 구성하고 **호스트·유효기간 검사를 유지**한다. 시스템 인증서 저장소를 변경하거나 전체 TLS 검증을 끄지 않는다.
- 인증 정보·문서 사진을 다른 서버에 보내는 HTTP 리다이렉트를 허용하지 않는다. URLSession은 영구 쿠키/캐시를 사용하지 않는다.
- 모바일 서버에는 웹 페이지에서 세션 토큰을 가져오는 경로가 없다. 브라우저 Origin 요청도 받지 않는다. 기존 PC 로컬 웹 실행은 그대로 유지한다.
- 카메라 프레임과 선택한 문서 사진은 **아이폰 밖의 지정한 PC로 전송된다.** 마스킹은 PC에서 수행하므로, ‘마스킹한 뒤에만 PC로 보낸다’고 설명하면 안 된다.
- 사진을 새로 받거나 선택이 바뀌면 이전 완료 판정을 폐기한다. 네트워크 오류·앱 백그라운드 전환 시 앱의 결과와 연결을 초기화한다. 다시 연결하면 PC의 이전 사진도 초기화한다.
- 앱의 ‘촬영 이미지 지우기’와 정상 연결 종료는 PC의 보유 촬영본을 초기화한다. 강제 종료·네트워크 단절 시 PC 메모리의 사진은 서버 종료·다음 초기화까지 남을 수 있다. PC는 원본을 저장 파일로 쓰지 않지만 OCR의 OS 임시 파일은 사용될 수 있다.
- 자동 마스킹 후보는 모든 개인정보가 가려졌다는 판정이 아니다.
- `PrivacyInfo.xcprivacy`에는 사진·문서 선택 데이터의 앱 기능용 PC 전달을 반영했다. App Store 배포 전에는 실제 보관·운영 방식에 맞춰 개인정보 안내와 배포 설정을 다시 확인한다. 현재는 내부 시연 범위다.

## 개발 검증

Xcode가 설치된 Mac:

```sh
swift test --package-path ios/DocAgentCore
xcodebuild -project ios/DocAgent.xcodeproj -scheme DocAgent \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath outputs/ios-build CODE_SIGNING_ALLOWED=NO build
```

공통 통신만 확인:

```sh
swift run --package-path ios/DocAgentCore docagent-connection-check \
  --check-contracts ios/DocAgentCore/Tests/DocAgentCoreTests/Fixtures/state.json
swift run --package-path ios/DocAgentCore docagent-connection-check \
  outputs/iphone-connection/pairing.json
swift run --package-path ios/DocAgentCore docagent-connection-check \
  --check-workflow outputs/iphone-connection/pairing.json outputs/iphone-demo
```

마지막 명령은 지정한 서버가 합성 홈 데모임을 확인한 뒤 그 세션에서 테스트 선택·촬영을 수행하고 초기화한다. 실제 양식에는 실행하지 않는다.

## 코드 위치

- `DocAgentApp/DocAgentApp.swift`: SwiftUI 화면·사진 선택·연결 정보 불러오기
- `DocAgentApp/CameraView.swift`: 후면 카메라 미리보기·프레임 전달
- `DocAgentApp/HomeModel.swift`: 사용자 작업 순서·요청 중복 방지·원문 TTS·오류 후 초기화
- `DocAgentCore/Sources/DocAgentCore/`: 서버 응답 모델·인증서 확인·API·지원 기종 판정
- `../src/docagent/home/mobile.py`: PC용 HTTPS 실행과 연결 파일 생성
- `generate_project.py`: 소스 파일을 추가했을 때 Xcode 프로젝트 재생성

기종 식별자 매핑은 [DeviceKit의 기종 매핑](https://github.com/devicekit/DeviceKit/blob/master/Source/Device.generated.swift)을 참고했다. DeviceKit 라이브러리 의존성은 추가하지 않았다. iOS 로컬 네트워크 권한은 [Apple 문서](https://developer.apple.com/documentation/bundleresources/information-property-list/nslocalnetworkusagedescription), 개인정보 항목 이름은 [Apple의 데이터 유형 정의](https://developer.apple.com/documentation/bundleresources/app-privacy-configuration/nsprivacycollecteddatatypes/nsprivacycollecteddatatype)를 참고한다.
