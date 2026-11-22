import SwiftUI
import AVFoundation
import PhotosUI
import DocAgentCore
import Darwin
import UniformTypeIdentifiers

@main
@MainActor
struct DocAgentApp: App {
    @UIApplicationDelegateAdaptor(AppSnapshotCover.self) private var snapshotCover
    @StateObject private var model = HomeModel()
    @Environment(\.scenePhase) private var scenePhase
    var body: some Scene {
        WindowGroup {
            HomeView(model: model)
                .onChange(of: scenePhase, initial: true) { _, phase in
                    if phase == .background { model.suspend() }
                    if phase == .active { model.activate() }
                }
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.protectedDataDidBecomeAvailableNotification)) { _ in
                    if scenePhase == .active { model.activate() }
                }
        }
    }
}

/// Cover the window itself so a presented document or question sheet is also hidden in the app switcher.
@MainActor
private final class AppSnapshotCover: NSObject, UIApplicationDelegate {
    private let cover = UIView()

    func application(_ application: UIApplication, didFinishLaunchingWithOptions options: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        let center = NotificationCenter.default
        center.addObserver(self, selector: #selector(conceal(_:)), name: UIScene.willDeactivateNotification, object: nil)
        center.addObserver(self, selector: #selector(conceal(_:)), name: UIScene.didEnterBackgroundNotification, object: nil)
        center.addObserver(self, selector: #selector(reveal(_:)), name: UIScene.didActivateNotification, object: nil)
        return true
    }

    @objc private func conceal(_ notification: Notification) {
        guard let scene = notification.object as? UIWindowScene,
              let window = scene.windows.first(where: { $0.windowLevel == .normal && !$0.isHidden }) else { return }
        cover.backgroundColor = .systemBackground
        cover.isUserInteractionEnabled = false
        cover.accessibilityElementsHidden = true
        cover.frame = window.bounds
        cover.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        window.addSubview(cover)
    }

    @objc private func reveal(_ notification: Notification) {
        if let scene = notification.object as? UIWindowScene, cover.window?.windowScene === scene {
            cover.removeFromSuperview()
        }
    }
}

@MainActor
struct HomeView: View {
    @ObservedObject var model: HomeModel
    @State private var photo: PhotosPickerItem?
    @State private var settings = false
    @State private var startPlayback = false
    @State private var confirmDeletion = false
    @State private var retakeAfterWriting = false
    @State private var audition: Task<Void, Never>?
    @State private var soundError: String?
    @StateObject private var sampleSound = CaptureSound()
    @AccessibilityFocusState private var resultFocused: Bool
    @Environment(\.dynamicTypeSize) private var typeSize
    @Environment(\.scenePhase) private var scenePhase
    private var supported: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        var info = utsname()
        uname(&info)
        let identifier = withUnsafePointer(to: &info.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
        return SupportedDevice.allows(identifier)
        #endif
    }
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                header
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        if let message = model.saveError {
                            Text(message).foregroundStyle(DocumentStyle.caution)
                            if model.phase == .idle, UIApplication.shared.isProtectedDataAvailable {
                                Button("저장된 작업 지우기", role: .destructive) { confirmDeletion = true }
                                    .frame(minHeight: 44)
                            }
                        }
                        if !supported {
                            Text("이 앱의 지원 기종은 iPhone 13 이상입니다.").padding().documentCard()
                        } else if model.phase == .idle { home }
                        else { result }
                    }.padding(.horizontal, 20).padding(.top, 25).padding(.bottom, 32)
                    .frame(maxWidth: 580)
                    .frame(maxWidth: .infinity)
                }
                .id(model.phase == .idle)
            }
            .background {
                if model.phase == .idle { DocumentStyle.homeWash.ignoresSafeArea() }
                else { DocumentStyle.canvas.ignoresSafeArea() }
            }.foregroundStyle(DocumentStyle.ink)
            .safeAreaInset(edge: .bottom, spacing: 0) {
                if supported && !model.lines.isEmpty {
                    Button { model.openQuestions() } label: {
                        Label("문서에 질문하기", systemImage: "bubble.left.and.bubble.right.fill")
                    }.buttonStyle(DocumentButtonStyle()).padding(.horizontal, 20).padding(.vertical, 8)
                    .frame(maxWidth: 580).frame(maxWidth: .infinity)
                    .background(DocumentStyle.canvas.ignoresSafeArea(edges: .bottom))
                }
            }
            .toolbar(.hidden, for: .navigationBar)
            .fullScreenCover(isPresented: $model.readingPresented) {
                DocumentPlaybackView(model: model, startPlaying: $startPlayback)
            }
            .fullScreenCover(isPresented: $model.cameraPresented) { CameraSheet(model: model) }
            .fullScreenCover(isPresented: $model.writingPresented, onDismiss: {
                if retakeAfterWriting {
                    retakeAfterWriting = false
                    if scenePhase == .active, model.writingPhoto != nil { startCamera() }
                }
            }) {
                WritingGuideSheet(model: model) {
                    retakeAfterWriting = true
                    model.writingPresented = false
                }
            }
            .sheet(isPresented: $settings) { soundSettings }
            .sheet(isPresented: $model.questionsPresented, onDismiss: {
                if scenePhase == .active, !model.questionsPresented { model.cancelQuestion() }
            }) {
                ConsentQuestionView(model: model)
            }
            .alert("저장된 문서를 지울까요?", isPresented: $confirmDeletion) {
                Button("취소", role: .cancel) {}
                Button("문서 지우기", role: .destructive) { model.clear() }
            } message: {
                Text("이 아이폰에 저장된 사진, 읽던 위치와 대화를 함께 지워요.")
            }
            .onChange(of: photo) { _, item in
                guard let item else { return }
                stopDemo()
                model.importPhoto(item)
                photo = nil
            }
            .onChange(of: model.phase) { _, phase in
                if phase == .finished && !model.cameraPresented && !model.questionsPresented && !model.synthesizer.isSpeaking { resultFocused = true }
            }
            .onChange(of: scenePhase) { _, value in
                if value != .active { stopDemo() }
                if value == .background { startPlayback = false; retakeAfterWriting = false }
            }
            .onDisappear { stopDemo() }
            .onReceive(NotificationCenter.default.publisher(for: UIAccessibility.announcementDidFinishNotification)) {
                model.announcementFinished($0)
            }
        }.tint(DocumentStyle.accent)
    }

    private var header: some View {
        HStack(spacing: 9) {
            Label("문서 도우미", systemImage: "doc.text.fill")
                .font(.headline).foregroundStyle(.white)
                .padding(.horizontal, 13).padding(.vertical, 8)
                .background(DocumentStyle.darkInk, in: Capsule())
            Spacer()
            Button {
                stopDemo()
                settings = true
            } label: {
                Image(systemName: "slider.horizontal.3").font(.body.weight(.semibold))
                    .foregroundStyle(DocumentStyle.ink)
                    .frame(width: 46, height: 46).background(DocumentStyle.card, in: Circle())
            }.accessibilityLabel("설정")
        }
        .padding(.horizontal, 20).padding(.vertical, 7)
        .background(DocumentStyle.header.ignoresSafeArea(edges: .top))
    }

    @ViewBuilder private var home: some View {
        VStack(spacing: 12) {
            Text("종이 문서를 소리로 읽어요").font(.largeTitle.bold())
                .multilineTextAlignment(.center).accessibilityAddTraits(.isHeader)
            if !typeSize.isAccessibilitySize { DocumentIllustration() }
            Text("문서 읽기는 아이폰에서 해요. 맥에 연결하면 OpenAI가 읽은 글자를 문맥과 대조해요.")
                .font(.footnote).foregroundStyle(DocumentStyle.muted).multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            let layout = typeSize.isAccessibilitySize ? AnyLayout(VStackLayout(spacing: 10)) : AnyLayout(HStackLayout(spacing: 10))
            layout {
                PhotosPicker(selection: $photo, matching: .images) {
                    Label("사진 불러오기", systemImage: "photo")
                }
                .buttonStyle(DocumentButtonStyle(prominent: false, whiteSurface: true))
                .simultaneousGesture(TapGesture().onEnded { stopDemo() })
                Button { startCamera() } label: { Label("카메라로 촬영", systemImage: "camera.fill") }
                    .buttonStyle(DocumentButtonStyle())
            }
        }.frame(maxWidth: .infinity).padding(.bottom, 4)
        Button { playDemo() } label: {
            HStack(spacing: 13) {
                Image(systemName: audition == nil ? "play.fill" : "stop.fill").foregroundStyle(.white)
                    .frame(width: 46, height: 46).background(DocumentStyle.action, in: Circle()).accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 5) {
                    Text(audition == nil ? "촬영 소리 미리 듣기" : "미리 듣기 멈추기").font(.headline)
                    Text("구도가 좋아지면 박자가 빨라져요.").font(.subheadline).foregroundStyle(DocumentStyle.muted)
                }.frame(maxWidth: .infinity, alignment: .leading)
                if !typeSize.isAccessibilitySize { ReadingWave().frame(width: 52).clipped() }
            }.padding(16).documentCard()
        }.buttonStyle(.plain)
        if let soundError { Text(soundError).font(.subheadline).foregroundStyle(DocumentStyle.muted) }
        if #available(iOS 17.2, *) {
            VStack(alignment: .leading, spacing: 14) {
                Text("촬영할 때 쓰는 버튼").font(.headline).accessibilityAddTraits(.isHeader)
                if typeSize.isAccessibilitySize { VStack(alignment: .leading, spacing: 12) { volumeKeys } }
                else { HStack(spacing: 14) { volumeKeys } }
            }.padding(16).frame(maxWidth: .infinity, alignment: .leading).documentCard()
        }
    }

    @ViewBuilder private var volumeKeys: some View {
        ForEach([true, false], id: \.self) { up in
            HStack(spacing: 8) {
                Image(systemName: up ? "plus" : "minus").font(.subheadline.bold()).foregroundStyle(DocumentStyle.accent)
                    .frame(width: 28, height: 32).background(DocumentStyle.raised, in: RoundedRectangle(cornerRadius: 8))
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(up ? "볼륨 위" : "볼륨 아래").font(.caption).foregroundStyle(DocumentStyle.muted)
                    Text(up ? "초점 맞추기" : "촬영하기").font(.subheadline.bold())
                }.fixedSize(horizontal: false, vertical: true)
            }.frame(maxWidth: .infinity, alignment: .leading).accessibilityElement(children: .combine)
        }
    }
    private var photoPicker: some View {
        PhotosPicker(selection: $photo, matching: .images) {
            DocumentRow(title: "사진 불러오기", detail: "사진 보관함에서 문서를 선택해요.", symbol: "photo.on.rectangle.angled",
                        tint: DocumentStyle.ink, fill: DocumentStyle.raised)
        }.buttonStyle(.plain).simultaneousGesture(TapGesture().onEnded { stopDemo() })
    }

    @ViewBuilder private var result: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("문서 읽기 결과").font(.largeTitle.bold()).accessibilityAddTraits(.isHeader)
            Text(model.notice).font(.body.weight(.medium)).accessibilityFocused($resultFocused)
            if model.demoForm != nil {
                Label("시연 양식의 인쇄 문구와 작성 위치를 확인했어요.", systemImage: "doc.text.magnifyingglass")
                    .font(.subheadline)
            }
            if let status = model.ocrReviewStatus {
                Label(status, systemImage: "text.magnifyingglass").font(.subheadline)
            }
            if model.canRetryOCRReview {
                Button("문맥 검증 다시 시도") { model.retryOCRReview() }
                    .foregroundStyle(.white).frame(minHeight: 44)
            }
            if !model.lines.isEmpty {
                Button {
                    model.stopSpeech()
                    startPlayback = true
                    model.readingPresented = true
                } label: {
                    Label("전체 원문 듣기", systemImage: "speaker.wave.2.fill")
                        .font(.headline).frame(maxWidth: .infinity, minHeight: 52)
                        .foregroundStyle(DocumentStyle.darkInk).background(DocumentStyle.gold, in: RoundedRectangle(cornerRadius: 14))
                }.buttonStyle(.plain)
            }
            if model.busy {
                HStack {
                    ProgressView().tint(.white).accessibilityHidden(true)
                    Spacer()
                    Button("확인 취소") { model.cancelReview() }.foregroundStyle(.white).frame(minHeight: 44)
                }
            }
        }
        .foregroundStyle(.white).padding(20).frame(maxWidth: .infinity, alignment: .leading)
        .background(DocumentStyle.hero, in: RoundedRectangle(cornerRadius: 22))
        if model.writingPhoto != nil && !model.lines.isEmpty {
            Button {
                model.stopSpeech()
                model.writingPresented = true
            } label: { Label("체크·서명 안내", systemImage: "hand.point.up.left.fill") }
                .buttonStyle(DocumentButtonStyle())
                .accessibilityHint("체크 상태를 듣고 작성할 칸을 골라 손끝 안내를 시작합니다.")
        }
        Button { startCamera() } label: { Label("다시 촬영", systemImage: "camera.fill") }
            .buttonStyle(DocumentButtonStyle(prominent: false, whiteSurface: true))
            .accessibilityHint("새 사진을 찍으면 현재 문서를 교체합니다.")
        if !model.lines.isEmpty {
            LazyVStack(alignment: .leading, spacing: 8) {
                Text("한 줄씩 듣기").font(.headline).accessibilityAddTraits(.isHeader)
                ForEach(Array(model.lines.enumerated()), id: \.element.id) { index, line in
                    Button {
                        model.readingIndex = index
                        model.readLines([line])
                    } label: {
                        HStack(alignment: .center, spacing: 12) {
                            VStack(alignment: .leading, spacing: 5) {
                                if line.uncertain {
                                    Text("인식 불확실").font(.caption.bold()).foregroundStyle(DocumentStyle.caution)
                                        .padding(.horizontal, 8).padding(.vertical, 3)
                                        .background(DocumentStyle.raised, in: Capsule())
                                }
                                Text(line.text).font(.body).multilineTextAlignment(.leading)
                            }
                            Spacer(minLength: 0)
                            Image(systemName: "play.circle.fill").foregroundStyle(DocumentStyle.accent)
                                .font(.title3).accessibilityHidden(true)
                        }.padding(14).frame(maxWidth: .infinity, minHeight: 52, alignment: .leading)
                        .documentCard().contentShape(Rectangle())
                    }.buttonStyle(.plain).accessibilityLabel(line.spoken)
                }
                Button("읽기 멈추기") { model.stopSpeech() }.frame(maxWidth: .infinity, minHeight: 44)
            }
        }
        if !model.busy {
            photoPicker
            if model.writingPhoto != nil {
                Text("이 아이폰에 마지막 문서를 보관해요. 다시 열면 읽던 위치와 대화를 이어갈 수 있어요.")
                    .font(.footnote).foregroundStyle(DocumentStyle.muted)
                Button("문서 지우고 홈으로", role: .destructive) { confirmDeletion = true }
                    .frame(minHeight: 44).frame(maxWidth: .infinity)
            } else {
                Button("홈으로 돌아가기") { model.clear() }.frame(minHeight: 44).frame(maxWidth: .infinity)
            }
        }
    }

    private var soundSettings: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Toggle("음성 안내", isOn: $model.automaticSpeech).font(.headline).frame(minHeight: 44)
                    Divider()
                    Picker("읽기 속도", selection: $model.readingRate) {
                        ForEach([0.8, 1.0, 1.5], id: \.self) { Text("\($0, specifier: "%.1f")배").tag($0) }
                    }.pickerStyle(.menu).frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
                    Divider()
                    AIConnectionSettings()
                }.padding(20).documentCard().padding(20)
            }
            .background(DocumentStyle.canvas).foregroundStyle(DocumentStyle.ink)
            .navigationTitle("설정").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("완료") { settings = false } } }
        }.presentationDetents([.medium, .large]).presentationDragIndicator(.visible).tint(DocumentStyle.accent)
    }

    private func startCamera() { stopDemo(); model.beginCamera() }

    private func stopDemo() {
        audition?.cancel()
        audition = nil
        sampleSound.stop()
    }

    private func playDemo() {
        if audition != nil { stopDemo(); return }
        soundError = nil
        do { try sampleSound.start() } catch {
            soundError = "소리를 재생하지 못했어요. 오디오 출력을 확인해 주세요."
            model.speak(soundError ?? "")
            return
        }
        audition = Task { @MainActor in
            do {
                for fit in [0.1, 0.1, 0.45, 0.45, 0.8, 0.8, 0.98] {
                    try Task.checkCancellation()
                    let level = Int((fit * Double(CameraCue.guidePitches.count - 1)).rounded())
                    sampleSound.play(.guiding(level), quietly: model.guidanceSpeaking)
                    try await Task.sleep(for: .seconds(CameraCue.period(fit: fit)))
                }
                sampleSound.play(.entered, quietly: model.guidanceSpeaking)
                try await Task.sleep(for: .milliseconds(400))
                stopDemo()
            } catch { /* stopDemo already stopped this audition. */ }
        }
    }
}

@MainActor
private struct AIConnectionSettings: View {
    @ObservedObject private var relay = CloudDocumentAPI.shared
    @State private var choosingFile = false
    @State private var message = ""
    @State private var checking = false
    @State private var statusTask: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("AI 연결").font(.headline).accessibilityAddTraits(.isHeader)
            Text(relay.pairedAddress.map { "등록된 맥 · " + $0 } ?? "맥에서 만든 pairing.json을 열거나 연결 정보를 붙여넣으세요.")
                .font(.subheadline)
            Button("연결 파일 열기") { choosingFile = true }.frame(minHeight: 44)
            PasteButton(payloadType: String.self) { strings in
                if let text = strings.first { pair(text) }
            }.frame(minHeight: 44).accessibilityLabel("맥 연결 정보 붙여넣기")
            if relay.isConfigured {
                Button(checking ? "연결 확인 중" : "연결 확인") { check() }
                    .frame(minHeight: 44).disabled(checking)
                Button("연결 해제") {
                    cancelStatus()
                    do {
                        try relay.forget()
                        report("맥 연결 정보를 지웠어요.")
                    } catch { report(error.localizedDescription) }
                }.frame(minHeight: 44)
            }
            if !message.isEmpty { Text(message).font(.subheadline) }
        }
        .onDisappear { cancelStatus() }
        .fileImporter(isPresented: $choosingFile, allowedContentTypes: [.json]) { result in
            do {
                let url = try result.get()
                let accessed = url.startAccessingSecurityScopedResource()
                defer { if accessed { url.stopAccessingSecurityScopedResource() } }
                let handle = try FileHandle(forReadingFrom: url)
                defer { try? handle.close() }
                guard let data = try handle.read(upToCount: 8192), data.count < 8192,
                      let text = String(data: data, encoding: .utf8) else { throw ConnectionError.invalidPairing }
                pair(text)
            } catch CocoaError.userCancelled { return }
            catch { report("연결 파일을 읽지 못했어요. 맥에서 만든 pairing.json을 선택해 주세요.") }
        }
    }

    private func pair(_ text: String) {
        cancelStatus()
        do {
            try relay.pair(text)
            report("맥을 등록했어요. 연결 확인을 눌러 주세요.")
        } catch { report(error.localizedDescription) }
    }

    private func check() {
        checking = true
        message = ""
        statusTask = Task {
            do {
                let status = try await relay.status()
                try Task.checkCancellation()
                let answers = status.answersReady ? "문서 질문용 OpenAI 키가 설정돼 있어요." : "문서 질문용 OpenAI 키가 필요해요."
                let fields = status.fieldsReady == true ? "일반 문서 작성란용 Gemini 키가 설정돼 있어요." : "일반 문서 작성란 분석에는 최신 중계와 Gemini 키가 필요해요."
                report("맥에 연결됐어요. " + answers + " " + fields)
            } catch {
                guard !Task.isCancelled else { return }
                report(error.localizedDescription)
            }
            checking = false
        }
    }

    private func cancelStatus() {
        statusTask?.cancel()
        statusTask = nil
        checking = false
    }

    private func report(_ text: String) {
        message = text
        if UIAccessibility.isVoiceOverRunning {
            UIAccessibility.post(notification: .announcement, argument: text)
        }
    }
}

@MainActor
private final class LinePlayer: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published private(set) var index = 0
    @Published private(set) var playing = false
    private weak var model: HomeModel?
    private var utterance: AVSpeechUtterance?
    private var completed = false

    func attach(_ model: HomeModel, autoplay: Bool) {
        self.model = model
        model.stopSpeech()
        model.synthesizer.delegate = self
        index = model.lines.indices.contains(model.readingIndex) ? model.readingIndex : 0
        if autoplay { play(index) }
    }

    func play(_ index: Int) {
        guard let model, model.lines.indices.contains(index) else { return }
        self.index = index
        if model.readingIndex != index { model.readingIndex = index }
        completed = false
        let next = AVSpeechUtterance(string: model.lines[index].spoken)
        next.voice = AVSpeechSynthesisVoice(language: "ko-KR")
        next.rate = AVSpeechUtteranceDefaultSpeechRate * Float(model.readingRate)
        utterance = next
        model.synthesizer.stopSpeaking(at: .immediate)
        playing = true
        model.synthesizer.speak(next)
    }

    func toggle() {
        guard let synth = model?.synthesizer else { return }
        if playing {
            if synth.pauseSpeaking(at: .immediate) { playing = false }
        } else if synth.isPaused, synth.continueSpeaking() {
            playing = true
        } else {
            play(completed ? 0 : index)
        }
    }

    func interrupt() {
        utterance = nil
        model?.stopSpeech()
        playing = false
    }

    func stop() {
        interrupt()
        model?.synthesizer.delegate = nil
        model = nil
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish spoken: AVSpeechUtterance) {
        let identity = ObjectIdentifier(spoken)
        Task { @MainActor [weak self] in
            guard let self, let model = self.model, let current = self.utterance, ObjectIdentifier(current) == identity else { return }
            self.utterance = nil
            if self.index + 1 < model.lines.count { self.play(self.index + 1) }
            else { self.playing = false; self.completed = true }
        }
    }
}

@MainActor
private struct DocumentPlaybackView: View {
    @ObservedObject var model: HomeModel
    @Binding var startPlaying: Bool
    @StateObject private var player = LinePlayer()
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if model.lines.indices.contains(player.index) {
                        VStack(spacing: 13) {
                            ReadingWave()
                            Text("\(player.index + 1) / \(model.lines.count)줄").font(.subheadline.bold())
                            Text(model.lines[player.index].text).font(.title3.bold())
                                .multilineTextAlignment(.center).fixedSize(horizontal: false, vertical: true)
                            HStack(spacing: 18) {
                                control("backward.end.fill", "이전 줄", disabled: player.index == 0) { player.play(player.index - 1) }
                                control(player.playing ? "pause.fill" : "play.fill",
                                        player.playing ? "일시 정지" : "재생", primary: true) { player.toggle() }
                                control("forward.end.fill", "다음 줄", disabled: player.index + 1 >= model.lines.count) { player.play(player.index + 1) }
                            }
                        }.foregroundStyle(.white).padding(20).frame(maxWidth: .infinity)
                        .background(DocumentStyle.hero, in: RoundedRectangle(cornerRadius: 22))
                    }
                    Text("문서 원문").font(.headline).accessibilityAddTraits(.isHeader)
                    ForEach(Array(model.lines.enumerated()), id: \.element.id) { position, line in
                        Button { player.play(position) } label: {
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    Text("\(position + 1)번째 줄").font(.caption.bold()).foregroundStyle(DocumentStyle.muted)
                                    Text(line.uncertain ? "인식 불확실" : "인식됨").font(.caption.weight(.semibold))
                                        .foregroundStyle(DocumentStyle.ink).padding(.horizontal, 9).padding(.vertical, 3)
                                        .background(DocumentStyle.raised, in: Capsule())
                                    Spacer(minLength: 0)
                                }
                                Text(line.text).font(.body).multilineTextAlignment(.leading)
                            }.padding(14).frame(maxWidth: .infinity, alignment: .leading).documentCard()
                        }.buttonStyle(.plain).accessibilityLabel("\(position + 1)번째 줄. " + line.spoken)
                        .accessibilityAddTraits(position == player.index ? .isSelected : [])
                    }
                }.padding(18).frame(maxWidth: 580).frame(maxWidth: .infinity)
            }
            .background(DocumentStyle.canvas).foregroundStyle(DocumentStyle.ink)
            .navigationTitle("전체 원문 듣기").navigationBarTitleDisplayMode(.inline)
            .toolbarColorScheme(.light, for: .navigationBar)
            .toolbarBackground(DocumentStyle.gold, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("닫기") { dismiss() }.foregroundStyle(DocumentStyle.darkInk) } }
        }
        .tint(DocumentStyle.accent)
        .onAppear {
            player.attach(model, autoplay: startPlaying && scenePhase == .active)
            startPlaying = false
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { player.interrupt() }
        }
        .onDisappear { player.stop() }
    }

    private func control(_ symbol: String, _ label: String, disabled: Bool = false,
                         primary: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol).font(primary ? .title2.bold() : .headline)
                .foregroundStyle(DocumentStyle.darkInk)
                .frame(width: primary ? 58 : 46, height: primary ? 58 : 46)
                .background(primary ? DocumentStyle.gold : .white, in: Circle())
        }
        .buttonStyle(.plain).disabled(disabled).opacity(disabled ? 0.55 : 1)
        .accessibilityLabel(label)
    }
}
