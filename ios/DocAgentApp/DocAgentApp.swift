import SwiftUI
import PhotosUI
import DocAgentCore
import Darwin

@main
@MainActor
struct DocAgentApp: App {
    @StateObject private var model = HomeModel()
    @Environment(\.scenePhase) private var scenePhase
    var body: some Scene {
        WindowGroup {
            HomeView(model: model)
                .onChange(of: scenePhase) { _, phase in
                    if phase == .background {
                        model.cameraPresented = false
                        model.clear()
                    }
                }
        }
    }
}

@MainActor
struct HomeView: View {
    @ObservedObject var model: HomeModel
    @State private var photo: PhotosPickerItem?
    @State private var settings = false
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
            .background(DocumentStyle.canvas).foregroundStyle(DocumentStyle.ink)
            .toolbar(.hidden, for: .navigationBar)
            .fullScreenCover(isPresented: $model.cameraPresented) { CameraSheet(model: model) }
            .sheet(isPresented: $settings) { soundSettings }
            .sheet(isPresented: $model.questionsPresented, onDismiss: { model.cancelQuestion() }) {
                ConsentQuestionView(model: model)
            }
            .onChange(of: photo) { _, item in
                guard let item else { return }
                stopDemo()
                model.importPhoto(item)
                photo = nil
            }
            .onChange(of: model.phase) { _, phase in
                if phase == .finished && !model.questionsPresented && !model.synthesizer.isSpeaking { resultFocused = true }
            }
            .onChange(of: scenePhase) { _, value in if value != .active { stopDemo() } }
            .onDisappear { stopDemo() }
            .onReceive(NotificationCenter.default.publisher(for: UIAccessibility.announcementDidFinishNotification)) {
                model.announcementFinished($0)
            }
        }.tint(DocumentStyle.accent)
    }

    private var header: some View {
        HStack(spacing: 9) {
            Image(systemName: "doc.on.doc.fill").symbolRenderingMode(.palette)
                .foregroundStyle(DocumentStyle.accent, Color(red: 0.79, green: 0.66, blue: 0.94))
                .font(.title2).accessibilityHidden(true)
            Text("문서 도우미").font(.title3.bold())
            Spacer()
            Button {
                stopDemo()
                settings = true
            } label: {
                Image(systemName: "slider.horizontal.3").font(.body.weight(.semibold))
                    .frame(width: 46, height: 46).background(DocumentStyle.canvas, in: Circle())
            }.accessibilityLabel("소리 설정")
        }
        .padding(.horizontal, 20).padding(.vertical, 7)
        .background(DocumentStyle.card.ignoresSafeArea(edges: .top))
        .overlay(alignment: .bottom) { Rectangle().fill(DocumentStyle.border).frame(height: 1).accessibilityHidden(true) }
    }

    @ViewBuilder private var home: some View {
        Text("문서 읽기").font(.largeTitle.bold()).accessibilityAddTraits(.isHeader)
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 6) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("종이 문서를\n소리로 읽어요").font(.title2.bold()).fixedSize(horizontal: false, vertical: true)
                    Text("종이 위에 휴대폰을 들어 주세요.")
                        .font(.subheadline).foregroundStyle(DocumentStyle.muted).fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                if !typeSize.isAccessibilitySize { DocumentIllustration() }
            }
            Button { startCamera() } label: {
                HStack { Image(systemName: "camera.fill"); Text("카메라로 촬영"); Spacer(); Image(systemName: "arrow.right") }
            }.buttonStyle(DocumentButtonStyle()).accessibilityLabel("카메라로 촬영")
        }
        .padding(20).frame(maxWidth: .infinity, alignment: .leading).background(DocumentStyle.wash, in: RoundedRectangle(cornerRadius: 24))
        VStack(spacing: 12) {
            photoPicker
            Button { playDemo() } label: {
                DocumentRow(title: audition == nil ? "촬영 소리 미리 듣기" : "미리 듣기 멈추기",
                            detail: "박자가 빨라지면 구도가 좋아지고 있어요.", symbol: "waveform",
                            tint: DocumentStyle.clay, fill: DocumentStyle.peach, accessory: audition == nil ? "play.fill" : "stop.fill")
            }.buttonStyle(.plain)
        }
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
                    .frame(width: 28, height: 32).background(DocumentStyle.lavender, in: RoundedRectangle(cornerRadius: 8))
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
                        tint: DocumentStyle.pine, fill: DocumentStyle.mint)
        }.buttonStyle(.plain).simultaneousGesture(TapGesture().onEnded { stopDemo() })
    }

    @ViewBuilder private var result: some View {
        Text("문서 읽기 결과").font(.largeTitle.bold()).accessibilityAddTraits(.isHeader)
        VStack(alignment: .leading, spacing: 14) {
            Label(model.busy ? (model.phase == .recognizing ? "글자 읽는 중" : "사진과 대조 중") :
                    model.lines.isEmpty ? "다시 확인해 주세요" : "문서 원문",
                  systemImage: model.busy ? "text.magnifyingglass" : model.lines.isEmpty ? "doc.viewfinder" : "text.alignleft")
                .font(.headline).foregroundStyle(DocumentStyle.accent)
            Text(model.notice).font(.body.weight(.medium)).accessibilityFocused($resultFocused)
            if !model.lines.isEmpty {
                Button {
                    model.readLines(model.lines)
                } label: { Label("전체 원문 듣기", systemImage: "speaker.wave.2.fill") }
                    .buttonStyle(DocumentButtonStyle())
                Button { model.openQuestions() } label: {
                    Label("문서에 질문하기", systemImage: "bubble.left.and.bubble.right.fill")
                }.buttonStyle(DocumentButtonStyle(prominent: false))
                    .accessibilityHint("읽은 문서 또는 관련 개인정보 사례를 말로 물어봅니다.")
                Button("읽기 멈추기") { model.stopSpeech() }
                    .frame(minHeight: 44).frame(maxWidth: .infinity)
            }
            if model.busy {
                HStack {
                    ProgressView().accessibilityHidden(true)
                    Spacer()
                    Button("확인 취소") { model.cancelReview() }.frame(minHeight: 44)
                }
            }
        }
        .padding(20).frame(maxWidth: .infinity, alignment: .leading).background(DocumentStyle.wash, in: RoundedRectangle(cornerRadius: 24))
        if !model.lines.isEmpty {
            VStack(alignment: .leading, spacing: 0) {
                Text("한 줄씩 듣기").font(.headline).padding(.bottom, 10).accessibilityAddTraits(.isHeader)
                ForEach(model.lines) { line in
                    Divider()
                    Button {
                        model.readLines([line])
                    } label: {
                        HStack(alignment: .firstTextBaseline, spacing: 10) {
                            VStack(alignment: .leading, spacing: 4) {
                                if line.uncertain { Text("인식 불확실").font(.caption.bold()).foregroundStyle(DocumentStyle.clay) }
                                Text(line.text).font(.body).multilineTextAlignment(.leading)
                            }
                            Spacer()
                            Image(systemName: "play.circle").foregroundStyle(DocumentStyle.accent).accessibilityHidden(true)
                        }.frame(minHeight: 44).padding(.vertical, 5).contentShape(Rectangle())
                    }.buttonStyle(.plain).accessibilityLabel(line.spoken)
                }
            }.padding(16).documentCard()
        }
        if !model.busy {
            Button { startCamera() } label: { Label("다시 촬영", systemImage: "camera") }.buttonStyle(DocumentButtonStyle(prominent: false))
            photoPicker
            Button(model.lines.isEmpty ? "홈으로 돌아가기" : "문서 지우고 홈으로") { model.clear() }.frame(minHeight: 44).frame(maxWidth: .infinity)
                .accessibilityHint("현재 문서를 지우고 촬영 홈으로 돌아갑니다.")
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
                }.padding(20).documentCard().padding(20)
            }
            .background(DocumentStyle.canvas).foregroundStyle(DocumentStyle.ink)
            .navigationTitle("소리 설정").navigationBarTitleDisplayMode(.inline)
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
