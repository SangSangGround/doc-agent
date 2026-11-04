import SwiftUI
import PhotosUI
import DocAgentCore
import Darwin
import UniformTypeIdentifiers

@main
@MainActor
struct DocAgentApp: App {
    @StateObject private var model = HomeModel()
    @Environment(\.scenePhase) private var scenePhase
    var body: some Scene {
        WindowGroup {
            HomeView(model: model)
                .onChange(of: scenePhase) { _, phase in
                    if phase == .background { model.suspend() }
                }
        }
    }
}

@MainActor
struct HomeView: View {
    @ObservedObject var model: HomeModel
    @State private var photo: PhotosPickerItem?
    @State private var importingConnection = false
    private var supported: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        var info = utsname(); uname(&info)
        let identifier = withUnsafePointer(to: &info.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
        return SupportedDevice.allows(identifier)
        #endif
    }
    var body: some View {
        NavigationStack {
            Form {
                if !supported {
                    Section { Text("이 앱은 iPhone 13 이상 기종을 지원합니다. iPhone SE와 iPad는 지원 대상에 포함하지 않습니다.") }
                } else {
                    Section {
                        Text(model.notice).textSelection(.enabled).accessibilityIdentifier("status-message")
                        if model.busy { ProgressView("처리 중") }
                    }
                    if !model.connected && !model.busy { connection }
                    if let state = model.state {
                        Section("현재 문서") {
                            Text(state.title).font(.headline)
                            if state.purpose == "demo" { Text("합성 문서 연습 모드 · 실제 동의서가 아닙니다").foregroundStyle(.secondary) }
                        }
                        Section("1. 문서 촬영") {
                            Text("사진과 촬영 안내용 프레임을 연결한 PC로 전송해 분석합니다.").font(.footnote)
                            Button("카메라로 촬영", systemImage: "camera") { model.beginCamera() }
                            PhotosPicker(selection: $photo, matching: .images, photoLibrary: .shared()) {
                                Label("사진 불러오기", systemImage: "photo")
                            }
                            Toggle("불러올 사진이 페이지 전체 스캔입니다", isOn: $model.scanImage)
                                .accessibilityHint("직접 촬영한 사진에서는 끄세요.")
                        }.disabled(model.busy)
                        Section("2. 원문 듣기와 선택") {
                            Toggle("자동 음성 안내", isOn: $model.automaticSpeech)
                            Button("전체 원문 읽기") { model.command(.read, body: ["scope": "page"]) }.disabled(!state.documentReady)
                            Button("현재 항목 읽기") { model.command(.read, body: ["scope": "current"]) }.disabled(!state.documentReady)
                            Picker("읽기 속도", selection: Binding(get: { state.session.preferences.speechRate }, set: {
                                model.command(.preferences, body: ["speech_rate": $0], message: "읽기 속도를 변경했습니다.")
                            })) {
                                Text("0.8배").tag(0.8); Text("1배").tag(1.0); Text("1.5배").tag(1.5); Text("2배").tag(2.0)
                            }
                            Text("\(state.session.currentIndex + 1) / \(state.session.groups.count) 항목").font(.headline)
                            if state.documentReady {
                                Text(state.session.currentGroup.sourceText).textSelection(.enabled)
                                choices(state.session.currentGroup)
                            } else { Text("촬영한 문서가 지원 양식인지 확인한 뒤 원문을 제공합니다.") }
                            HStack {
                                Button("이전 항목") { model.command(.move, body: ["direction":"previous"], message: "이전 항목으로 이동했습니다.") }
                                Spacer()
                                Button("다음 항목") { model.command(.move, body: ["direction":"next"], message: "다음 항목으로 이동했습니다.") }
                            }.buttonStyle(.bordered)
                        }.disabled(model.busy)
                        Section("3. 체크 결과 확인") {
                            Text("선택을 확정하고 종이에 표시한 뒤 새 사진을 촬영해 주세요.")
                            Button("촬영한 체크 결과 대조") { model.command(.verify) }.disabled(!state.canVerify || model.busy || model.draftChanged)
                            if model.draftChanged { Text("선택 내용을 재확인한 뒤 확정해 주세요.") }
                            else { ForEach(Array(state.session.groups.enumerated()), id: \.element.id) { index, group in
                                Text("\(index + 1)번째 항목: \(group.verification.message)")
                            }
                            if state.session.complete { Label("모든 선택 항목의 표시가 일치합니다.", systemImage: "checkmark.circle") } }
                        }
                        Section("개인정보 가림 후보") {
                            Button("PC에서 마스킹 후보 확인") { model.privacy() }.disabled(!state.hasCapture || model.busy)
                            Text("자동 후보는 완전한 마스킹 승인이 아닙니다. 사진은 이미 연결한 PC에서 처리됩니다.").font(.footnote)
                            if !model.privacyNotice.isEmpty { Text(model.privacyNotice) }
                            if let image = model.maskedImage {
                                Image(uiImage: image).resizable().scaledToFit().accessibilityLabel("개인정보 후보를 가린 검토용 이미지")
                            }
                            Button("촬영 이미지 지우기", role: .destructive) { model.clearImage() }.disabled(model.busy)
                        }
                        Section("PC 연결") {
                            Text(model.endpoint).font(.footnote)
                            Button("연결 종료") { model.disconnect() }.disabled(model.busy)
                        }
                    }
                }
            }
            .navigationTitle("문서 도우미")
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("읽기 정지") { model.stopSpeech() } } }
            .sheet(isPresented: $model.cameraPresented) {
                CameraSheet(model: model)
            }
            .fileImporter(isPresented: $importingConnection, allowedContentTypes: [.json]) { result in
                do {
                    let url = try result.get()
                    let scoped = url.startAccessingSecurityScopedResource()
                    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                    let data = try Data(contentsOf: url)
                    guard data.count < 8192, let text = String(data: data, encoding: .utf8) else { throw ConnectionError.invalidPairing }
                    _ = try ConnectionSettings.parse(text)
                    model.connectionText = text
                    model.notice = "연결 정보를 불러왔습니다. 전송할 PC를 확인하고 연결해 주세요."
                } catch { model.notice = "연결 정보 파일을 읽지 못했습니다. PC가 만든 pairing.json을 선택해 주세요." }
            }
            .onChange(of: photo) { _, item in
                guard let item, !model.busy else { return }
                model.importPhoto(item, scan: model.scanImage)
                photo = nil
            }
        }.tint(.blue)
    }
    private var connection: some View {
        Section("PC 연결") {
            Text("같은 Wi-Fi의 PC에서 만든 pairing.json 내용을 붙여넣으세요. 연결 정보는 앱에 영구 저장하지 않습니다.")
            Button("연결 정보 파일 불러오기", systemImage: "doc") { importingConnection = true }
            TextEditor(text: $model.connectionText).frame(minHeight: 130)
                .autocorrectionDisabled().textInputAutocapitalization(.never)
                .accessibilityLabel("PC 연결 정보")
            if let settings = try? ConnectionSettings.parse(model.connectionText) {
                Text("전송 대상: " + settings.serverUrl.absoluteString).font(.footnote)
            }
            Toggle("촬영 사진을 이 PC로 보내 분석합니다", isOn: $model.transferAccepted)
            Button("PC와 연결") { model.connect() }
                .disabled(!model.transferAccepted || model.connectionText.isEmpty)
        }
    }
    @ViewBuilder
    private func choices(_ group: DocumentGroup) -> some View {
        let selected = model.draftSelection
        ForEach(group.options) { option in
            Button { model.choose(option) } label: {
                HStack {
                    Image(systemName: selected.contains(option.id) ? "checkmark.circle.fill" : "circle")
                    Text(option.label); Spacer()
                }.frame(minHeight: 44).contentShape(Rectangle())
            }
            .accessibilityLabel(option.label)
            .accessibilityValue(selected.contains(option.id) ? "선택됨" : "선택 안 됨")
        }
        Button("선택 내용 재확인") { model.proposeSelection() }
        if group.pendingOptionIds != nil && !model.draftChanged {
            Text("선택 내용: " + group.options.filter { selected.contains($0.id) }.map(\.label).joined(separator: ", "))
            Button("이 선택으로 확정") {
                model.command(.confirm, body: ["group_id": group.id], message: "선택을 확정했습니다. 작성 후 새로 촬영해 주세요.")
            }.buttonStyle(.borderedProminent)
            Button("선택 변경 취소") { model.command(.cancel, body: ["group_id": group.id], message: "변경 중인 선택을 취소했습니다.") }
        } else if group.confirmed && !model.draftChanged { Text("선택을 확정했습니다.").font(.footnote) }
    }
}
