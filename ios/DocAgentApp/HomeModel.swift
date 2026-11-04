import SwiftUI
import AVFoundation
import UIKit
import DocAgentCore
import PhotosUI
import ImageIO

@MainActor
final class HomeModel: ObservableObject {
    @Published var state: HomeState? {
        didSet {
            let group = state?.session.currentGroup
            draftSelection = Set(group?.pendingOptionIds ?? group?.selectedOptionIds ?? [])
            draftChanged = false
        }
    }
    @Published var draftSelection: Set<String> = []
    @Published var draftChanged = false
    @Published var busy = false
    @Published var notice = "PC 서버의 연결 정보를 입력해 주세요."
    @Published var connectionText = ""
    @Published var transferAccepted = false
    @Published var cameraPresented = false
    @Published var cameraNotice = "문서 네 모서리가 보이도록 맞춰 주세요."
    @Published var readyToCapture = false
    @Published var maskedImage: UIImage?
    @Published var privacyNotice = ""
    @Published var scanImage = false
    @Published var automaticSpeech = true
    private var client: APIClient?
    private let synthesizer = AVSpeechSynthesizer()
    private var revision = 0
    private var frameBusy = false
    private var task: Task<Void, Never>?

    var connected: Bool { client != nil && state != nil }
    var endpoint: String { client?.settings.serverUrl.absoluteString ?? "" }
    var current: DocumentGroup? { state?.session.currentGroup }

    func speak(_ text: String, force: Bool = false) {
        guard automaticSpeech || force else { return }
        synthesizer.stopSpeaking(at: .immediate)
        if UIAccessibility.isVoiceOverRunning && !force {
            UIAccessibility.post(notification: .announcement, argument: text)
            return
        }
        guard let voice = AVSpeechSynthesisVoice(language: "ko-KR") else { return }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        utterance.rate = min(AVSpeechUtteranceMaximumSpeechRate,
                             AVSpeechUtteranceDefaultSpeechRate * Float(state?.session.preferences.speechRate ?? 1))
        synthesizer.speak(utterance)
    }
    func stopSpeech() { synthesizer.stopSpeaking(at: .immediate) }
    private func announce(_ text: String, force: Bool = false) {
        notice = text; speak(text, force: force)
    }
    private func operation(_ work: @escaping () async throws -> Void) {
        guard !busy else { return }
        busy = true; revision += 1; notice = "처리하고 있습니다."; stopSpeech()
        let generation = revision
        task = Task {
            do { try await work() }
            catch {
                guard generation == revision else { return }
                // 연결이나 변경 요청 실패 후에는 이전 완료 결과를 계속 보여주지 않는다.
                state = nil; maskedImage = nil; privacyNotice = ""
                announce("처리하지 못했습니다. 같은 Wi-Fi와 PC 서버를 확인한 뒤 다시 연결해 주세요. \(error.localizedDescription)")
                client?.close(); client = nil
            }
            if generation == revision { busy = false; task = nil }
        }
    }
    func connect() {
        guard transferAccepted else { return }
        operation {
            let settings = try ConnectionSettings.parse(self.connectionText)
            self.client?.close()
            let client = APIClient(settings: settings)
            self.client = client
            // 서버의 이전 사진을 재사용하지 않는다. 선택 ID는 서버 설정에 따라 유지된다.
            let value: HomeState = try await client.request(.resetCapture)
            try Task.checkCancellation()
            self.state = value; self.connectionText = ""
            self.announce("PC와 연결했습니다. 문서를 촬영하거나 사진을 선택해 주세요.")
        }
    }
    func suspend() {
        revision += 1; task?.cancel(); task = nil; busy = false
        cameraPresented = false; transferAccepted = false; stopSpeech()
        state = nil; maskedImage = nil; privacyNotice = ""
        client?.close(); client = nil
        notice = "연결을 종료했습니다. PC 연결 정보를 다시 입력하면 이어갈 수 있습니다."
    }
    func disconnect() {
        guard let client else { suspend(); return }
        operation {
            let _: HomeState = try await client.request(.resetCapture)
            self.suspend()
        }
    }
    func command(_ action: HomeAction, body: [String: Any] = [:], message: String? = nil) {
        guard let client, !(action == .verify && draftChanged) else { return }
        operation {
            let value: HomeState = try await client.request(action, body: body)
            try Task.checkCancellation()
            let preserveDraft = self.draftChanged && (action == .read || action == .preferences)
            let draft = self.draftSelection
            self.state = value
            if preserveDraft { self.draftSelection = draft; self.draftChanged = true }
            if action == .verify {
                let messages = value.session.groups.enumerated().map { "\($0.offset + 1)번째 항목. \($0.element.verification.message)" }
                self.announce(messages.joined(separator: "\n") + (value.session.complete ? "\n모든 선택 항목의 표시가 일치합니다." : ""))
            } else {
                self.announce(value.instruction ?? message ?? "처리했습니다.", force: action == .read)
            }
        }
    }
    func choose(_ option: DocumentOption) {
        guard !busy, let group = current else { return }
        if draftSelection.contains(option.id) {
            if group.minSelected == 0 || group.maxSelected > 1 { draftSelection.remove(option.id) }
        } else if group.maxSelected == 1 { draftSelection = [option.id] }
        else { draftSelection.insert(option.id) }
        draftChanged = true
        notice = "선택을 변경하고 있습니다. 선택 내용 재확인을 눌러 주세요."
    }
    func proposeSelection() {
        guard let group = current else { return }
        guard draftSelection.count >= group.minSelected, draftSelection.count <= group.maxSelected else {
            announce("이 항목은 \(group.minSelected)개에서 \(group.maxSelected)개까지 선택할 수 있습니다."); return
        }
        let labels = group.options.filter { draftSelection.contains($0.id) }.map(\.label).joined(separator: ", ")
        command(.select, body: ["group_id": group.id, "option_ids": draftSelection.sorted()],
                message: "선택 내용: \(labels.isEmpty ? "선택 없음" : labels). 맞으면 이 선택으로 확정을 눌러 주세요.")
    }
    func importPhoto(_ item: PhotosPickerItem, scan: Bool) {
        guard let client else { return }
        operation {
            self.state = nil; self.maskedImage = nil; self.privacyNotice = ""
            let _: HomeState = try await client.request(.resetCapture)
            guard let data = try await item.loadTransferable(type: Data.self) else {
                throw ConnectionError.server("사진을 읽지 못했습니다.")
            }
            try Task.checkCancellation()
            try await self.processPhoto(data, scan: scan, client: client)
        }
    }
    func beginCamera() {
        guard let client else { return }
        operation {
            self.maskedImage = nil; self.privacyNotice = ""; self.state = nil
            let value: HomeState = try await client.request(.resetCapture)
            try Task.checkCancellation(); self.state = value
            self.readyToCapture = false; self.cameraNotice = "문서 네 모서리가 보이도록 맞춰 주세요."
            self.cameraPresented = true
        }
    }
    func frame(_ data: Data) {
        guard cameraPresented, !busy, !frameBusy, let client else { return }
        frameBusy = true; let generation = revision
        Task {
            defer { frameBusy = false }
            do {
                let result: FrameResult = try await client.request(.frame, body: ["image": data.base64EncodedString()])
                guard generation == revision, cameraPresented else { return }
                cameraNotice = result.instruction; readyToCapture = result.readyToCapture
                if result.shouldAnnounce { speak(result.instruction) }
            } catch {
                guard generation == revision, cameraPresented else { return }
                readyToCapture = false; cameraNotice = "촬영 안내 연결이 끊겼습니다. PC 서버를 확인해 주세요."
            }
        }
    }
    func capture(_ data: Data, scan: Bool) {
        guard let client, !busy else { return }
        cameraPresented = false
        // 결과와 미리보기를 먼저 없애므로 사진 디코딩 실패도 과거 완료를 유지하지 않는다.
        operation {
            self.state = nil; self.maskedImage = nil; self.privacyNotice = ""
            let _: HomeState = try await client.request(.resetCapture)
            try Task.checkCancellation()
            try await self.processPhoto(data, scan: scan, client: client)
        }
    }
    private func processPhoto(_ data: Data, scan: Bool, client: APIClient) async throws {
        guard let jpeg = Self.normalizedJPEG(data) else {
            throw ConnectionError.server("사진을 읽지 못했습니다.")
        }
        let value: HomeState = try await client.request(.capture, body: ["image": jpeg.base64EncodedString(), "mode": scan ? "scan" : "photo"])
        try Task.checkCancellation(); state = value
        announce(value.instruction ?? "문서를 확인했습니다.")
    }
    func privacy() {
        guard let client else { return }
        operation {
            self.maskedImage = nil; self.privacyNotice = ""
            let result: PrivacyResult = try await client.request(.privacy)
            try Task.checkCancellation()
            if let preview = result.preview, let encoded = preview.split(separator: ",", maxSplits: 1).last,
               let data = Data(base64Encoded: String(encoded)) { self.maskedImage = UIImage(data: data) }
            self.privacyNotice = result.instruction; self.announce(result.instruction)
        }
    }
    func clearImage() {
        maskedImage = nil; privacyNotice = ""
        command(.resetCapture, message: "PC에 보낸 촬영 이미지를 지웠습니다. 다시 촬영해 주세요.")
    }
    static func normalizedJPEG(_ data: Data) -> Data? {
        guard data.count <= 32 * 1024 * 1024,
              let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary) else { return nil }
        let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                      kCGImageSourceCreateThumbnailWithTransform: true,
                                      kCGImageSourceThumbnailMaxPixelSize: 1800]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        return UIImage(cgImage: image).jpegData(compressionQuality: 0.92)
    }
}
