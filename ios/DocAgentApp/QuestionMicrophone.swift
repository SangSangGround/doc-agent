import AVFoundation
import Speech
import SwiftUI

/// Apple Speech produces a draft for confirmation. The microphone has no network client.
@MainActor
final class QuestionMicrophone: ObservableObject {
    enum Phase { case idle, preparing, listening, finishing }
    @Published private(set) var phase = Phase.idle
    @Published private(set) var transcript = ""
    @Published private(set) var error: String?
    private let engine = AVAudioEngine()
    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "ko-KR"))
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var recognition: SFSpeechRecognitionTask?
    private var timer: Task<Void, Never>?
    private var generation = 0
    private var hasTap = false
    private var activated = false
    private var onDraft: ((String) -> Void)?
    private var onSilence: (() -> Void)?

    /// - Parameter onSilence: 지정하면 아무 말도 받아쓰지 못했을 때 오류 안내 대신 이 함수를 부른다.
    func begin(onSilence: (() -> Void)? = nil, onQuestion: @escaping (String) -> Void) {
        cancel()
        error = nil
        guard let recognizer, recognizer.supportsOnDeviceRecognition else {
            fail("이 아이폰에서 한국어 기기 내 받아쓰기를 사용할 수 없어요. 질문을 입력하거나 예시 질문을 선택해 주세요.")
            return
        }
        onDraft = onQuestion
        self.onSilence = onSilence
        phase = .preparing
        let ticket = generation
        Task {
            let authorization = await Self.speechPermission()
            guard ticket == generation else { return }
            guard authorization == .authorized else {
                fail("설정에서 문서 도우미의 음성 인식 권한을 확인해 주세요. 질문을 입력해도 돼요.")
                return
            }
            let allowed = await AVAudioApplication.requestRecordPermission()
            guard ticket == generation else { return }
            guard allowed else {
                fail("설정에서 마이크 권한을 확인해 주세요. 질문을 입력해도 돼요.")
                return
            }
            do {
                let session = AVAudioSession.sharedInstance()
                try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .mixWithOthers])
                try session.setActive(true)
                activated = true
                let input = engine.inputNode
                let format = input.outputFormat(forBus: 0)
                guard format.channelCount > 0, format.sampleRate > 0 else {
                    fail("마이크 입력을 찾지 못했어요. 질문을 입력해 주세요.")
                    return
                }
                let endpoint = try QuestionAudioCapture(format: format)
                let audio = SFSpeechAudioBufferRecognitionRequest()
                audio.requiresOnDeviceRecognition = true
                audio.shouldReportPartialResults = true
                audio.taskHint = .dictation
                audio.contextualStrings = ["개인정보", "첨부서류", "직업훈련 생계비 대부", "제삼자 제공", "보유 기간", "동의서"]
                request = audio
                input.installTap(onBus: 0, bufferSize: 1024, format: format, block: Self.tap(audio, endpoint))
                hasTap = true
                recognition = recognizer.recognitionTask(with: audio, resultHandler: Self.handler(self, ticket))
                engine.prepare()
                try engine.start()
                phase = .listening
                timer = Task {
                    while !Task.isCancelled {
                        do { try await Task.sleep(for: .milliseconds(100)) } catch { return }
                        guard ticket == generation else { return }
                        if endpoint.shouldFinish() { finish(); return }
                    }
                }
            } catch {
                fail("마이크를 시작하지 못했어요. 다시 말하거나 질문을 입력해 주세요.")
            }
        }
    }

    private nonisolated static func speechPermission() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { done in
            SFSpeechRecognizer.requestAuthorization { done.resume(returning: $0) }
        }
    }

    private nonisolated static func tap(_ audio: SFSpeechAudioBufferRecognitionRequest, _ endpoint: QuestionAudioCapture) -> AVAudioNodeTapBlock {
        { buffer, _ in
            audio.append(buffer)
            endpoint.append(buffer)
        }
    }

    private nonisolated static func handler(_ microphone: QuestionMicrophone, _ ticket: Int)
        -> (SFSpeechRecognitionResult?, Error?) -> Void {
        { [weak microphone] result, error in
            let text = result?.bestTranscription.formattedString
            let final = result?.isFinal == true
            let failed = error != nil
            Task { @MainActor in
                guard let microphone, microphone.generation == ticket else { return }
                if let text, !text.isEmpty { microphone.transcript = text }
                if final || (failed && microphone.phase == .finishing) {
                    microphone.deliverDraft()
                } else if failed {
                    microphone.fail("아이폰에서 받아쓰기가 중단됐어요. 다시 말하거나 질문을 입력해 주세요.")
                }
            }
        }
    }

    func finish() {
        guard phase == .listening else { return }
        phase = .finishing
        stopInput()
        request?.endAudio()
        request = nil
        timer?.cancel()
        let ticket = generation
        timer = Task {
            do { try await Task.sleep(for: .seconds(2)) } catch { return }
            guard ticket == generation else { return }
            deliverDraft()
        }
    }

    private func deliverDraft() {
        let text = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        let callback = onDraft
        let silence = onSilence
        cancel()
        if !text.isEmpty { callback?(text) }
        else if let silence { silence() }
        else { error = "질문을 듣지 못했어요. 다시 말하거나 입력해 주세요." }
    }

    private func fail(_ message: String) {
        cancel()
        error = message
    }

    private func stopInput() {
        if engine.isRunning { engine.stop() }
        if hasTap { engine.inputNode.removeTap(onBus: 0); hasTap = false }
    }

    func cancel() {
        generation += 1
        timer?.cancel()
        timer = nil
        stopInput()
        request?.endAudio()
        recognition?.cancel()
        recognition = nil
        request = nil
        if activated {
            let session = AVAudioSession.sharedInstance()
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            try? session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
            activated = false
        }
        onDraft = nil
        onSilence = nil
        transcript = ""
        phase = .idle
    }
}
