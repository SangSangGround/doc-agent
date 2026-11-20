import AVFoundation
import Speech
import SwiftUI

/// A single, optional Korean dictation turn. No audio file or server fallback.
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
    private var onQuestion: ((String) -> Void)?

    func begin(onQuestion: @escaping (String) -> Void) {
        cancel()
        transcript = ""
        error = nil
        guard let recognizer, recognizer.supportsOnDeviceRecognition else {
            error = "이 기기에서 한국어 오프라인 음성 인식을 사용할 수 없어요. 질문을 입력하거나 예시 질문을 선택해 주세요."
            return
        }
        self.onQuestion = onQuestion
        phase = .preparing
        let ticket = generation
        Task { [self] in
            let speech = await withCheckedContinuation { done in
                SFSpeechRecognizer.requestAuthorization { done.resume(returning: $0) }
            }
            guard ticket == generation else { return }
            guard speech == .authorized else {
                fail("음성 인식 권한을 확인해 주세요. 질문을 입력해도 돼요.")
                return
            }
            let mic = await AVAudioApplication.requestRecordPermission()
            guard ticket == generation else { return }
            guard mic else {
                fail("마이크 권한을 확인해 주세요. 질문을 입력해도 돼요.")
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
                let audio = SFSpeechAudioBufferRecognitionRequest()
                audio.requiresOnDeviceRecognition = true
                audio.shouldReportPartialResults = true
                request = audio
                input.installTap(onBus: 0, bufferSize: 1024, format: format, block: Self.tap(for: audio))
                hasTap = true
                recognition = recognizer.recognitionTask(with: audio) { [weak self] result, error in
                    let text = result?.bestTranscription.formattedString
                    let final = result?.isFinal == true
                    let failed = error != nil
                    Task { @MainActor [weak self] in
                        guard let self, ticket == self.generation else { return }
                        if let text, !text.isEmpty, self.transcript != text {
                            self.transcript = text
                            if self.phase == .listening { self.schedule(seconds: 2.4) { self.finish() } }
                        }
                        if final || (failed && self.phase == .finishing) { self.deliver() }
                        else if failed { self.fail("음성 인식이 중단됐어요. 다시 말하거나 입력해 주세요.") }
                    }
                }
                engine.prepare()
                try engine.start()
                phase = .listening
                schedule(seconds: 12) { self.finish() }
            } catch { fail("마이크를 시작하지 못했어요. 다시 말하거나 입력해 주세요.") }
        }
    }

    private nonisolated static func tap(for request: SFSpeechAudioBufferRecognitionRequest) -> AVAudioNodeTapBlock {
        { buffer, _ in request.append(buffer) }
    }

    func finish() {
        guard phase == .listening else { return }
        phase = .finishing
        stopInput()
        request?.endAudio()
        request = nil
        schedule(seconds: 1.5) { self.deliver() }
    }

    private func schedule(seconds: Double, _ action: @escaping @MainActor () -> Void) {
        timer?.cancel()
        timer = Task {
            do { try await Task.sleep(for: .seconds(seconds)) } catch { return }
            action()
        }
    }

    private func deliver() {
        let question = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        let callback = onQuestion
        cancel()
        if question.isEmpty { error = "질문을 듣지 못했어요. 다시 말해 주세요." }
        else { callback?(question) }
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
        onQuestion = nil
        transcript = ""
        phase = .idle
    }
}
