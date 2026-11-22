import Foundation
// Keep whisper's GGML headers private. The document model uses another GGML version in llama.
@_implementationOnly import whisper

public enum QuestionWhisper {
    static let voiceDetector = Bundle.module.url(forResource: "ggml-silero-v6.2.0", withExtension: "bin",
                                                  subdirectory: "SpeechModel")
    static let weights = Bundle.module.url(forResource: "ggml-base", withExtension: "bin",
                                            subdirectory: "SpeechModel")

    /// Transcribes a recorded question locally. Samples must be 16 kHz mono PCM normalized to ±1.
    public static func transcribe(_ samples: [Float]) async throws -> String {
        let cancellation = InferenceCancellation()
        return try await withTaskCancellationHandler {
            try await WhisperDecoder.shared.transcribe(samples, cancellation: cancellation)
        } onCancel: {
            cancellation.cancel()
        }
    }
}

enum TranscriptionError: LocalizedError {
    case modelUnavailable, unreadableAudio, noSpeech, inferenceFailed

    var errorDescription: String? {
        switch self {
        case .modelUnavailable: return "기기에서 Whisper 음성 모델을 열 수 없어요."
        case .unreadableAudio: return "녹음한 소리를 읽을 수 없어요. 다시 말씀해 주세요."
        case .noSpeech: return "질문을 알아듣지 못했어요. 다시 말씀해 주세요."
        case .inferenceFailed: return "Whisper가 질문을 받아쓰지 못했어요. 다시 시도해 주세요."
        }
    }
}

// whisper.cpp can query this flag from a native worker thread.
private final class InferenceCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    var isCancelled: Bool { lock.withLock { cancelled } }

    func cancel() { lock.withLock { cancelled = true } }
}

private actor WhisperDecoder {
    static let shared = WhisperDecoder()
    private static let disableLogs: Void = whisper_log_set({ _, _, _ in }, nil)

    func transcribe(_ samples: [Float], cancellation: InferenceCancellation) throws -> String {
        try Task.checkCancellation()
        guard let count = Int32(exactly: samples.count), samples.allSatisfy(\.isFinite) else {
            throw TranscriptionError.unreadableAudio
        }
        guard count > 0, samples.contains(where: { abs($0) > 0.0001 }) else {
            throw TranscriptionError.noSpeech
        }
        _ = Self.disableLogs
        try detectSpeech(in: samples, count: count)
        try Task.checkCancellation()
        guard let file = QuestionWhisper.weights else { throw TranscriptionError.modelUnavailable }
        var settings = whisper_context_default_params()
        settings.use_gpu = false   // The GGML CPU backend polls the abort callback during inference.
        guard let context = whisper_init_from_file_with_params(file.path, settings) else {
            throw TranscriptionError.modelUnavailable
        }
        defer { whisper_free(context) }  // Release speech weights before Qwen answers the question.
        try Task.checkCancellation()
        var options = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        options.translate = false
        options.no_context = true
        options.no_timestamps = true
        options.print_progress = false
        options.print_realtime = false
        options.print_timestamps = false
        options.print_special = false
        options.suppress_nst = true
        options.n_threads = Int32(max(1, min(4, ProcessInfo.processInfo.activeProcessorCount - 1)))
        options.greedy.best_of = 1
        options.temperature_inc = 0
        options.abort_callback_user_data = Unmanaged.passUnretained(cancellation).toOpaque()
        options.abort_callback = { data in
            guard let data else { return false }
            return Unmanaged<InferenceCancellation>.fromOpaque(data).takeUnretainedValue().isCancelled
        }

        let status = withExtendedLifetime(cancellation) {
            "ko".withCString { language in
                options.language = language
                return samples.withUnsafeBufferPointer {
                    whisper_full(context, options, $0.baseAddress, count)
                }
            }
        }
        try Task.checkCancellation()
        guard status == 0 else { throw TranscriptionError.inferenceFailed }
        var segments: [String] = []
        for index in 0..<whisper_full_n_segments(context) {
            guard whisper_full_get_segment_no_speech_prob(context, index) < 0.8,
                  let text = whisper_full_get_segment_text(context, index) else { continue }
            let segment = String(cString: text).trimmingCharacters(in: .whitespacesAndNewlines)
            if !segment.isEmpty { segments.append(segment) }
        }
        let result = segments.joined(separator: " ")
        guard result.contains(where: { $0.isLetter || $0.isNumber }) else {
            throw TranscriptionError.noSpeech
        }
        return result
    }

    // Whisper can confidently invent words for constant fan noise. Check for a voice first,
    // then give Whisper the original recording so short syllables at either end are preserved.
    private func detectSpeech(in samples: [Float], count: Int32) throws {
        guard let file = QuestionWhisper.voiceDetector else { throw TranscriptionError.modelUnavailable }
        var settings = whisper_vad_default_context_params()
        settings.use_gpu = false
        settings.n_threads = 1
        guard let context = whisper_vad_init_from_file_with_params(file.path, settings) else {
            throw TranscriptionError.modelUnavailable
        }
        defer { whisper_vad_free(context) }
        var options = whisper_vad_default_params()
        options.min_speech_duration_ms = 96 // Three Silero windows allow short replies such as “네”.
        let segments = samples.withUnsafeBufferPointer {
            whisper_vad_segments_from_samples(context, options, $0.baseAddress, count)
        }
        guard let segments else { throw TranscriptionError.inferenceFailed }
        defer { whisper_vad_free_segments(segments) }
        guard whisper_vad_segments_n_segments(segments) > 0 else { throw TranscriptionError.noSpeech }
    }
}
