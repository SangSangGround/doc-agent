import Foundation
import XCTest
@testable import QuestionSpeech

final class QuestionWhisperTests: XCTestCase {
    func testContinuousMechanicalNoiseDoesNotBecomeAQuestion() async throws {
        guard QuestionWhisper.voiceDetector != nil, QuestionWhisper.weights != nil else {
            throw XCTSkip("Prepare bundled speech models with ios/prepare_review_model.py.")
        }
        let rate = 16_000
        var seed: UInt32 = 19
        var samples = [Float]()
        for i in 0..<(2 * rate) {
            seed = seed &* 1_664_525 &+ 1_013_904_223
            let noise = Float(seed) / Float(UInt32.max) * 2 - 1
            let hum = sin(2 * Double.pi * 120 * Double(i) / Double(rate))
            samples.append(0.012 * Float(hum) + 0.004 * noise)
        }
        do {
            let transcript = try await QuestionWhisper.transcribe(samples)
            XCTFail("Fan noise produced a question: \(transcript)")
        } catch TranscriptionError.noSpeech {
            // The recording contained sound but no human voice.
        }
    }
}
