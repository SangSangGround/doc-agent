import AVFoundation

/// Estimates the end of speech on the device without retaining the recording.
final class QuestionAudioCapture: @unchecked Sendable {
    private static let sampleRate = 16_000
    private static let maximumSamples = 25 * sampleRate
    private static let speechWindow = sampleRate / 50  // 20 ms of audio
    private let lock = NSLock()
    private let converter: AVAudioConverter
    private let started = ContinuousClock.now
    private var output: AVAudioPCMBuffer?
    private var sampleCount = 0
    private var failed = false
    private var speechDetected = false
    private var lastVoicedEnd = 0
    private var recentLevels: [Float] = []
    private var windowSamples = 0
    private var windowEnergy: Float = 0
    private var analyzedSamples = 0
    private var noiseLevel: Float = 0.001

    init(format: AVAudioFormat) throws {
        guard format.channelCount > 0, format.sampleRate.isFinite, format.sampleRate > 0,
              let mono = AVAudioFormat(standardFormatWithSampleRate: Double(Self.sampleRate), channels: 1),
              let converter = AVAudioConverter(from: format, to: mono) else {
            throw CocoaError(.coderInvalidValue)
        }
        self.converter = converter
        converter.primeMethod = .none
        converter.downmix = true
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard !failed, sampleCount < Self.maximumSamples, buffer.frameLength > 0 else { return }
        guard buffer.format == converter.inputFormat else { failed = true; return }
        let expected = Double(buffer.frameLength) * Double(Self.sampleRate) / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(min(Double(Self.maximumSamples), ceil(expected) + 64))
        if (output?.frameCapacity ?? 0) < capacity {
            output = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: capacity)
        }
        guard let output else { failed = true; return }
        output.frameLength = 0
        var supplied = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, inputStatus in
            if supplied {
                inputStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            inputStatus.pointee = .haveData
            return buffer
        }
        guard status != .error, error == nil, let data = output.floatChannelData?[0] else {
            failed = true
            return
        }
        let count = min(Int(output.frameLength), Self.maximumSamples - sampleCount)
        let samples = UnsafeBufferPointer(start: data, count: count)
        sampleCount += count
        for sample in samples {
            windowEnergy += sample * sample
            windowSamples += 1
            if windowSamples == Self.speechWindow { inspectWindow() }
        }
    }

    // Require an envelope change across several audible windows before starting the end-of-speech timer.
    // A continuous motor hum can exceed the energy threshold for the entire recording.
    private func inspectWindow() {
        analyzedSamples += windowSamples
        let level = sqrt(windowEnergy / Float(windowSamples))
        let threshold = max(0.0025, 2.5 * noiseLevel)
        if level > threshold {
            if recentLevels.count == 8 { recentLevels.removeFirst() }
            recentLevels.append(level)
            if recentLevels.count >= 3, let low = recentLevels.min(),
               let high = recentLevels.max(), high >= low * 1.45 {
                speechDetected = true
                lastVoicedEnd = analyzedSamples
            }
        } else {
            recentLevels.removeAll(keepingCapacity: true)
            noiseLevel += (level - noiseLevel) * 0.05
        }
        windowEnergy = 0
        windowSamples = 0
    }

    func shouldFinish() -> Bool {
        lock.withLock {
            let elapsed = started.duration(to: .now)
            if failed || sampleCount >= Self.maximumSamples || elapsed >= .seconds(25) { return true }
            if speechDetected { return sampleCount - lastVoicedEnd >= Self.sampleRate * 18 / 10 }
            return sampleCount >= Self.sampleRate * 10 || elapsed >= .seconds(10)
        }
    }
}
