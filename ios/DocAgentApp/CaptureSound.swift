import AVFoundation
import DocAgentCore
import SwiftUI

/// Short PCM cues mixed with the system's accessibility audio.
final class CaptureSound: ObservableObject {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
    private var tones: [CameraCue: AVAudioPCMBuffer] = [:]

    init() {
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        var pitches: [(CameraCue, Int, Double)] = [
            (.searching, 355, 0.045), (.entered, 1150, 0.29), (.aligned, 830, 0.045), (.ready, 1450, 0.12)
        ]
        for (level, pitch) in CameraCue.guidePitches.enumerated() {
            pitches.append((.guiding(level), pitch, 0.070))
        }
        for (cue, hz, seconds) in pitches {
            let length = AVAudioFrameCount(format.sampleRate * seconds)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: length),
                  let samples = buffer.floatChannelData?[0] else { continue }
            buffer.frameLength = length
            for i in 0..<Int(length) {
                let envelope = min(1.0, Double(min(i, Int(length)-1-i)) / 300)
                samples[i] = Float(envelope * sin(Double(i * hz) * 2 * .pi / format.sampleRate))
            }
            tones[cue] = buffer
        }
    }
    func start() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
        try session.setActive(true)
        if !engine.isRunning { try engine.start() }
    }
    func play(_ cue: CameraCue, quietly: Bool) {
        guard engine.isRunning, let buffer = tones[cue] else { return }
        player.volume = quietly ? 0.10 : 0.32
        player.scheduleBuffer(buffer, at: nil, options: .interrupts)
        if !player.isPlaying { player.play() }
    }
    func stop() {
        player.stop()
        engine.stop()
    }
}
