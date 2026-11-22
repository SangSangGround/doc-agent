import ImageIO
import XCTest
@testable import DocAgentCore

final class WritingGuidanceTests: XCTestCase {
    private let yes = WritingTarget(id: "yes", label: "동의함", kind: .checkbox,
                                    bounds: CGRect(x: 0.62, y: 0.72, width: 0.014, height: 0.010))
    private let no = WritingTarget(id: "no", label: "동의하지 않음", kind: .checkbox,
                                   bounds: CGRect(x: 0.64, y: 0.72, width: 0.014, height: 0.010))
    private let signature = WritingTarget(id: "sign", label: "서명", kind: .signature,
                                          bounds: CGRect(x: 0.68, y: 0.82, width: 0.19, height: 0.06))
    private let identity = WritingTransform(rowMajor: [1,0,0, 0,1,0, 0,0,1])!

    func testArrivalAndAdjacentRejectionUnderThreeCameraPoses() throws {
        let poses = [identity,
                     WritingTransform(rowMajor: [0.82,-0.05,0.1, 0.12,0.86,0.01, 0.03,-0.05,1])!,
                     WritingTransform(rowMajor: [0,-0.8,0.92, 0.92,0,0.03, 0,0,1])!]
        let all = [yes, no, signature]
        var accepted = 0, rejected = 0
        for pose in poses {
            for selected in all {
                for pointed in all {
                    let b = pointed.bounds
                    let tip = try XCTUnwrap(pose.livePoint(.init(x: b.midX, y: b.midY)))
                    var guidance = WritingGuidance()
                    var outcome = WritingGuidance.seeking
                    for t in 0...11 {
                        outcome = guidance.update(pose: pose, fingertip: tip, target: selected, fields: all, time: Double(t)*0.1)
                        if t <= 6 { XCTAssertFalse(outcome.arrived) }
                    }
                    if selected.id == pointed.id {
                        XCTAssertTrue(outcome.arrived); accepted += 1
                    } else {
                        XCTAssertFalse(outcome.arrived); rejected += 1
                    }
                }
            }
        }
        XCTAssertEqual(accepted, 9)
        XCTAssertEqual(rejected, 18)
    }

    func testOverlapAndLostPoseOrHandCancelArrival() {
        var guide = WritingGuidance()
        let tip = CGPoint(x: yes.bounds.midX, y: yes.bounds.midY)
        let overlap = WritingTarget(id: "other", label: "다른 칸", kind: .input, bounds: yes.bounds)
        for t in 0...12 {
            XCTAssertFalse(guide.update(pose: identity, fingertip: tip, target: yes, fields: [yes, overlap], time: Double(t)*0.1).arrived)
        }
        for t in 13...17 {
            _ = guide.update(pose: identity, fingertip: tip, target: yes, fields: [yes], time: Double(t)*0.1)
        }
        XCTAssertNil(guide.update(pose: nil, fingertip: tip, target: yes, fields: [yes], time: 1.8).beepInterval)
        XCTAssertFalse(guide.update(pose: identity, fingertip: tip, target: yes, fields: [yes], time: 1.9).arrived)
        for t in 20...24 {
            _ = guide.update(pose: identity, fingertip: tip, target: yes, fields: [yes], time: Double(t)*0.1)
        }
        XCTAssertNil(guide.update(pose: identity, fingertip: nil, target: yes, fields: [yes], time: 2.5).beepInterval)
        XCTAssertFalse(guide.update(pose: identity, fingertip: tip, target: yes, fields: [yes], time: 2.6).arrived)
        XCTAssertFalse(guide.update(pose: identity, fingertip: tip, target: yes, fields: [yes], time: 8.0).arrived)
    }

    func testOffscreenGuidanceStopsAndApproachIncreasesCadence() throws {
        let off = WritingTransform(rowMajor: [1,0,0.5, 0,1,0, 0,0,1])!
        var guide = WritingGuidance()
        for t in 0...8 {
            let f = guide.update(pose: off, fingertip: .init(x: 0.7,y: 0.8), target: yes, fields: [yes], time: Double(t)*0.15)
            XCTAssertFalse(f.arrived); XCTAssertNil(f.beepInterval)
        }
        guide = WritingGuidance()
        _ = guide.update(pose: identity, fingertip: nil, target: yes, fields: [yes], time: 0)
        let far = guide.update(pose: identity, fingertip: .init(x: 0.32,y: yes.bounds.midY), target: yes, fields: [yes], time: 0.1)
        let near = guide.update(pose: identity, fingertip: .init(x: 0.60,y: yes.bounds.midY), target: yes, fields: [yes], time: 0.2)
        XCTAssertGreaterThan(try XCTUnwrap(far.beepInterval), try XCTUnwrap(near.beepInterval))
        XCTAssertTrue(near.message.contains("오른쪽"))
    }

    func testPaperDirectionWithRotatedCameraAndHandReacquisition() throws {
        let rotated = try XCTUnwrap(WritingTransform(rowMajor: [0,-1,1, 1,0,0, 0,0,1]))
        let center = CGPoint(x: yes.bounds.midX, y: yes.bounds.midY)
        let fromUpperLeft = try XCTUnwrap(rotated.livePoint(CGPoint(x: center.x - 0.15, y: center.y - 0.10)))
        var guide = WritingGuidance()
        func observe(_ tip: CGPoint?, _ t: Double) -> WritingFeedback {
            guide.update(pose: rotated, fingertip: tip, target: yes, fields: [yes, no], time: t)
        }
        _ = observe(nil, 0)
        let diagonal = observe(fromUpperLeft, 0.1)
        XCTAssertTrue(diagonal.message.contains("오른쪽 아래 대각선"))
        XCTAssertNotNil(diagonal.repeatSpeechAfter)
        _ = observe(nil, 0.2)
        let fromRight = try XCTUnwrap(rotated.livePoint(CGPoint(x: center.x + 0.15, y: center.y)))
        XCTAssertTrue(observe(fromRight, 0.3).message.contains("왼쪽"))
    }

    func testProximityReportsNearTargetButNeverOnAnotherBox() {
        let center = CGPoint(x: yes.bounds.midX, y: yes.bounds.midY)
        let nearby = CGPoint(x: center.x - 6.0/210, y: center.y)
        let neighbor = WritingTarget(id: "adjacent", label: "거부", kind: .checkbox,
                                    bounds: CGRect(x: nearby.x - 0.003, y: nearby.y - 0.003,
                                                   width: 0.006, height: 0.006))
        for protectNeighbor in [false, true] {
            var guide = WritingGuidance()
            var output = WritingGuidance.seeking
            for i in 0...10 {
                output = guide.update(pose: identity, fingertip: nearby, target: yes,
                                      fields: protectNeighbor ? [yes, neighbor] : [yes], time: Double(i)/10)
            }
            XCTAssertEqual(output.arrived, !protectNeighbor)
            if output.arrived { XCTAssertTrue(output.message.contains("근처")) }
        }
    }

    /// Optional replay of private reference + labelled transforms generated by scripts/evaluate_writing_tracker.py.
    func testRegistrationReplay() throws {
        guard let path = ProcessInfo.processInfo.environment["DOCAGENT_WRITING_REPLAY"] else {
            throw XCTSkip("To replay private photos, run scripts/evaluate_writing_tracker.py")
        }
        struct Frame: Decodable { let file: String; let expected: Bool; let landmarks: [[Double]] }
        struct Manifest: Decodable { let reference: String; let frames: [Frame] }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: directory.appendingPathComponent("frames.json")))
        func image(_ file: String) throws -> CGImage {
            let source = try XCTUnwrap(CGImageSourceCreateWithURL(directory.appendingPathComponent(file) as CFURL,nil))
            return try XCTUnwrap(CGImageSourceCreateImageAtIndex(source,0,nil))
        }
        let tracker = try XCTUnwrap(WritingPageRegistration(reference: image(manifest.reference)))
        var rows: [[String: Any]] = []
        for frame in manifest.frames {
            let input = try image(frame.file)
            let start = ProcessInfo.processInfo.systemUptime
            let pose = tracker.register(input)
            let elapsed = (ProcessInfo.processInfo.systemUptime - start)*1000
            let errors = frame.landmarks.compactMap { r -> Double? in
                guard let p = pose?.livePoint(.init(x: r[0],y:r[1])) else { return nil }
                return hypot((p.x-r[2])*Double(input.width), (p.y-r[3])*Double(input.height))
            }
            rows.append(["file": frame.file, "expected": frame.expected, "registered": pose != nil,
                         "milliseconds": elapsed, "errors_px": errors])
        }
        let data = try JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: directory.appendingPathComponent("registration-results.json"))
        // A metric is still recorded if any case fails, so the report can identify regressions.
        XCTAssertTrue(rows.allSatisfy { ($0["registered"] as? Bool) == ($0["expected"] as? Bool) })
    }
}
