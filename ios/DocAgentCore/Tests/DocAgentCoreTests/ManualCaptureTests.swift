import XCTest
@testable import DocAgentCore

final class ManualCaptureTests: XCTestCase {
    let corners = [CGPoint(x: 0.2, y: 0.8), CGPoint(x: 0.8, y: 0.8),
                   CGPoint(x: 0.8, y: 0.2), CGPoint(x: 0.2, y: 0.2)]

    private func focused() -> ManualCapture {
        var camera = ManualCapture()
        camera.observe(corners: corners, clear: true, adjusting: false, at: -0.21)
        XCTAssertFalse(camera.requestFocus(at: -0.1))
        camera.observe(corners: corners, clear: true, adjusting: false, at: 0)
        XCTAssertTrue(camera.requestFocus(at: 0.1))
        camera.observe(corners: corners, clear: true, adjusting: false, at: 0.6)
        XCTAssertEqual(camera.step, .focusing)
        camera.observe(corners: corners, clear: true, adjusting: false, at: 0.9)
        XCTAssertEqual(camera.step, .ready)
        return camera
    }

    func testFramingAloneNeverFocusesOrShoots() {
        var camera = ManualCapture()
        for tick in 0..<40 {
            let now = Double(tick) / 4
            camera.observe(corners: corners, clear: true, adjusting: false, at: now)
            XCTAssertEqual(camera.step, tick == 0 ? .confirming : .framed)
            XCTAssertFalse(camera.requestShutter(at: now, adjusting: false))
        }
    }

    func testManualShutterCapturesOnlyOnce() {
        var camera = focused()
        camera.observe(corners: corners, clear: true, adjusting: false, at: 1.1)
        XCTAssertEqual(camera.step, .ready)
        XCTAssertTrue(camera.requestShutter(at: 1.2, adjusting: false))
        XCTAssertFalse(camera.requestShutter(at: 1.2, adjusting: false))
    }

    func testBriefOutlineCannotEnableFocusAndClippingRevokesItImmediately() {
        var gate = ManualCapture()
        var audio = FramingFeedback()
        let left = FramingCorrection(.left, fit: 0.42)
        func frame(_ points: [CGPoint], _ partial: FramingCorrection?, _ t: Double) -> CameraCue? {
            gate.observe(corners: points, clear: true, adjusting: false, at: t, partial: partial)
            return audio.update(step: gate.step, correction: gate.correction, at: t, speechAvailable: false).cue
        }
        _ = frame([], left, 0)
        XCTAssertNil(frame(corners, nil, 0.15))
        XCTAssertFalse(gate.requestFocus(at: 0.16))
        XCTAssertNil(frame([], left, 0.3))
        XCTAssertNil(frame(corners, nil, 0.45))
        XCTAssertEqual(frame(corners, nil, 0.75), .entered)
        XCTAssertTrue(gate.requestFocus(at: 0.76))
        _ = frame([], left, 0.8)
        XCTAssertEqual(gate.step, .clipped)
        XCTAssertFalse(gate.requestShutter(at: 0.81, adjusting: false))
    }

    func testMovedClippedOrStaleDocumentRevokesReadiness() {
        var camera = focused()
        camera.observe(corners: corners.map { CGPoint(x: $0.x + 0.1, y: $0.y) },
                       clear: true, adjusting: false, at: 1.1)
        XCTAssertEqual(camera.step, .framed)
        camera.observe(corners: [], clear: true, adjusting: false, at: 1.3)
        XCTAssertFalse(camera.requestFocus(at: 1.4))
        var clipped = corners
        clipped[0].x = 0
        camera.observe(corners: clipped, clear: true, adjusting: false, at: 1.6)
        XCTAssertEqual(camera.step, .clipped)
        XCTAssertFalse(camera.requestFocus(at: 1.7))
        camera = focused()
        XCTAssertFalse(camera.requestShutter(at: 2.1, adjusting: false))
        XCTAssertEqual(camera.step, .searching)
    }

    func testBlurOrLensMovementBlocksShutter() {
        var camera = focused()
        XCTAssertFalse(camera.requestShutter(at: 1.0, adjusting: true))
        for i in 1...18 {
            camera.observe(corners: corners, clear: false, adjusting: false, at: 1 + Double(i) / 4)
            XCTAssertFalse(camera.requestShutter(at: 1 + Double(i) / 4, adjusting: false))
        }
        XCTAssertEqual(camera.step, .framed)
    }

    func testDirectionsAreRelativeToMovingThePhone() {
        var camera = ManualCapture()
        for (corner, hint) in [(CGPoint(x: 0, y: 0.8), FramingHint.left),
                               (.init(x: 1, y: 0.8), .right), (.init(x: 0.2, y: 1), .up),
                               (.init(x: 0.2, y: 0), .down), (.init(x: 0, y: 1), .left)] {
            var cut = corners
            cut[0] = corner
            camera.observe(corners: cut, clear: true, adjusting: false, at: 0)
            XCTAssertEqual(camera.framingHint, hint)
        }
        let grid = (0..<24).flatMap { y in (0..<16).map { x in UInt8(x < 13 && (4..<20).contains(y) ? 220 : 65) } }
        let partial = FramingHint.partialPage(luma: grid, width: 16, height: 24)
        XCTAssertEqual(partial, .left)
        camera.observe(corners: [], clear: true, adjusting: false, at: 1, partialHint: partial)
        XCTAssertFalse(camera.requestFocus(at: 1))
        XCTAssertFalse(camera.requestShutter(at: 1, adjusting: false))
        XCTAssertNil(FramingHint.partialPage(luma: Array(repeating: 190, count: 96), width: 8, height: 12))
    }

    func testPartialSheetOffCenterAndOppositeEdges() throws {
        let w = 24, h = 32
        func sheet(xs: Range<Int>, ys: Range<Int>) throws -> FramingCorrection {
            let grid = (0..<h).flatMap { y in (0..<w).map { x in UInt8(xs.contains(x) && ys.contains(y) ? 218 : 55) } }
            return try XCTUnwrap(FramingCorrection.partialPage(luma: grid, width: w, height: h))
        }
        let sliver = try sheet(xs: 0..<7, ys: 3..<27)
        let more = try sheet(xs: 0..<14, ys: 3..<27)
        XCTAssertEqual(sliver.direction, .left)
        XCTAssertGreaterThan(try XCTUnwrap(more.fit), try XCTUnwrap(sliver.fit))
        let tooClose = try sheet(xs: 2..<22, ys: 0..<h)
        let lifted = try sheet(xs: 5..<19, ys: 0..<h)
        XCTAssertEqual(tooClose.direction, .farther)
        XCTAssertGreaterThan(try XCTUnwrap(lifted.fit), try XCTUnwrap(tooClose.fit))
        XCTAssertEqual(try sheet(xs: 5..<19, ys: 0..<17).direction, .up)
        XCTAssertEqual(try sheet(xs: 5..<19, ys: 18..<h).direction, .down)
    }

    func testContrastingDarkObjectCannotPassAsBlankPaper() throws {
        let bounds = CGRect(x: 0.2, y: 0.2, width: 0.6, height: 0.6)
        func surface(_ background: Int, _ inside: (CGPoint) -> Int) throws -> PreviewPaperEvidence {
            try XCTUnwrap(PreviewPaperEvidence(corners: corners) { p in bounds.contains(p) ? inside(p) : background })
        }
        let darkObject = try surface(174) { p in Int(p.x * 25 + p.y * 15) + 47 }
        XCTAssertTrue(darkObject.hasDistinctEdges)
        XCTAssertFalse(darkObject.hasPaperBackground)
        XCTAssertTrue(try surface(155) { _ in 215 }.hasPaperBackground)
        XCTAssertTrue(try surface(43) { _ in 97 }.hasPaperBackground)
        XCTAssertTrue(try surface(64) { p in p.y < 0.48 ? 50 : 216 }.hasPaperBackground)
    }

    func testActualBeatIntervalsShortenWithProgressAndSlowWhenBackingOut() throws {
        var feedback = FramingFeedback()
        var tick = 0
        func beats(width: Double, seconds: Int) -> [Double] {
            let points = [CGPoint(x: 0, y: 0.8), CGPoint(x: width, y: 0.8),
                          CGPoint(x: width, y: 0.2), CGPoint(x: 0, y: 0.2)]
            var times = [Double]()
            for _ in 0..<(seconds * 10) {
                let time = Double(tick) / 10
                let signal = feedback.update(step: .clipped, correction: FramingCorrection.clipped(points),
                                             at: time, speechAvailable: false)
                if case .guiding = signal.cue { times.append(time) }
                tick += 1
            }
            return times
        }
        func interval(_ beats: [Double]) throws -> Double {
            let last = try XCTUnwrap(beats.last), previous = try XCTUnwrap(beats.dropLast().last)
            return last - previous
        }
        let far = try interval(beats(width: 0.2, seconds: 5))
        let advancing = try interval(beats(width: 0.47, seconds: 4))
        let near = try interval(beats(width: 0.71, seconds: 4))
        XCTAssertLessThan(advancing, far - 0.15)
        XCTAssertLessThan(near, advancing - 0.05)
        XCTAssertEqual(try interval(beats(width: 0.71, seconds: 5)), near, accuracy: 0.06)
        XCTAssertGreaterThan(try interval(beats(width: 0.2, seconds: 4)), near + 0.25)
    }

    func testEntryBeepIsStableAndRearmsOnlyAfterLoss() {
        var sound = FramingFeedback()
        func cue(_ step: CaptureStep, _ t: Double) -> CameraCue? {
            sound.update(step: step, correction: nil, at: t, speechAvailable: false).cue
        }
        XCTAssertNil(cue(.searching, 0))
        XCTAssertEqual(cue(.searching, 0.4), .searching)
        XCTAssertNil(cue(.framed, 0.5))
        XCTAssertNil(cue(.clipped, 0.6))
        XCTAssertNil(cue(.framed, 0.7))
        XCTAssertEqual(cue(.framed, 1.0), .entered)
        XCTAssertNil(cue(.clipped, 1.05))
        XCTAssertNil(cue(.framed, 1.2))
        XCTAssertNil(cue(.framed, 1.55))
        XCTAssertEqual(cue(.ready, 1.7), .ready)
        XCTAssertNil(cue(.ready, 1.8))
        _ = cue(.clipped, 2)
        _ = cue(.clipped, 2.6)
        XCTAssertNil(cue(.framed, 2.7))
        XCTAssertEqual(cue(.framed, 3.0), .entered)
        XCTAssertNil(cue(.capturing, 3.1))
    }

    func testAudioClockBetweenVisionFramesAndAfterAStall() {
        var feedback = FramingFeedback()
        let near = FramingCorrection(.left, fit: 0.98)
        _ = feedback.update(step: .clipped, correction: near, at: 0, speechAvailable: false)
        let first = feedback.update(step: .clipped, correction: near, at: 0.35, speechAvailable: false)
        XCTAssertEqual(first.cue, .guiding(4))
        XCTAssertNil(feedback.pulse(at: 0.50))
        XCTAssertEqual(feedback.pulse(at: 0.64), .guiding(4))
        XCTAssertEqual(feedback.pulse(at: 0.93), .guiding(4))
        XCTAssertNil(feedback.pulse(at: 1.36))
    }

    func testSpeechWaitsItsTurnAndDoesNotPraiseJitterOrAnOldDirection() {
        var f = FramingFeedback()
        var messages = [String]()
        for i in 0...60 {
            let t = Double(i) / 10
            let fit = i < 32 ? (i.isMultiple(of: 2) ? 0.30 : 0.32) : min(0.75, 0.30 + Double(i - 32) * 0.035)
            let output = f.update(step: .clipped, correction: .init(.left, fit: fit), at: t, speechAvailable: t >= 1)
            if let message = output.message { messages.append(message) }
            if i == 9 { XCTAssertTrue(messages.isEmpty) }
            if i == 31 { XCTAssertEqual(messages, [FramingHint.left.instruction]) }
        }
        XCTAssertEqual(messages.filter { $0.contains("좋아요") }.count, 1)
        var reversal = [String]()
        for i in 61...90 {
            let fit = max(0.25, 0.75 - Double(i - 60) * 0.035)
            if let message = f.update(step: .clipped, correction: .init(.left, fit: fit),
                                      at: Double(i) / 10, speechAvailable: true).message { reversal.append(message) }
        }
        XCTAssertEqual(reversal.filter { $0.contains("벗어났어요") }.count, 1)
        for i in 91...102 {
            _ = f.update(step: .searching, correction: .init(.findPage), at: Double(i) / 10, speechAvailable: false)
        }
        XCTAssertNil(f.fit)
        XCTAssertNil(f.trend)
        XCTAssertEqual(f.direction, .findPage)
    }
}
