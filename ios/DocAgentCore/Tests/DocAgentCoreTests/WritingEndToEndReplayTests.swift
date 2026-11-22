import CoreImage
import XCTest
@testable import DocAgentCore

/// Optional, repeatable offline exercise of the actual photographed page, OpenCV tracker and arrival decision.
/// No real hand or iPhone is present. Only the fingertip landmarks and camera changes are simulated.
final class WritingEndToEndReplayTests: XCTestCase {
    private struct Viewpoint {
        let name: String
        /// Known projective transform in upright normalized coordinates, independent of the tracker estimate.
        let matrix: [Double]

        func project(_ p: CGPoint, jitter: CGFloat) -> CGPoint {
            let x = p.x, y = p.y
            let s = matrix[6] * x + matrix[7] * y + matrix[8]
            return CGPoint(x: (matrix[0]*x + matrix[1]*y + matrix[2]) / s + jitter,
                           y: (matrix[3]*x + matrix[4]*y + matrix[5]) / s - jitter / 2)
        }
    }

    private let raster = CGRect(x: 0, y: 0, width: 720, height: 960)

    private func frame(_ source: CGImage, viewpoint: Viewpoint, jitter: CGFloat, context: CIContext) throws -> CGImage {
        func corner(_ x: CGFloat, _ y: CGFloat) -> CIVector {
            let p = viewpoint.project(CGPoint(x: x, y: y), jitter: jitter)
            return CIVector(x: p.x * raster.width, y: (1-p.y) * raster.height)
        }
        let background = CIImage(color: CIColor(red: 0.18, green: 0.18, blue: 0.18)).cropped(to: raster)
        var output = CIImage(cgImage: source).applyingFilter("CIPerspectiveTransform", parameters: [
            "inputTopLeft": corner(0,0), "inputTopRight": corner(1,0),
            "inputBottomRight": corner(1,1), "inputBottomLeft": corner(0,1)
        ]).composited(over: background).cropped(to: raster)
        if viewpoint.name == "occluded margin" {
            let paperCover = CGRect(x: 75, y: 85, width: 195, height: 310)
            output = CIImage(color: CIColor(red: 0.67, green: 0.5, blue: 0.40))
                .cropped(to: paperCover).composited(over: output)
        }
        if viewpoint.name == "dim" {
            output = output.applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: -1.05])
        }
        return try XCTUnwrap(context.createCGImage(output, from: raster))
    }

    func testPhotoToArrivalThroughRegisteredLiveFrames() throws {
        guard let path = ProcessInfo.processInfo.environment["DOCAGENT_DEMO_PHOTO"] else {
            throw XCTSkip("Set DOCAGENT_DEMO_PHOTO to the original IMG_2510.heic to run the private photo replay.")
        }
        let bytes = try Data(contentsOf: URL(fileURLWithPath: path))
        let fields = try XCTUnwrap(DemoConsentForm.exactPhoto(bytes)).targets
        let reference = try WritingReference(photo: bytes, rectify: false)
        let aspect = CGFloat(reference.image.width) / CGFloat(reference.image.height)
        XCTAssertEqual(fields.filter { $0.kind == .checkbox }.count, 4)
        XCTAssertEqual(fields.filter { $0.kind == .signature }.count, 1)
        let viewpoints: [Viewpoint] = [
            .init(name: "upright", matrix: [1,0,0, 0,1,0, 0,0,1]),
            .init(name: "table margin", matrix: [0.76,0,0.12, 0,0.84,0.08, 0,0,1]),
            .init(name: "tilt left", matrix: [0.76,-0.08,0.15, 0.10,0.78,0.07, 0.03,-0.02,1]),
            .init(name: "tilt right", matrix: [0.73,0.08,0.08, -0.11,0.78,0.17, -0.02,0.05,1]),
            .init(name: "quarter turn", matrix: [0,-0.83,0.91, 0.64,0,0.16, 0,0,1]),
            .init(name: "occluded margin", matrix: [0.89,-0.02,0.06, -0.02,0.87,0.06, 0,0,1]),
            .init(name: "dim", matrix: [0.87,0.03,0.04, 0.04,0.88,0.04, 0,0,1])
        ]
        let context = CIContext(options: [.cacheIntermediates: false])
        let blank = try XCTUnwrap(context.createCGImage(CIImage(color: CIColor(red: 0.85, green: 0.85, blue: 0.85)), from: raster))
        let frameCount = 23, entry = 6
        var delays = [Int](), registrationTimes = [Double]()
        var wrongTrials = 0, wrongAlerts = 0, lostHandTrials = 0, lostHandAlerts = 0
        var lostPageTrials = 0, lostPageAlerts = 0, earlyEpisodes = 0, recovered = 0
        for viewpoint in viewpoints {
            let tracker = try XCTUnwrap(WritingPageRegistration(reference: reference.image))
            let sample = try [0.0, 0.0015].map { try frame(reference.image, viewpoint: viewpoint, jitter: $0, context: context) }
            // Compute ORB on every frame. All labelled trajectories share the same camera video for a viewpoint.
            let poses = (0..<frameCount).map { i -> WritingTransform? in
                let image = sample[i % 2]
                var buffer: CVPixelBuffer?
                XCTAssertEqual(CVPixelBufferCreate(nil, image.width, image.height, kCVPixelFormatType_32BGRA,
                                                   [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary,
                                                   &buffer), kCVReturnSuccess)
                guard let buffer else { return nil }
                context.render(CIImage(cgImage: image), to: buffer)
                let started = ProcessInfo.processInfo.systemUptime
                let pose = tracker.register(buffer)
                registrationTimes.append((ProcessInfo.processInfo.systemUptime - started) * 1000)
                return pose
            }
            let lostPose = tracker.register(blank)
            XCTAssertNil(lostPose, viewpoint.name)
            for target in fields {
                let center = CGPoint(x: target.bounds.midX, y: target.bounds.midY)
                func observe(_ guide: inout WritingGuidance, _ i: Int, point: CGPoint?, pose: WritingTransform?) -> Bool {
                    let tip = point.map { viewpoint.project($0, jitter: i.isMultiple(of: 2) ? 0.0 : 0.0015) }
                    return guide.update(pose: pose, fingertip: tip, target: target, fields: fields,
                                        time: Double(i) / 10, paperCoordinates: tracker.paperCoordinates, referenceAspect: aspect).arrived
                }
                var guide = WritingGuidance(), firstArrival: Int?, firstNear: Int?
                var early = false
                for i in 0..<frameCount {
                    let remaining = CGFloat(max(0, entry-i))
                    let point = CGPoint(x: center.x, y: center.y - remaining * max(0.014, target.bounds.height))
                    // Dwell begins at entry into the 8 mm band, before the trajectory reaches its center.
                    func paperPosition(_ reference: CGPoint) -> CGPoint {
                        if let p = tracker.paperCoordinates?.livePoint(reference) {
                            return CGPoint(x: p.x * 210, y: p.y * 297)
                        }
                        return CGPoint(x: reference.x * 210, y: reference.y * 210 / aspect)
                    }
                    let a = paperPosition(point), b = paperPosition(center)
                    if firstNear == nil && hypot(a.x-b.x, a.y-b.y) <= 8 { firstNear = i }
                    if observe(&guide, i, point: point, pose: poses[i]) {
                        early = early || firstNear == nil || i < firstNear!
                        if firstArrival == nil { firstArrival = i }
                    }
                }
                if early { earlyEpisodes += 1 }
                if let firstArrival, let firstNear { delays.append((firstArrival-firstNear) * 100) }
                else { XCTFail("No arrival for \(viewpoint.name), \(target.id)") }

                for other in fields where other.id != target.id {
                    var offGuide = WritingGuidance()
                    let otherTip = CGPoint(x: other.bounds.midX, y: other.bounds.midY)
                    let fired = poses.indices.map { observe(&offGuide, $0, point: otherTip, pose: poses[$0]) }
                    wrongTrials += 1
                    if fired.contains(true) { wrongAlerts += 1 }
                }
                // Interrupt an already running hold, then present the hand again.
                var handGuide = WritingGuidance(), offDuringReset = false, recoveredOnTime = false
                for i in 0..<frameCount {
                    let visible = i < 6 || i >= 12
                    let arrived = observe(&handGuide, i, point: visible ? center : nil, pose: poses[i])
                    if (6..<19).contains(i) { offDuringReset = offDuringReset || arrived }
                    if i >= 19 && arrived { recoveredOnTime = true }
                }
                lostHandTrials += 1
                if offDuringReset { lostHandAlerts += 1 }
                if recoveredOnTime { recovered += 1 }
                // The page vanishes after the same target previously produced a real arrival event.
                var pageGuide = WritingGuidance(), offDuringPageLoss = false
                for i in 0..<frameCount {
                    let visible = i < 10
                    let arrived = observe(&pageGuide, i, point: center, pose: visible ? poses[i] : lostPose)
                    if !visible { offDuringPageLoss = offDuringPageLoss || arrived }
                }
                lostPageTrials += 1
                if offDuringPageLoss { lostPageAlerts += 1 }
            }
        }
        let attempts = fields.count * viewpoints.count
        let times = registrationTimes.sorted()
        let sorted = delays.sorted()
        print("writing-guidance-replay correct=\(delays.count)/\(attempts) wrong=\(wrongAlerts)/\(wrongTrials) early=\(earlyEpisodes)/\(attempts) handLoss=\(lostHandAlerts)/\(lostHandTrials) pageLoss=\(lostPageAlerts)/\(lostPageTrials) reacquired=\(recovered)/\(attempts)")
        if !sorted.isEmpty { print("simulated_entry_to_alert_ms min=\(sorted.first!) median=\(sorted[sorted.count/2]) max=\(sorted.last!)") }
        print(String(format: "mac_ORB_ms n=%d median=%.2f p95=%.2f", times.count, times[times.count/2],
                     times[Int(ceil(Double(times.count)*0.95))-1]))
        XCTAssertEqual(delays.count, attempts)
        XCTAssertEqual(earlyEpisodes + wrongAlerts + lostHandAlerts + lostPageAlerts, 0)
        XCTAssertEqual(recovered, attempts)
        XCTAssertTrue(delays.allSatisfy { (650...900).contains($0) })
    }
}
