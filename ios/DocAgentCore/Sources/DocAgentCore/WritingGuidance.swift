import Foundation
import simd

/// The same upright image must be used for locating fields and creating the tracker.
/// Coordinates have their origin at the top left, independently of camera resolution.
public struct WritingTarget: Codable, Identifiable, Equatable, Sendable {
    public enum Kind: String, Codable, Sendable { case checkbox, signature, input }
    public let id: String
    public let label: String
    public let kind: Kind
    public let bounds: CGRect

    public init(id: String, label: String, kind: Kind, bounds: CGRect) {
        self.id = id
        self.label = label
        self.kind = kind
        self.bounds = bounds
    }
}

public struct WritingFeedback: Equatable, Sendable {
    public let message: String
    public let arrived: Bool
    public let beepInterval: Double?
    public let repeatSpeechAfter: Double?
    public let targetOutline: [CGPoint]
    public let fingertip: CGPoint?

    public init(message: String, arrived: Bool = false, beepInterval: Double? = nil,
                repeatSpeechAfter: Double? = nil, targetOutline: [CGPoint] = [], fingertip: CGPoint? = nil) {
        self.message = message
        self.arrived = arrived
        self.beepInterval = beepInterval
        self.repeatSpeechAfter = repeatSpeechAfter
        self.targetOutline = targetOutline
        self.fingertip = fingertip
    }
}

public struct WritingTransform: Sendable {
    private let matrix: simd_double3x3

    init?(rowMajor values: [Double]) {
        guard values.count == 9, values.allSatisfy(\.isFinite) else { return nil }
        matrix = simd_double3x3(rows: [SIMD3(values[0], values[1], values[2]),
                                      SIMD3(values[3], values[4], values[5]),
                                      SIMD3(values[6], values[7], values[8])])
        guard abs(simd_determinant(matrix)) > 1e-9 else { return nil }
    }

    private func project(_ point: CGPoint, using matrix: simd_double3x3) -> CGPoint? {
        let p = matrix * SIMD3(Double(point.x), Double(point.y), 1)
        guard abs(p.z) > 1e-8, p.x.isFinite, p.y.isFinite, p.z.isFinite else { return nil }
        return CGPoint(x: p.x / p.z, y: p.y / p.z)
    }

    public func livePoint(_ reference: CGPoint) -> CGPoint? { project(reference, using: matrix) }
    public func referencePoint(_ live: CGPoint) -> CGPoint? { project(live, using: matrix.inverse) }
}

public struct WritingGuidance {
    public static let seeking = WritingFeedback(message: "읽었던 문서의 글자가 보이게 휴대폰을 들어 주세요.")
    public static let holdDuration = 0.65
    private static let liveArea = CGRect(x: 0.008, y: 0.008, width: 0.984, height: 0.984)
    private var previousPose: WritingTransform?
    private var previousTime: Double?
    private var insideSince: Double?
    private var smoothedPoint: CGPoint?

    public init() {}

    public mutating func update(pose: WritingTransform?, fingertip: CGPoint?, target: WritingTarget,
                                fields: [WritingTarget], time: Double,
                                paperCoordinates: WritingTransform? = nil,
                                referenceAspect: CGFloat = 210.0 / 297.0) -> WritingFeedback {
        guard let pose, time.isFinite else { self = Self(); return Self.seeking }
        // Distance is a normalized reference-image proxy, not a physical ruler.
        // Gate a new pose across two frames and discard held time on a sensor gap.
        let previous = previousPose
        let gap = time - (previousTime ?? time)
        previousPose = pose
        previousTime = time
        guard gap >= 0, gap <= 0.6, let previous, stable(previous, pose) else {
            insideSince = nil
            smoothedPoint = nil
            return Self.seeking
        }
        let b = target.bounds
        let corners = [CGPoint(x: b.minX,y: b.minY), CGPoint(x: b.maxX,y: b.minY),
                       CGPoint(x: b.maxX,y: b.maxY), CGPoint(x: b.minX,y: b.maxY)]
        let outline = corners.compactMap(pose.livePoint)
        guard !b.isNull, !b.isEmpty, outline.count == 4, outline.allSatisfy(Self.liveArea.contains) else {
            insideSince = nil
            smoothedPoint = nil
            return .init(message: "작성할 칸이 화면 밖에 있어요. 휴대폰을 조금 높여 주세요.")
        }
        guard let fingertip, let tip = pose.referencePoint(fingertip),
              Self.liveArea.contains(fingertip) else {
            insideSince = nil
            smoothedPoint = nil
            return .init(message: "한 손의 검지 끝이 보이게 종이 위에 놓아 주세요.", targetOutline: outline)
        }
        // A4-equivalent distances guide movement. Other paper sizes are not physically calibrated.
        func paperPoint(_ p: CGPoint) -> CGPoint {
            if let normalized = paperCoordinates?.livePoint(p) {
                return CGPoint(x: normalized.x * 210, y: normalized.y * 297)
            }
            return CGPoint(x: p.x * 210, y: p.y * 210 / max(0.15, min(6, referenceAspect)))
        }
        let point = paperPoint(tip)
        let center = paperPoint(CGPoint(x: b.midX, y: b.midY))
        let previousPoint = gap <= 0.5 ? smoothedPoint : nil
        let smooth = previousPoint.map {
            CGPoint(x: 0.55 * point.x + 0.45 * $0.x, y: 0.55 * point.y + 0.45 * $0.y)
        } ?? point
        smoothedPoint = smooth
        let dx = center.x - smooth.x, dy = center.y - smooth.y
        let distance = hypot(dx, dy)
        let inOther = fields.contains { $0.id != target.id && $0.bounds.contains(tip) }
        let atTarget = distance <= 8 && hypot(center.x - point.x, center.y - point.y) <= 8 && !inOther
        if atTarget {
            let since = insideSince ?? time
            insideSince = since
            let arrived = time - since >= Self.holdDuration
            return .init(message: arrived ? "\(target.label) 위치 근처에 도달했어요." : "그 자리에 손끝을 잠시 멈춰 주세요.",
                         arrived: arrived, beepInterval: arrived ? nil : 0.085, targetOutline: outline, fingertip: fingertip)
        }
        insideSince = nil
        let message: String
        if distance <= 10 && !inOther {
            message = "거의 다 왔습니다. 손가락을 천천히 움직여 주세요."
        } else {
            let horizontal = dx > 0 ? "오른쪽" : "왼쪽"
            let vertical = dy > 0 ? "아래" : "위"
            let heading: String
            if min(abs(dx), abs(dy)) >= max(11, 0.38 * max(abs(dx), abs(dy))) {
                heading = "\(horizontal) \(vertical) 대각선으로"
            } else if abs(dx) >= abs(dy) {
                heading = horizontal + "으로"
            } else {
                heading = vertical + "로"
            }
            if distance > 100 { message = heading + " 이동하세요." }
            else if distance > 35 { message = "계속 " + heading + " 움직이세요." }
            else { message = "조금 더 " + heading + " 움직이세요." }
        }
        let cadence: Double
        switch distance {
        case ...12: cadence = 5.2
        case ...35: cadence = 4.2
        case ...100: cadence = 4.5
        default: cadence = 3.7
        }
        return .init(message: message, beepInterval: Self.interval(distance), repeatSpeechAfter: cadence,
                     targetOutline: outline, fingertip: fingertip)
    }

    private func stable(_ a: WritingTransform, _ b: WritingTransform) -> Bool {
        // v7.5 compares five visible anchors in reference space so partial pages work too.
        let probes = [CGPoint(x: 0.5,y: 0.5), CGPoint(x: 0.25,y: 0.3), CGPoint(x: 0.75,y: 0.3),
                      CGPoint(x: 0.25,y: 0.7), CGPoint(x: 0.75,y: 0.7)]
        let distances = probes.compactMap { probe -> CGFloat? in
            guard let pa = a.referencePoint(probe), let pb = b.referencePoint(probe) else { return nil }
            return hypot(pa.x - pb.x, pa.y - pb.y)
        }.sorted()
        return distances.count == probes.count && distances[2] < 0.065
    }

    private static func interval(_ units: Double) -> Double? {
        switch units {
        case ...8: return 0.085
        case ...12: return 0.105
        case ...20: return 0.18
        case ...35: return 0.32
        case ...60: return 0.55
        case ...100: return 0.85
        default: return nil
        }
    }
}
