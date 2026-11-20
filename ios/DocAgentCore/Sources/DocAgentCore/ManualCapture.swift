import Foundation

public enum CaptureStep: Hashable {
    case searching, clipped, confirming, framed, focusing, ready, capturing

    public var needsFraming: Bool { self == .searching || self == .clipped || self == .confirming }

    public var instruction: String {
        switch self {
        case .searching: return "문서 네 모서리가 보이도록 카메라를 맞춰 주세요."
        case .clipped: return "문서 끝이 화면에 걸렸어요. 카메라를 조금 멀리 이동해 주세요."
        case .confirming: return "잠시 그대로 들어 주세요. 네 모서리를 확인하고 있어요."
        case .framed: return "문서가 들어왔어요. 초점 맞추기를 눌러 주세요."
        case .focusing: return "초점을 확인하고 있어요. 잠시 그대로 유지해 주세요."
        case .ready: return "준비됐어요. 촬영 버튼을 눌러 주세요."
        case .capturing: return "사진을 촬영하고 있어요."
        }
    }
}

/// All methods are called on the camera session queue. Only requestShutter can start a capture.
public struct ManualCapture {
    public private(set) var step = CaptureStep.searching
    public private(set) var correction: FramingCorrection? = FramingCorrection(.findPage)
    public var framingHint: FramingHint? { correction?.direction }
    private var lastCorners = [CGPoint]()
    private var focusedCorners = [CGPoint]()
    private var lastSeen = -Double.infinity
    private var candidateSince = 0.0
    private var focusedAt = -Double.infinity
    private var clearSamples = 0
    public init() {}

    public mutating func observe(corners: [CGPoint], clear: Bool, adjusting: Bool, at now: Double,
                                 partialHint: FramingHint? = nil) {
        observe(corners: corners, clear: clear, adjusting: adjusting, at: now,
                partial: partialHint.map { FramingCorrection($0) })
    }

    public mutating func observe(corners: [CGPoint], clear: Bool, adjusting: Bool, at now: Double,
                                 partial: FramingCorrection?) {
        guard step != .capturing else { return }
        guard corners.count == 4 else {
            // Brief low-contrast frames should not restart the direction prompt.
            // Both buttons are immediately unavailable until a real outline returns.
            if partial == nil, !lastCorners.isEmpty, now - lastSeen <= 0.32 {
                focusedCorners = []
                clearSamples = 0
                candidateSince = now
                step = .confirming
                correction = nil
                return
            }
            self = ManualCapture()
            if let partial {
                step = .clipped
                correction = partial
            }
            return
        }
        if let edge = FramingCorrection.clipped(corners) {
            self = ManualCapture()
            step = .clipped
            correction = edge
            return
        }
        correction = nil
        let interrupted = now - lastSeen > 1.25
        if interrupted || step.needsFraming {
            let jumped = zip(corners, lastCorners).contains { hypot($0.0.x - $0.1.x, $0.0.y - $0.1.y) > 0.09 }
            if step != .confirming || now - lastSeen > 0.65 || jumped { candidateSince = now }
            lastSeen = now
            lastCorners = corners
            focusedCorners = []
            clearSamples = 0
            step = now - candidateSince >= 0.18 ? .framed : .confirming
            return
        }
        let moved = zip(corners, focusedCorners).contains { point, anchor in
            hypot(point.x - anchor.x, point.y - anchor.y) > 0.045
        }
        lastCorners = corners
        lastSeen = now
        if moved {
            focusedCorners = []
            clearSamples = 0
            step = .framed
        }
        guard step == .focusing || step == .ready else { return }
        if adjusting || !clear {
            if step == .ready { focusedAt = now }
            clearSamples = 0
            step = .focusing
        } else if now - focusedAt >= 0.4 {
            clearSamples += 1
            if clearSamples >= 2 { step = .ready }
        }
        if step == .focusing && now - focusedAt > 4 {
            step = .framed
        }
    }

    public mutating func requestFocus(at now: Double) -> Bool {
        guard !step.needsFraming, step != .capturing else { return false }
        guard now - lastSeen < 1 else {
            self = ManualCapture()
            return false
        }
        focusedAt = now
        focusedCorners = lastCorners
        clearSamples = 0
        step = .focusing
        return true
    }

    public mutating func requestShutter(at now: Double, adjusting: Bool) -> Bool {
        guard step == .ready else { return false }
        guard now - lastSeen < 1 else {
            self = ManualCapture()
            return false
        }
        guard !adjusting else {
            focusedAt = now
            clearSamples = 0
            step = .focusing
            return false
        }
        step = .capturing
        return true
    }
}
