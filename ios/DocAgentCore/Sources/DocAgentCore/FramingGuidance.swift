import Foundation

public enum FramingHint: Equatable {
    case findPage, left, right, up, down, farther

    public var instruction: String {
        switch self {
        case .findPage: return "종이 위에서 휴대폰을 조금 높여 천천히 찾아 주세요."
        case .left: return "종이 왼쪽이 잘렸어요. 휴대폰을 왼쪽으로 옮겨 주세요."
        case .right: return "종이 오른쪽이 잘렸어요. 휴대폰을 오른쪽으로 옮겨 주세요."
        case .up: return "종이 위쪽이 잘렸어요. 휴대폰을 몸에서 먼 쪽으로 옮겨 주세요."
        case .down: return "종이 아래쪽이 잘렸어요. 휴대폰을 몸 쪽으로 옮겨 주세요."
        case .farther: return "휴대폰을 종이에서 조금 더 높이 들어 주세요."
        }
    }

    public static func partialPage(luma: [UInt8], width: Int, height: Int) -> FramingHint? {
        FramingCorrection.partialPage(luma: luma, width: width, height: height)?.direction
    }
}

/// fit measures visible framing geometry, not OCR confidence or physical distance.
public struct FramingCorrection: Equatable {
    public let direction: FramingHint
    public let fit: Double?
    public init(_ direction: FramingHint, fit: Double? = nil) {
        self.direction = direction
        self.fit = fit.map { min(1, max(0, $0)) }
    }

    static func clipped(_ corners: [CGPoint]) -> Self? {
        guard corners.count == 4 else { return nil }
        let xs = corners.map(\.x), ys = corners.map(\.y)
        let x0 = xs.min()!, x1 = xs.max()!, y0 = ys.min()!, y1 = ys.max()!
        let margin = 0.006
        return fromEdges(left: x0 <= margin, right: x1 >= 1 - margin, top: y1 >= 1 - margin, bottom: y0 <= margin,
                         width: x1 - x0, height: y1 - y0)
    }

    static func fromEdges(left: Bool, right: Bool, top: Bool, bottom: Bool,
                                  width: Double, height: Double) -> Self? {
        if left && right || top && bottom {
            // Lifting the phone reveals background on the perpendicular axis. When
            // both axes are clipped there is no reliable distance measurement.
            let fit: Double? = left && right && top && bottom ? nil : 1 - (left && right ? height : width)
            return Self(.farther, fit: fit)
        }
        let horizontal = left || right, vertical = top || bottom
        guard horizontal || vertical else { return nil }
        let xFit = horizontal ? width : 1, yFit = vertical ? height : 1
        let direction: FramingHint
        if horizontal && (!vertical || xFit <= yFit + 0.001) { direction = left ? .left : .right }
        else { direction = top ? .up : .down }
        return Self(direction, fit: xFit * yFit)
    }

    /// When Vision loses a corner, measure a contiguous paper-colored region touching the frame.
    /// This fallback only guides movement. It never enables focus or shutter.
    public static func partialPage(luma: [UInt8], width: Int, height: Int) -> Self? {
        guard width >= 8, height >= 8, luma.count == width * height else { return nil }
        let sorted = luma.sorted(), count = luma.count
        let dark = Int(sorted[count / 10]), light = Int(sorted[count * 7 / 8])
        guard light >= 145, light - dark >= 38 else { return nil }
        let threshold = dark + (light - dark) * 3 / 5
        var pending = luma.map { Int($0) >= threshold }
        var largest = [Int]()
        for seed in pending.indices where pending[seed] {
            var region = [seed], next = 0
            pending[seed] = false
            while next < region.count {
                let index = region[next], x = index % width
                next += 1
                for neighbor in [x > 0 ? index - 1 : -1, x + 1 < width ? index + 1 : -1, index - width, index + width]
                    where pending.indices.contains(neighbor) && pending[neighbor] {
                    pending[neighbor] = false
                    region.append(neighbor)
                }
            }
            if region.count > largest.count { largest = region }
        }
        guard largest.count >= count / 9, largest.count <= count * 93 / 100 else { return nil }
        var x0 = width, x1 = 0, y0 = height, y1 = 0
        var left = 0, right = 0, top = 0, bottom = 0
        for index in largest {
            let x = index % width, y = index / width
            x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
            if x == 0 { left += 1 }
            if x == width - 1 { right += 1 }
            if y == 0 { top += 1 }
            if y == height - 1 { bottom += 1 }
        }
        let w = x1 - x0 + 1, h = y1 - y0 + 1
        guard w >= 4, h >= 4, largest.count * 100 >= w * h * 55 else { return nil }
        let side = max(2, h / 5), end = max(2, w / 5)
        // Raster rows start at the top, while Vision's Y axis starts at the bottom.
        return fromEdges(left: left >= side, right: right >= side, top: top >= end, bottom: bottom >= end,
                         width: Double(w) / Double(width), height: Double(h) / Double(height))
    }
}

public enum CameraCue: Hashable {
    case searching, guiding(Int), entered, aligned, ready
    /// Pitch and spacing rise together so changes also work through a mono phone speaker.
    public static let guidePitches = [490, 575, 665, 765, 880]
    public static func period(fit: Double) -> Double { 0.94 - 0.67 * min(1, max(0, fit)) }
}

public enum FramingTrend { case improving, worsening }
public struct FramingFeedback {
    public private(set) var fit: Double?
    public private(set) var direction: FramingHint?
    public private(set) var trend: FramingTrend?
    private var candidate: FramingHint?
    private var candidateSince = 0.0
    private var insideSince: Double?
    private var outsideSince: Double?
    private var entered = false
    private var previousStep: CaptureStep?
    private var lastSample = -Double.infinity
    private var lastTone = -Double.infinity
    private var firstFit: Double?
    private var filteredFit: Double?
    private var anchor: Double?
    private var trendAt = -Double.infinity
    private var spokenTrend: FramingTrend?
    private var lastSpoken: FramingHint?
    private var lastSpokenAt = -Double.infinity
    public init() {}

    public mutating func update(step: CaptureStep, correction: FramingCorrection?, at now: Double,
                                speechAvailable: Bool) -> (cue: CameraCue?, message: String?) {
        let dt = max(0, now - lastSample)
        if dt > 1.25 { self = FramingFeedback() }
        lastSample = now
        let previous = previousStep
        previousStep = step
        guard step != .capturing else { return (nil, nil) }
        if step == .confirming {
            insideSince = insideSince ?? now
            candidate = nil
            return (nil, nil)
        }
        if !step.needsFraming {
            outsideSince = nil
            insideSince = insideSince ?? now
            guard step == .ready || now - (insideSince ?? now) >= 0.25 else { return (nil, nil) }
            candidate = nil
            direction = nil
            fit = nil
            firstFit = nil
            filteredFit = nil
            anchor = nil
            trend = nil
            if step == .ready {
                entered = true
                guard previous != .ready else { return (nil, nil) }
                lastTone = now
                return (.ready, nil)
            }
            if !entered {
                entered = true
                lastSpoken = nil
                lastTone = now
                return (.entered, nil)
            }
            return (pulse(at: now), nil)
        }

        insideSince = nil
        outsideSince = outsideSince ?? now
        if entered && now - (outsideSince ?? now) < 0.45 { return (nil, nil) }
        entered = false
        let observed = correction?.direction ?? .findPage
        if observed != candidate { candidate = observed; candidateSince = now }
        let delay = direction == nil && correction?.fit != nil ? 0.15 : 0.30
        guard now - candidateSince >= delay else { return (nil, nil) }
        if direction != observed {
            direction = observed
            fit = nil
            firstFit = nil
            filteredFit = nil
            anchor = nil
            trend = nil
            spokenTrend = nil
        }
        if let measured = correction?.fit {
            let previous = filteredFit ?? measured
            let filtered = previous + (1 - exp(-min(dt, 1) / 0.22)) * (measured - previous)
            filteredFit = filtered
            firstFit = firstFit ?? filtered
            // Make sustained progress in the same direction audible even when the sheet
            // only spans part of the image. Standing still cannot increase the pace.
            let gain = filtered - (firstFit ?? filtered)
            fit = min(1, max(0, filtered + 0.9 * gain))
            let change = filtered - (anchor ?? filtered)
            if anchor == nil || abs(change) >= 0.09 {
                if anchor != nil {
                    trend = change > 0 ? .improving : .worsening
                    trendAt = now
                }
                anchor = filtered
            }
        } else {
            fit = nil
            firstFit = nil
            filteredFit = nil
            anchor = nil
            trend = nil
        }
        if now - trendAt > 1.8 { trend = nil }
        let message = speechAvailable ? nextMessage(for: observed, at: now) : nil
        return (pulse(at: now), message)
    }

    /// The sound clock also ticks between preview frames. Only fresh camera evidence may drive it.
    public mutating func pulse(at now: Double) -> CameraCue? {
        guard now >= lastSample, now - lastSample <= 0.70, let step = previousStep,
              step != .ready, step != .capturing, step != .confirming else { return nil }
        let cue: CameraCue, interval: Double
        if step.needsFraming {
            guard !entered, let direction, direction == candidate else { return nil }
            interval = fit.map(CameraCue.period) ?? 0.98
            cue = fit.map { .guiding(Int(($0 * Double(CameraCue.guidePitches.count - 1)).rounded())) } ?? .searching
        } else {
            guard entered else { return nil }
            interval = 0.85
            cue = .aligned
        }
        guard now - lastTone >= interval else { return nil }
        lastTone = now
        return cue
    }

    private mutating func nextMessage(for hint: FramingHint, at now: Double) -> String? {
        let elapsed = now - lastSpokenAt
        guard elapsed >= 2.5 else { return nil }
        let message: String
        if hint != lastSpoken {
            message = hint.instruction
        } else if let trend, trend != spokenTrend {
            spokenTrend = trend
            message = trend == .improving ? "좋아요. 그 방향으로 조금 더 옮겨 주세요." : "조금 더 벗어났어요. " + hint.instruction
        } else if elapsed >= 12 {
            message = hint.instruction
        } else { return nil }
        lastSpokenAt = now
        lastSpoken = hint
        return message
    }
}
