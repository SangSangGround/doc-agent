import Foundation
import ImageIO
import Vision

public struct PrintedLine: Codable, Identifiable, Sendable {
    public let id: Int
    public let text: String
    /// An internal decision score for this transcription, independent of Q&A relevance.
    static let warningThreshold: Float = 0.70
    let confidence: Float
    static let contextThreshold: Float = 0.60
    let conflictingReadings: Bool
    let contextSupport: Float?
    public var uncertain: Bool {
        if conflictingReadings { return true }
        if confidence >= Self.warningThreshold { return false }
        guard let support = contextSupport, support >= Self.contextThreshold else { return true }
        return (7 * confidence + 3 * support) / 10 < Self.contextThreshold
    }
    /// An independent context quote may support a weak reading only when local readings do not conflict.
    public var canReviewContext: Bool { uncertain && !conflictingReadings }
    // Vision coordinates (origin at the bottom left) keep neighboring table cells distinct in Q&A.
    let bounds: CGRect?
    public var spoken: String { uncertain ? "인식 불확실. " + text : text }

    init(id: Int, text: String, confidence: Float, bounds: CGRect? = nil,
         conflictingReadings: Bool = false, contextSupport: Float? = nil) {
        self.id = id
        self.text = text
        self.confidence = confidence.isFinite ? min(1, max(0, confidence)) : 0
        self.bounds = bounds
        self.conflictingReadings = conflictingReadings
        self.contextSupport = contextSupport
    }

    init(id: Int, text: String, uncertain: Bool, bounds: CGRect? = nil) {
        self.init(id: id, text: text, confidence: uncertain ? 0 : Self.warningThreshold, bounds: bounds)
    }

    func withText(_ text: String) -> Self {
        Self(id: id, text: text, confidence: confidence, bounds: bounds,
             conflictingReadings: conflictingReadings, contextSupport: contextSupport)
    }

    func corroborated(by score: Float) -> Self {
        guard !conflictingReadings, score.isFinite, (Self.contextThreshold...1).contains(score),
              score > (contextSupport ?? 0) else { return self }
        return Self(id: id, text: text, confidence: confidence, bounds: bounds,
                    conflictingReadings: false, contextSupport: score)
    }
}

struct OCRReading {
    let text: String
    let confidence: Float
}

struct LocatedLine {
    let value: PrintedLine
    let bounds: CGRect
    let second: OCRReading?

    func reviewed() -> PrintedLine {
        DocumentReviewer.reviewLine(value, second: second)
    }
}

public struct ReviewedPage: Sendable {
    public let lines: [PrintedLine]
    public let notice: String
}

public enum PhotoReader {
    struct Page {
        let lines: [LocatedLine]
        let upright: CGImage
        let bounds: CGRect
    }

    /// Preserve the original still and EXIF orientation for the first OCR pass.
    public static func read(_ data: Data) throws -> [PrintedLine] {
        try scan(data).lines.map { $0.reviewed() }
    }

    static func scan(_ data: Data) throws -> Page {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let upright = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true,
                  kCGImageSourceThumbnailMaxPixelSize: 2200
              ] as CFDictionary) else { throw CocoaError(.fileReadCorruptFile) }
        let raw = (properties[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
        let orientation = CGImagePropertyOrientation(rawValue: raw) ?? .up
        let handler = VNImageRequestHandler(data: data, orientation: orientation, options: [:])
        let request = textRequest(corrected: false)
        try handler.perform([request])
        try Task.checkCancellation()
        let document = VNDetectDocumentSegmentationRequest()
        try? VNImageRequestHandler(cgImage: upright, options: [:]).perform([document])
        var boundary: CGPath?
        var bounds = CGRect(x: 0, y: 0, width: 1, height: 1)
        if let page = document.results?.first, page.confidence >= 0.7,
           page.boundingBox.width * page.boundingBox.height > 0.1 {
            let path = CGMutablePath()
            path.addLines(between: [page.topLeft, page.topRight, page.bottomRight, page.bottomLeft])
            path.closeSubpath()
            boundary = path
            bounds = page.boundingBox
        }
        let observations = request.results ?? []
        // Compare independent Vision reads at two scales before publishing any line.
        let corrected = textRequest(corrected: true)
        if !observations.isEmpty {
            try? VNImageRequestHandler(cgImage: upright, options: [:]).perform([corrected])
        }
        try Task.checkCancellation()
        let secondPass = corrected.results ?? []
        let lines = observations.enumerated().compactMap { index, observation -> LocatedLine? in
            let rect = observation.boundingBox
            guard boundary?.contains(CGPoint(x: rect.midX, y: rect.midY)) ?? true,
                  let result = observation.topCandidates(1).first,
                  !result.string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
            let line = PrintedLine(id: index, text: result.string, confidence: result.confidence, bounds: rect)
            let matches = secondPass.filter { DocumentReviewer.sameLocation($0.boundingBox, rect) }
            // Ambiguous overlapping second-pass rows cannot corroborate a specific line.
            let candidate = matches.count == 1 ? matches[0].topCandidates(1).first : nil
            let second = candidate.map { OCRReading(text: $0.string, confidence: $0.confidence) }
            return LocatedLine(value: line, bounds: rect, second: second)
        }
        return Page(lines: lines, upright: upright, bounds: bounds)
    }

    static func textRequest(corrected: Bool) -> VNRecognizeTextRequest {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["ko-KR", "en-US"]
        request.usesLanguageCorrection = corrected
        request.minimumTextHeight = 0.003
        return request
    }

    /// Vision uses bottom-left normalized coordinates; CGImage crops use a top-left origin.
    static func crop(_ image: CGImage, to bounds: CGRect) -> CGImage? {
        let w = CGFloat(image.width), h = CGFloat(image.height)
        let rect = CGRect(x: bounds.minX * w, y: (1 - bounds.maxY) * h, width: bounds.width * w,
                          height: bounds.height * h).integral.intersection(CGRect(x: 0, y: 0, width: w, height: h))
        return rect.isEmpty ? nil : image.cropping(to: rect)
    }
}

enum PaperAppearance {
    case blankCandidate, lightProblem, other

    static func inspect(_ image: CGImage) -> Self {
        let size = 256
        var values = [UInt8](repeating: 0, count: size * size)
        let drawn = values.withUnsafeMutableBytes { bytes -> Bool in
            guard let ctx = CGContext(data: bytes.baseAddress, width: size, height: size, bitsPerComponent: 8,
                                      bytesPerRow: size, space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            ctx.interpolationQuality = .medium
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
            return true
        }
        guard drawn else { return .other }
        let sorted = values.sorted()
        let median = Int(sorted[sorted.count / 2])
        let glare = sorted[sorted.count * 5 / 6] >= 252 && median < 215
        if median < 65 || median >= 252 || glare { return .lightProblem }
        guard median >= 140 else { return .other }
        // Count local strokes as well as dark ink. A smooth illumination gradient is common on blank paper.
        var marks = 0
        for y in 2..<(size - 2) {
            for x in 2..<(size - 2) {
                let i = y * size + x, pixel = Int(values[i])
                let neighbors = (Int(values[i-2]) + Int(values[i+2]) + Int(values[i-2*size]) + Int(values[i+2*size])) / 4
                if pixel < median - 65 || neighbors - pixel > 13 { marks += 1 }
            }
        }
        return marks <= 12 ? .blankCandidate : .other
    }
}

/// The captured image stays inside Vision and the local page-quality check.
public actor DocumentReviewer {
    public init() {}

    public func read(_ data: Data) throws -> ReviewedPage {
        try Task.checkCancellation()
        return try autoreleasepool {
            let page = try PhotoReader.scan(data)
            let lines = page.lines.map { $0.reviewed() }
            if !lines.isEmpty { return ReviewedPage(lines: lines, notice: Self.countNotice(lines)) }
            let cropped = PhotoReader.crop(page.upright, to: page.bounds) ?? page.upright
            let inner = CGRect(x: 0.04, y: 0.04, width: 0.92, height: 0.92)
            let appearance = PaperAppearance.inspect(PhotoReader.crop(cropped, to: inner) ?? cropped)
            return ReviewedPage(lines: [], notice: Self.emptyNotice(appearance: appearance))
        }
    }

    static func sameLocation(_ a: CGRect, _ b: CGRect) -> Bool {
        let overlap = a.intersection(b)
        return !overlap.isNull && overlap.height >= 0.55 * max(a.height, b.height)
            && overlap.width >= 0.65 * max(a.width, b.width)
    }

    static func reviewLine(_ line: PrintedLine, second: OCRReading?) -> PrintedLine {
        guard let second else { return line }
        func normalized(_ text: String) -> String {
            // Ignore OCR word-spacing changes in Korean, but do not merge separate numbers.
            var result = ""
            for word in text.precomposedStringWithCanonicalMapping.split(whereSeparator: \.isWhitespace) {
                if let left = result.last, let right = word.first,
                   (left.isNumber && right.isNumber || left.isASCII && right.isASCII && left.isLetter && right.isLetter) {
                    result.append(" ")
                }
                result.append(contentsOf: word)
            }
            return result
        }
        let score: Float
        let disagree = normalized(second.text) != normalized(line.text)
        if !disagree {
            // This is an agreement heuristic, not a calibrated probability.
            score = max(line.confidence, (line.confidence + second.confidence) / 2 + 0.20)
        } else {
            score = min(line.confidence, PrintedLine.warningThreshold.nextDown)
        }
        return PrintedLine(id: line.id, text: line.text, confidence: score, bounds: line.bounds, conflictingReadings: disagree)
    }

    static func emptyNotice(appearance: PaperAppearance) -> String {
        if appearance == .blankCandidate {
            return "글자를 찾지 못했고 이 면은 비어 보여요. 뒷면일 수 있으니 종이를 뒤집어 반대쪽을 확인해 주세요."
        }
        if appearance == .lightProblem {
            return "글자를 읽기 어려워요. 종이 위의 빛 반사나 그림자를 줄이고 다시 촬영해 주세요."
        }
        return "글자를 확인하기 어려워요. 종이의 방향과 초점을 확인하고 다시 촬영해 주세요."
    }

    public static func countNotice(_ lines: [PrintedLine]) -> String {
        let uncertain = lines.filter(\.uncertain).count
        return "\(lines.count)줄을 읽었어요." + (uncertain == 0 ? "" : " \(uncertain)줄은 인식이 불확실해요.")
    }
}
