import Foundation
import ImageIO
import Vision

public struct PrintedLine: Identifiable, Sendable {
    public let id: Int
    public let text: String
    public let uncertain: Bool
    public var spoken: String { uncertain ? "인식 불확실. " + text : text }
}

struct LocatedLine {
    let value: PrintedLine
    let bounds: CGRect
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
        try scan(data).lines.map(\.value)
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
        let lines = (request.results ?? []).enumerated().compactMap { index, observation -> LocatedLine? in
            let rect = observation.boundingBox
            guard boundary?.contains(CGPoint(x: rect.midX, y: rect.midY)) ?? true,
                  let result = observation.topCandidates(1).first,
                  !result.string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
            let line = PrintedLine(id: index, text: result.string, uncertain: result.confidence < 0.7)
            return LocatedLine(value: line, bounds: rect)
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

/// Serializes recognition so a cancelled photo cannot compete with its replacement for model memory.
public actor DocumentReviewer {
    public init() {}

    public func read(_ data: Data, onOCR: @Sendable ([PrintedLine]) async -> Void) async throws -> ReviewedPage {
        try Task.checkCancellation()
        let page = try autoreleasepool { try PhotoReader.scan(data) }
        try Task.checkCancellation()
        await onOCR(page.lines.map(\.value))
        try Task.checkCancellation()
        let cancellation = ReviewCancellation()
        return try await withTaskCancellationHandler {
            try autoreleasepool { try review(page, cancellation: cancellation) }
        } onCancel: {
            cancellation.cancel()
        }
    }

    private func review(_ page: PhotoReader.Page, cancellation: ReviewCancellation) throws -> ReviewedPage {
        let originals = page.lines.map(\.value)
        do {
            let reviewer = try VisualReview(cancellation: cancellation)
            try Task.checkCancellation()
            if originals.isEmpty {
                let pageImage = PhotoReader.crop(page.upright, to: page.bounds) ?? page.upright
                let vote = try reviewer.blankPage(pageImage)
                let inner = CGRect(x: 0.04, y: 0.04, width: 0.92, height: 0.92)
                let appearance = PaperAppearance.inspect(PhotoReader.crop(pageImage, to: inner) ?? pageImage)
                return ReviewedPage(lines: [], notice: Self.emptyNotice(appearance: appearance, vote: vote))
            }
            // A second OCR reading at a different scale helps distinguish a genuinely legible low-confidence line.
            let second = PhotoReader.textRequest(corrected: true)
            try? VNImageRequestHandler(cgImage: page.upright, options: [:]).perform([second])
            let secondPass = second.results ?? []
            var lines = originals
            // Work at crop scale: small print may disappear in a whole-page VLM preview.
            for start in stride(from: 0, to: page.lines.count, by: 4) {
                try Task.checkCancellation()
                let block = page.lines[start..<min(start + 4, page.lines.count)]
                let bounds = block.reduce(CGRect.null) { $0.union($1.bounds) }
                let padded = bounds.insetBy(dx: -0.012, dy: -max(0.006, bounds.height * 0.08))
                guard let crop = PhotoReader.crop(page.upright, to: padded) else { continue }
                let vote = try reviewer.matches(block.map { $0.value.text }, in: crop)
                for index in block.indices {
                    let line = page.lines[index]
                    let candidates = secondPass.filter { Self.sameLocation($0.boundingBox, line.bounds) }
                        .compactMap { $0.topCandidates(1).first?.string }
                    lines[index] = Self.reviewLine(line.value, candidates: candidates, vote: vote)
                }
            }
            return ReviewedPage(lines: lines, notice: Self.countNotice(lines))
        } catch {
            try Task.checkCancellation()
            let notice = originals.isEmpty ? "글자를 읽지 못했어요. 조명과 초점을 확인하고 다시 촬영해 주세요." : Self.countNotice(originals)
            return ReviewedPage(lines: originals, notice: notice + " 사진의 AI 재확인을 마치지 못했어요.")
        }
    }

    static func sameLocation(_ a: CGRect, _ b: CGRect) -> Bool {
        let overlap = a.intersection(b)
        return !overlap.isNull && overlap.height >= 0.55 * max(a.height, b.height)
            && overlap.width >= 0.65 * max(a.width, b.width)
    }

    static func reviewLine(_ line: PrintedLine, candidates: [String], vote: VisualVote) -> PrintedLine {
        func normalized(_ text: String) -> String {
            text.precomposedStringWithCanonicalMapping.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        }
        let matches = candidates.contains { normalized($0) == normalized(line.text) }
        var uncertain = line.uncertain
        if vote == .yes && matches { uncertain = false }
        if vote == .no && !candidates.isEmpty && !matches { uncertain = true }
        return PrintedLine(id: line.id, text: line.text, uncertain: uncertain)
    }

    static func emptyNotice(appearance: PaperAppearance, vote: VisualVote) -> String {
        if appearance == .blankCandidate && vote == .yes {
            return "글자를 찾지 못했고 이 면은 비어 보여요. 뒷면일 수 있으니 종이를 뒤집어 반대쪽을 확인해 주세요."
        }
        if appearance == .lightProblem {
            return "글자를 읽기 어려워요. 종이 위의 빛 반사나 그림자를 줄이고 다시 촬영해 주세요."
        }
        return "글자를 확인하기 어려워요. 종이의 방향과 초점을 확인하고 다시 촬영해 주세요."
    }

    private static func countNotice(_ lines: [PrintedLine]) -> String {
        let uncertain = lines.filter(\.uncertain).count
        return "\(lines.count)줄을 읽었어요." + (uncertain == 0 ? "" : " \(uncertain)줄은 인식이 불확실해요.")
    }
}
