import Foundation
import Vision

public struct PreviewPage {
    /// Top left, top right, bottom right, bottom left in upright Vision coordinates.
    public let corners: [CGPoint]
    public let bounds: CGRect

    init(_ corners: [CGPoint]) {
        self.corners = corners
        let xs = corners.map(\.x), ys = corners.map(\.y)
        bounds = CGRect(x: xs.min()!, y: ys.min()!, width: xs.max()! - xs.min()!, height: ys.max()! - ys.min()!)
    }
}

private struct PaperRegion {
    let page: PreviewPage
    let cut: FramingCorrection?
    let area: Double

    /// Select connected paper-colored surfaces, even when darker furniture touches
    /// the frame. The four sampled sides distinguish a sheet from an irregular patch.
    static func find(y: [Int], chroma: [Int], width w: Int, height h: Int) -> [Self] {
        let neutral = y.indices.compactMap { chroma[$0] <= 23 ? y[$0] : nil }.sorted()
        guard neutral.count >= y.count / 20 else { return [] }
        let threshold = max(86, neutral[neutral.count * 9 / 10] * 81 / 100)
        let inkGaps = y.indices.map { y[$0] >= threshold && chroma[$0] <= 23 }
        var mask = inkGaps
        for row in 1..<(h - 1) {
            for col in 1..<(w - 1) {
                let i = row * w + col
                if !inkGaps[i] {
                    mask[i] = (inkGaps[i - 1] && inkGaps[i + 1]) || (inkGaps[i - w] && inkGaps[i + w])
                }
            }
        }
        var pending = mask, result = [Self]()
        for seed in pending.indices where pending[seed] {
            var component = [seed], next = 0
            pending[seed] = false
            while next < component.count {
                let i = component[next], x = i % w
                next += 1
                for n in [x > 0 ? i - 1 : -1, x + 1 < w ? i + 1 : -1, i - w, i + w]
                    where pending.indices.contains(n) && pending[n] {
                    pending[n] = false
                    component.append(n)
                }
            }
            if let region = PaperRegion(component, mask: mask, width: w, height: h) { result.append(region) }
        }
        return result.sorted { $0.area > $1.area }
    }

    private init?(_ cells: [Int], mask: [Bool], width w: Int, height h: Int) {
        area = Double(cells.count) / Double(w * h)
        guard area >= 0.045, area <= 0.94 else { return nil }
        var xmin = w, xmax = 0, ymin = h, ymax = 0
        var left = 0, right = 0, top = 0, bottom = 0
        var extremes = [Int](repeating: .min, count: 4), tips = [CGPoint](repeating: .zero, count: 4)
        for i in cells {
            let x = i % w, y = i / w
            xmin = min(xmin, x); xmax = max(xmax, x); ymin = min(ymin, y); ymax = max(ymax, y)
            if x == 0 { left += 1 }
            if x == w - 1 { right += 1 }
            if y == 0 { top += 1 }
            if y == h - 1 { bottom += 1 }
            let scores = [-x-y, x-y, x+y, y-x]
            for k in scores.indices where scores[k] > extremes[k] {
                extremes[k] = scores[k]
                tips[k] = CGPoint(x: CGFloat(x) + 0.5, y: CGFloat(y) + 0.5)
            }
        }
        let width = xmax - xmin + 1, height = ymax - ymin + 1
        guard width >= 4, height >= 5 else { return nil }
        func inMask(_ x: CGFloat, _ y: CGFloat) -> Bool {
            guard x >= 0, x < CGFloat(w), y >= 0, y < CGFloat(h) else { return false }
            return mask[Int(y) * w + Int(x)]
        }
        var doubleArea: CGFloat = 0, supportedSides = 0
        for side in tips.indices {
            let a = tips[side], b = tips[(side + 1) % 4]
            let dx = b.x - a.x, dy = b.y - a.y, length = hypot(dx, dy)
            guard length >= 3 else { return nil }
            doubleArea += a.x * b.y - b.x * a.y
            let nx = -dy / length, ny = dx / length
            var support = 0
            for t: CGFloat in [0.15, 0.32, 0.5, 0.68, 0.85] {
                let x = a.x + dx * t, y = a.y + dy * t
                if inMask(x + 1.6 * nx, y + 1.6 * ny) && !inMask(x - 1.8 * nx, y - 1.8 * ny) { support += 1 }
            }
            if support >= 3 { supportedSides += 1 }
        }
        let coverage = abs(doubleArea) / (2 * CGFloat(cells.count))
        guard (0.70...1.30).contains(coverage) else { return nil }
        cut = FramingCorrection.fromEdges(left: left >= max(2, height / 7), right: right >= max(2, height / 7),
                                          top: top >= max(2, width / 7), bottom: bottom >= max(2, width / 7),
                                          width: Double(width) / Double(w), height: Double(height) / Double(h))
        if cut == nil {
            guard supportedSides == 4, xmin > 0, ymin > 0, xmax < w - 1, ymax < h - 1, area >= 0.075 else { return nil }
        } else if supportedSides < 3 { return nil }
        page = PreviewPage(tips.map { CGPoint(x: $0.x / CGFloat(w), y: 1 - $0.y / CGFloat(h)) })
    }
}

/// Cheap evidence from the actual candidate surface. A high-confidence rectangle
/// alone also describes furniture, bags and screens.
struct PreviewPaperEvidence {
    let hasPaperBackground: Bool
    let hasDistinctEdges: Bool

    init?(corners: [CGPoint], luma: (CGPoint) -> Int?) {
        guard corners.count == 4 else { return nil }
        func between(_ a: CGPoint, _ b: CGPoint, _ t: CGFloat) -> CGPoint {
            CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t)
        }
        let center = between(between(corners[0], corners[1], 0.5), between(corners[3], corners[2], 0.5), 0.5)
        var samples = [Int]()
        for row in 1...7 {
            let v = CGFloat(row) / 8
            let left = between(corners[0], corners[3], v), right = between(corners[1], corners[2], v)
            for col in 1...7 {
                if let value = luma(between(left, right, CGFloat(col) / 8)) { samples.append(value) }
            }
        }
        guard samples.count >= 24 else { return nil }
        samples.sort()
        let paperTone = samples[samples.count * 7 / 10]
        let spread = samples[samples.count * 5 / 6] - samples[samples.count / 2]
        var distinctSides = 0, brighterSides = 0
        for side in corners.indices {
            var contrast = 0, brighter = 0
            for t: CGFloat in [0.25, 0.5, 0.75] {
                let edge = between(corners[side], corners[(side + 1) % 4], t)
                guard let inside = luma(between(edge, center, 0.06)),
                      let outside = luma(between(edge, center, -0.06)) else { continue }
                if abs(inside - outside) > 12 { contrast += 1 }
                if inside - outside > 16 { brighter += 1 }
            }
            if contrast >= 2 { distinctSides += 1 }
            if brighter >= 2 { brighterSides += 1 }
        }
        hasDistinctEdges = distinctSides >= 3
        // Blank backs still work. In dim light, compare the page to its surroundings.
        hasPaperBackground = (paperTone >= 132 && spread <= 76)
            || (paperTone >= 76 && spread <= 47 && brighterSides >= 2)
    }
}

public enum PreviewDocumentDetector {
    /// Samples and the image share upright Vision coordinates. Chroma is the larger
    /// absolute difference of Cb or Cr from neutral (128), on the camera's 8-bit scale.
    public static func observe(in image: CGImage, sample: (CGPoint) -> (luma: Int, chroma: Int)?)
            -> (page: PreviewPage?, partial: FramingCorrection?) {
        let w = 48, h = 64
        var values = [Int](), colors = [Int]()
        values.reserveCapacity(w * h); colors.reserveCapacity(w * h)
        for y in 0..<h {
            for x in 0..<w {
                let point = CGPoint(x: (CGFloat(x) + 0.5) / CGFloat(w), y: 1 - (CGFloat(y) + 0.5) / CGFloat(h))
                let pixel = sample(point)
                values.append(pixel?.luma ?? 0)
                colors.append(pixel?.chroma ?? 128)
            }
        }
        let regions = PaperRegion.find(y: values, chroma: colors, width: w, height: h)
        // An obvious cut edge supersedes any text/table rectangle on the visible
        // half of a page. Direction feedback does not wait for a Vision pass.
        if let first = regions.first, let cut = first.cut, first.area >= 0.085,
           first.area > (regions.dropFirst().first?.area ?? 0) * 1.5 { return (nil, cut) }
        if let vision = visionPage(in: image, luma: { sample($0)?.luma }) {
            let page = PreviewPage([vision.topLeft, vision.topRight, vision.bottomRight, vision.bottomLeft])
            let pageArea = page.bounds.width * page.bounds.height
            for region in regions {
                let overlap = region.page.bounds.intersection(page.bounds)
                guard !overlap.isNull else { continue }
                let shared = overlap.width * overlap.height
                let regionArea = region.page.bounds.width * region.page.bounds.height
                if let cut = region.cut, shared >= min(pageArea, regionArea) * 0.65 { return (nil, cut) }
                // Vision can select a rectangular print block inside a sheet.
                if region.cut == nil, shared >= pageArea * 0.92, regionArea > pageArea * 1.24 { return (region.page, nil) }
            }
            if let cut = edgeCorrection(for: page, sample: sample) { return (nil, cut) }
            return (page, nil)
        }
        if let region = regions.first {
            if let cut = region.cut { return (nil, cut) }
            return (region.page, nil)
        }
        return (nil, FramingCorrection.partialPage(luma: values.map { UInt8(clamping: $0) }, width: w, height: h))
    }

    // Vision sometimes fits a side a few pixels *inside* the preview when the
    // physical paper continues off screen. A visible strip of desk disproves that.
    static func edgeCorrection(for page: PreviewPage, sample: (CGPoint) -> (luma: Int, chroma: Int)?) -> FramingCorrection? {
        if let cut = FramingCorrection.clipped(page.corners) { return cut }
        let rect = page.bounds, margin: CGFloat = 0.023
        guard rect.minX <= margin || rect.maxX >= 1 - margin || rect.minY <= margin || rect.maxY >= 1 - margin else { return nil }
        var left = false, right = false, top = false, bottom = false
        for i in page.corners.indices {
            let a = page.corners[i], b = page.corners[(i + 1) % 4]
            let atLeft = max(a.x, b.x) <= margin, atRight = min(a.x, b.x) >= 1 - margin
            let atBottom = max(a.y, b.y) <= margin, atTop = min(a.y, b.y) >= 1 - margin
            guard atLeft || atRight || atTop || atBottom else { continue }
            var continuous = 0
            for t: CGFloat in [0.15, 0.33, 0.5, 0.67, 0.85] {
                let x = a.x + (b.x - a.x) * t, y = a.y + (b.y - a.y) * t
                let atEdge: CGPoint, inside: CGPoint
                if atLeft || atRight {
                    atEdge = CGPoint(x: atLeft ? 0.001 : 0.999, y: y)
                    inside = CGPoint(x: x + (atLeft ? 0.018 : -0.018), y: y)
                } else {
                    atEdge = CGPoint(x: x, y: atBottom ? 0.001 : 0.999)
                    inside = CGPoint(x: x, y: y + (atBottom ? 0.018 : -0.018))
                }
                if let edge = sample(atEdge), let inner = sample(inside),
                   abs(edge.luma - inner.luma) < 22, abs(edge.chroma - inner.chroma) < 13 { continuous += 1 }
            }
            if continuous >= 3 {
                left = left || atLeft; right = right || atRight
                top = top || atTop; bottom = bottom || atBottom
            }
        }
        return FramingCorrection.fromEdges(left: left, right: right, top: top, bottom: bottom,
                                           width: rect.width, height: rect.height)
    }

    private static func visionPage(in image: CGImage, luma: (CGPoint) -> Int?) -> VNRectangleObservation? {
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        var textLines: [CGRect]?
        func supported(_ page: VNRectangleObservation, requiresEdges: Bool) -> Bool {
            let corners = [page.topLeft, page.topRight, page.bottomRight, page.bottomLeft]
            guard let surface = PreviewPaperEvidence(corners: corners, luma: luma),
                  !requiresEdges || surface.hasDistinctEdges else { return false }
            if surface.hasPaperBackground { return true }
            // A dark or illustrated document can instead show lines of print.
            // Calculate them at most once per frame, only for ambiguous surfaces.
            if textLines == nil {
                let text = VNDetectTextRectanglesRequest()
                try? handler.perform([text])
                textLines = text.results?.map(\.boundingBox) ?? []
            }
            let path = CGMutablePath()
            path.addLines(between: corners)
            path.closeSubpath()
            let width = page.boundingBox.width
            var lineCount = 0
            for line in textLines ?? [] where line.width >= 2 * line.height && line.width >= width * 0.10 {
                guard path.contains(CGPoint(x: line.midX - 0.4 * line.width, y: line.midY)),
                      path.contains(CGPoint(x: line.midX + 0.4 * line.width, y: line.midY)) else { continue }
                lineCount += 1
                if line.width >= width * 0.32 || lineCount >= 2 { return true }
            }
            return false
        }
        let document = VNDetectDocumentSegmentationRequest()
        try? handler.perform([document])
        if let page = document.results?.first, page.confidence >= 0.7,
           page.boundingBox.width * page.boundingBox.height > 0.1,
           supported(page, requiresEdges: false) { return page }
        let rectangles = VNDetectRectanglesRequest()
        rectangles.minimumAspectRatio = 0.28
        rectangles.minimumSize = 0.35
        rectangles.maximumObservations = 3
        try? handler.perform([rectangles])
        return rectangles.results?.first { page in
            page.confidence >= 0.8 && page.boundingBox.width * page.boundingBox.height > 0.14
                && supported(page, requiresEdges: true)
        }
    }
}
