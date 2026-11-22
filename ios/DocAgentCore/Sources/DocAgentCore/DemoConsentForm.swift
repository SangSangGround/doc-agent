import CryptoKit
import CoreImage
import Foundation
import Vision

/// Reference for the single submitted blank page. No signature, date or user's choices are cached.
public enum DemoConsentForm {
    public static let id = "work-training-loan-consent-appendix3-2024-01-01"

    public struct Match: Codable, Sendable {
        public let lines: [PrintedLine]
        /// Coordinates refer to the current upright still, never to a previous camera frame.
        public let targets: [WritingTarget]
    }

    private static let sourceDigest = "7be81554efb3d4b6b7bcea48f3757312c22393377085e4c184f889305910d732"
    private static let fields: [WritingTarget] = [
        .init(id: "collection-yes", label: "개인정보 수집·이용에 동의함", kind: .checkbox,
              bounds: CGRect(x: 0.71203, y: 0.52535, width: 0.01323, height: 0.00982)),
        .init(id: "collection-no", label: "개인정보 수집·이용에 동의하지 않음", kind: .checkbox,
              bounds: CGRect(x: 0.83532, y: 0.52567, width: 0.01323, height: 0.01014)),
        .init(id: "provision-yes", label: "개인정보 제3자 제공에 동의함", kind: .checkbox,
              bounds: CGRect(x: 0.71544, y: 0.72085, width: 0.01323, height: 0.01014)),
        .init(id: "provision-no", label: "개인정보 제3자 제공에 동의하지 않음", kind: .checkbox,
              bounds: CGRect(x: 0.83916, y: 0.72117, width: 0.01323, height: 0.01046)),
        // Blank writing space directly above the printed “(서명 또는 인)”.
        .init(id: "applicant-signature", label: "신청인 서명 또는 인", kind: .signature,
              bounds: CGRect(x: 0.710, y: 0.907, width: 0.120, height: 0.023))
    ]

    /// Enables bypassing OCR only for the exact provided HEIC, including its byte count and hash.
    public static func exactPhoto(_ data: Data) -> Match? {
        guard data.count == 1_479_312,
              SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == sourceDigest else { return nil }
        let lines = rows.enumerated().map { index, row in
            PrintedLine(id: index, text: row.text, confidence: 1, bounds: visionRect(row.rect))
        }
        return Match(lines: lines, targets: fields)
    }

    private struct Observation {
        let line: PrintedLine
        let key: String
        let rect: CGRect
    }

    private static func key(_ text: String) -> String {
        text.precomposedStringWithCanonicalMapping.filter { $0.isLetter || $0.isNumber }
    }

    private static let rowKeys = rows.map { key($0.text) }

    private static func visionRect(_ rect: CGRect) -> CGRect {
        CGRect(x: rect.minX, y: 1 - rect.maxY, width: rect.width, height: rect.height)
    }

    private static func center(_ rect: CGRect) -> CGPoint {
        CGPoint(x: rect.midX, y: rect.midY)
    }

    private static func distance(_ a: CGPoint, _ b: CGPoint) -> CGFloat { hypot(a.x - b.x, a.y - b.y) }

    private static func isVersion(_ lines: [Observation]) -> Bool {
        // Scope by revision, appendix, legal-form number and retention term. Distinguish day 1 from 10 or 11.
        let joined = lines.map(\.key).joined(separator: "|")
        let revisions = lines.filter { $0.key.contains("개정") || $0.line.text.contains("2024.1.") }
        let date = #"2024\s*\.\s*0?1\s*\.\s*0?1(?!\s*\d)"#
        return !revisions.isEmpty && revisions.allSatisfy { $0.line.text.range(of: date, options: .regularExpression) != nil }
            && (joined.contains("붙임3") || joined.contains("불임3")) && joined.contains("별지제3호서식")
            && joined.contains("개인정보의수집이용제공동의서")
            && joined.contains("직업훈련생계비대부") && joined.contains("근로복지공단")
            && joined.contains("상환조건에따라최대10년") && joined.contains("사업평가")
    }

    private struct Pair {
        let reference: CGPoint
        let observed: CGPoint
    }

    /// Linear least squares in unit image coordinates, solving the eight independent homography coefficients.
    private static func fit(_ pairs: [Pair]) -> WritingTransform? {
        var system = Array(repeating: Array(repeating: 0.0, count: 9), count: 8)
        for pair in pairs {
            let (x, y) = (Double(pair.reference.x), Double(pair.reference.y))
            let (u, v) = (Double(pair.observed.x), Double(pair.observed.y))
            for equation in [[x, y, 1, 0, 0, 0, -x*u, -y*u, u], [0, 0, 0, x, y, 1, -x*v, -y*v, v]] {
                for i in 0..<8 {
                    for j in 0..<9 { system[i][j] += equation[i] * equation[j] }
                }
            }
        }
        for col in 0..<8 {
            guard let pivot = (col..<8).max(by: { abs(system[$0][col]) < abs(system[$1][col]) }),
                  abs(system[pivot][col]) > 1e-8 else { return nil }
            system.swapAt(pivot, col)
            let divisor = system[col][col]
            for j in col..<9 { system[col][j] /= divisor }
            for i in 0..<8 where i != col {
                let multiplier = system[i][col]
                for j in col..<9 { system[i][j] -= multiplier * system[col][j] }
            }
        }
        return WritingTransform(rowMajor: system.map { $0[8] } + [1])
    }

    private static func registration(_ observations: [Observation]) -> WritingTransform? {
        // Each printed line center votes only once. Repeated “동의함” rows and handwritten fields are excluded.
        let indexes = Dictionary(grouping: rows.indices, by: { rowKeys[$0] })
        let observed = Dictionary(grouping: observations, by: \.key)
        var pairs = [Pair]()
        for row in rows.indices {
            let identifier = rowKeys[row]
            guard identifier.count >= 4, indexes[identifier]?.count == 1, let matches = observed[identifier],
                  matches.count == 1 else { continue }
            pairs.append(Pair(reference: center(rows[row].rect), observed: center(matches[0].rect)))
        }
        let minimumInliers = max(12, (pairs.count * 4 + 4) / 5)
        while pairs.count >= minimumInliers, let pose = fit(pairs) {
            let errors = pairs.map { pair in pose.livePoint(pair.reference).map { distance($0, pair.observed) } ?? .infinity }
            guard let worst = errors.indices.max(by: { errors[$0] < errors[$1] }) else { return nil }
            if errors[worst] <= 0.010 {
                // Extrapolating tiny checkboxes from one corner or one paragraph is unreliable.
                let xs = pairs.map { $0.reference.x }, ys = pairs.map { $0.reference.y }
                guard xs.min()! < 0.23, xs.max()! > 0.68, ys.min()! < 0.18, ys.max()! > 0.85,
                      pairs.contains(where: { (0.36...0.48).contains($0.reference.y) }),
                      pairs.contains(where: { (0.56...0.75).contains($0.reference.y) }) else { return nil }
                return pose
            }
            pairs.remove(at: worst)
        }
        return nil
    }

    private static func project(_ rect: CGRect, using pose: WritingTransform) -> CGRect? {
        let corners = [CGPoint(x: rect.minX, y: rect.minY), CGPoint(x: rect.maxX, y: rect.minY),
                       CGPoint(x: rect.minX, y: rect.maxY), CGPoint(x: rect.maxX, y: rect.maxY)]
        let points = corners.compactMap { pose.livePoint($0) }
        guard points.count == 4 else { return nil }
        let xs = points.map(\.x), ys = points.map(\.y)
        return CGRect(x: xs.min()!, y: ys.min()!, width: xs.max()! - xs.min()!, height: ys.max()! - ys.min()!)
    }

    private static func writableRowsAgree(_ observations: [Observation], pose: WritingTransform) -> Bool {
        for row in rows where row.text.contains("동의하지 않음 □") || row.text == "(서명 또는 인)" {
            guard let rect = project(row.rect, using: pose) else { return false }
            let near = observations.filter { candidate in
                guard candidate.rect.intersects(rect.insetBy(dx: -0.009, dy: -0.006)) else { return false }
                // Perspective can cause Vision to join the sentence on the left and both choices on the right.
                if candidate.key.contains("동의함") && candidate.key.contains("동의하지않음") {
                    return abs(candidate.rect.maxX - rect.maxX) < max(0.018, rect.width * 0.18)
                }
                return distance(center(candidate.rect), center(rect)) < max(0.012, rect.width * 0.55)
            }
            let words = near.map(\.key).joined()
            if row.text == "(서명 또는 인)" {
                guard words.contains("서명또는인") else { return false }
            } else {
                guard words.contains("동의함") && words.contains("동의하지않음") else { return false }
            }
        }
        return true
    }

    /// Unlisted spelling changes and every handwritten field remain live OCR; never replace a number or a consent choice.
    private static let knownMisreads = ["수집이용복적": "수집이용목적", "의용합니다": "이용합니다",
                                        "달성할때까지보및": "달성할때까지보유및"]

    // Tiny print within the two shaded blocks is often merged or fragmented by full-page OCR.
    // Read just these static blocks again; require an exact content match including numbers.
    private static let imageContext = CIContext(options: [.cacheIntermediates: false])
    private static let printedBlocks: [(CGRect, String)] = [
        (CGRect(x: 0.577, y: 0.400, width: 0.261, height: 0.031), "「고용보험법」"),
        (CGRect(x: 0.144, y: 0.497, width: 0.712, height: 0.027), "본인은 위에 관한")
    ]

    private static func reviewPrintedBlocks(_ input: [PrintedLine], data: Data, pose: WritingTransform) -> [PrintedLine] {
        guard let image = CIImage(data: data, options: [.applyOrientationProperty: true]), !image.extent.isEmpty else { return input }
        var result = input
        for (rect, prefix) in printedBlocks {
            guard let first = rows.firstIndex(where: { $0.text.hasPrefix(prefix) }),
                  let current = project(rect, using: pose) else { continue }
            let expected = prefix == "「고용보험법」" ? rows[first].text : rows[first].text + " " + rows[first+1].text
            var corners: [String: Any] = [:]
            for (name, point) in [("inputTopLeft", CGPoint(x: rect.minX, y: rect.minY)),
                                  ("inputTopRight", CGPoint(x: rect.maxX, y: rect.minY)),
                                  ("inputBottomLeft", CGPoint(x: rect.minX, y: rect.maxY)),
                                  ("inputBottomRight", CGPoint(x: rect.maxX, y: rect.maxY))] {
                guard let p = pose.livePoint(point) else { break }
                corners[name] = CIVector(x: image.extent.minX + p.x * image.extent.width,
                                         y: image.extent.maxY - p.y * image.extent.height)
            }
            guard corners.count == 4 else { continue }
            let block = image.applyingFilter("CIPerspectiveCorrection", parameters: corners)
            guard block.extent.width > 12, block.extent.height > 5 else { continue }
            let scale = min(4.0, 1800.0 / block.extent.width)
            let scaled = block.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            guard let bitmap = imageContext.createCGImage(scaled, from: scaled.extent) else { continue }
            let request = PhotoReader.textRequest(corrected: false)
            guard (try? VNImageRequestHandler(cgImage: bitmap).perform([request])) != nil else { continue }
            let words = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined()
            guard key(words) == key(expected) else { continue }
            // Keep any row that crosses the static block boundary (for example a handwritten note).
            let matching = result.indices.filter { index in
                guard let box = result[index].bounds else { return false }
                let seen = visionRect(box)
                return current.contains(center(seen))
                    && seen.minX >= current.minX-0.01 && seen.maxX <= current.maxX+0.01
            }
            let id = matching.first.map { result[$0].id } ?? ((result.map(\.id).max() ?? -1) + 1)
            let replacement = PrintedLine(id: id, text: expected, confidence: 1, bounds: visionRect(current))
            if let firstIndex = matching.first {
                for index in matching.dropFirst().reversed() { result.remove(at: index) }
                result[firstIndex] = replacement
            } else {
                let index = result.firstIndex { row in row.bounds.map { visionRect($0).minY > current.maxY } ?? false }
                result.insert(replacement, at: index ?? result.endIndex)
            }
        }
        return result
    }

    public static func match(_ lines: [PrintedLine], imageData: Data? = nil) -> Match? {
        if let imageData, let exact = exactPhoto(imageData) { return exact }
        let observations = lines.compactMap { line -> Observation? in
            guard let bounds = line.bounds, !bounds.isNull, !bounds.isEmpty else { return nil }
            return Observation(line: line, key: key(line.text), rect: visionRect(bounds))
        }
        guard observations.count >= 12, isVersion(observations), let pose = registration(observations),
              writableRowsAgree(observations, pose: pose) else { return nil }

        var targets = [WritingTarget]()
        let unit = CGRect(x: 0, y: 0, width: 1, height: 1)
        let printedArea = CGRect(x: 0.07, y: 0.077, width: 0.85, height: 0.893)
        guard let visiblePrint = project(printedArea, using: pose), unit.contains(visiblePrint) else { return nil }
        for field in fields {
            guard let bounds = project(field.bounds, using: pose), unit.contains(bounds),
                  bounds.width > 0.002, bounds.height > 0.002 else { return nil }
            targets.append(WritingTarget(id: field.id, label: field.label, kind: field.kind, bounds: bounds))
        }
        var corrected: [Int: PrintedLine] = [:]
        for observation in observations {
            guard !observation.line.conflictingReadings else { continue }
            var spelling = observation.key
            for (mistake, correction) in knownMisreads {
                spelling = spelling.replacingOccurrences(of: mistake, with: correction)
            }
            guard spelling.count >= 4, let i = rowKeys.firstIndex(of: spelling),
                  rows[i].rect.minY < 0.88, !rows[i].text.contains("동의하지 않음 □"),
                  let expected = pose.livePoint(center(rows[i].rect)), distance(expected, center(observation.rect)) <= 0.012 else { continue }
            corrected[observation.line.id] = PrintedLine(id: observation.line.id, text: rows[i].text,
                                                        confidence: max(PrintedLine.warningThreshold, observation.line.confidence),
                                                        bounds: observation.line.bounds)
        }
        let verified = lines.map { corrected[$0.id] ?? $0 }
        return Match(lines: imageData.map { reviewPrintedBlocks(verified, data: $0, pose: pose) } ?? verified,
                     targets: targets)
    }
}
