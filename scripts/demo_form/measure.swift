// Compiled with the production document reader, form matcher and coordinate transform by evaluate.py.
import Foundation

private struct Sample: Decodable {
    let name: String
    let path: String
    let positive: Bool
    let matrix: [Double]?
    let width: Int
    let height: Int
}

private struct Result: Encodable {
    let name: String
    let positive: Bool
    let matched: Bool
    let lines: Int
    let ocrMs: Double
    let lookupMs: Double
    var sampledCharacters: Int?
    var rawEdits: Int?
    var matchedEdits: Int?
    var rawKeyPhrases: Int?
    var displayedKeyPhrases: Int?
    var centerErrorPx: [Double]?
    var centerInsideCorrectTarget: [Bool]?
    var centerInsideAnotherTarget: [Bool]?
}

@main private enum Measurement {
    // Human-checked printed lines spanning the title, legal basis, selection purpose, retention, recipients, refusal.
    // Match rows by known physical location, not by OCR text or by the transform inferred by the app.
    static let sampleKeys = ["개인정보의 수집", "1. 직업훈련생계비", "하거나 제3자", "가. 수집·이용 목적",
        "나. 수집·이용할 항목", "주민등록번호", "「고용보험법」", "이용합니다.(상환조건", "가. 제공받는 자",
        "▸ 고용노동부", "나. 제공받는 자의 이용 목적", "▪ 제공할 개인정보의 항목", "위 사항에 동의하셔야만"]
    static let keyPhrases = ["최대10년", "고용노동부및한국고용정보원", "제145조의2제1항제12호", "정부재정사업이중수혜방지",
                             "고유식별정보인적사항수혜내역", "위사항에동의하셔야만직업훈련생계비대부신청이가능합니다"]
    // Source-pixel annotations: the four outer square borders and blank area above the signature label.
    static let boxes = [CGRect(x: 1669, y: 1658, width: 31, height: 31), CGRect(x: 1958, y: 1659, width: 31, height: 32),
        CGRect(x: 1677, y: 2275, width: 31, height: 32), CGRect(x: 1967, y: 2276, width: 31, height: 33),
        CGRect(x: 1664, y: 2863, width: 281, height: 73)]

    static func errors(_ left: [Character], _ right: [Character]) -> Int {
        var previous = Array(0...right.count)
        for (i, c) in left.enumerated() {
            var current = [i + 1]
            for (j, d) in right.enumerated() {
                current.append(min(previous[j + 1] + 1, current[j] + 1, previous[j] + (c == d ? 0 : 1)))
            }
            previous = current
        }
        return previous.last!
    }

    static func normalized(_ text: String) -> [Character] {
        Array(text.precomposedStringWithCanonicalMapping.filter { $0.isLetter || $0.isNumber })
    }

    static func measure(_ sample: Sample) throws -> Result {
        let data = try Data(contentsOf: URL(fileURLWithPath: sample.path))
        let start = ProcessInfo.processInfo.systemUptime
        let raw = try PhotoReader.read(data)
        let afterOCR = ProcessInfo.processInfo.systemUptime
        let match = DemoConsentForm.match(raw, imageData: data)
        let end = ProcessInfo.processInfo.systemUptime
        var row = Result(name: sample.name, positive: sample.positive, matched: match != nil, lines: raw.count,
                         ocrMs: (afterOCR-start)*1000, lookupMs: (end-afterOCR)*1000)
        guard sample.positive, let matrix = sample.matrix, let truthPose = WritingTransform(rowMajor: matrix) else { return row }
        let displayed = match?.lines ?? raw
        var characterCount = 0, rawErrors = 0, shownErrors = 0
        for key in sampleKeys {
            let truth = DemoConsentForm.rows.first { $0.text.hasPrefix(key) }!
            let p = truthPose.livePoint(CGPoint(x: truth.rect.midX, y: truth.rect.midY))!
            func closest(_ lines: [PrintedLine]) -> String {
                // The same legal cell may be a single Vision observation or several fragments.
                if key == "「고용보험법」" {
                    let cell = CGRect(x: 0.577, y: 0.400, width: 0.261, height: 0.031)
                    return lines.filter { row in
                        guard let b = row.bounds, let at = truthPose.referencePoint(CGPoint(x: b.midX, y: 1-b.midY)) else { return false }
                        return cell.contains(at)
                    }.map(\.text).joined()
                }
                var best: (CGFloat, String)?
                for line in lines {
                    guard let box = line.bounds else { continue }
                    let x = abs(box.midX-p.x), y = abs(1-box.midY-p.y)
                    guard x < 0.075, y < 0.012 else { continue }
                    let d = hypot(0.6*x, y)
                    if best == nil || d < best!.0 { best = (d, line.text) }
                }
                return best?.1 ?? ""
            }
            let gold = normalized(truth.text)
            characterCount += gold.count
            rawErrors += errors(gold,normalized(closest(raw)))
            shownErrors += errors(gold,normalized(closest(displayed)))
        }
        row.sampledCharacters = characterCount
        row.rawEdits = rawErrors
        row.matchedEdits = shownErrors
        func countKeyPhrases(_ lines: [PrintedLine]) -> Int {
            let words = String(normalized(lines.map(\.text).joined()))
            return keyPhrases.filter { words.contains($0) }.count
        }
        row.rawKeyPhrases = countKeyPhrases(raw)
        row.displayedKeyPhrases = countKeyPhrases(displayed)
        if let match {
            let unit = CGAffineTransform(scaleX: 1/2344.0, y: 1/3156.0)
            let truthBoxes = boxes.map { $0.applying(unit) }
            var px = [Double](), correct = [Bool](), wrong = [Bool]()
            for (index, target) in match.targets.enumerated() {
                let gb = truthBoxes[index]
                let actual = CGPoint(x: target.bounds.midX, y: target.bounds.midY)
                let expected = truthPose.livePoint(CGPoint(x: gb.midX, y: gb.midY))!
                px.append(hypot((actual.x-expected.x)*Double(sample.width), (actual.y-expected.y)*Double(sample.height)))
                let inverse = truthPose.referencePoint(actual)!
                correct.append(gb.contains(inverse))
                wrong.append(truthBoxes.indices.contains { $0 != index && truthBoxes[$0].contains(inverse) })
            }
            row.centerErrorPx = px
            row.centerInsideCorrectTarget = correct
            row.centerInsideAnotherTarget = wrong
        }
        return row
    }

    static func main() throws {
        let args = CommandLine.arguments
        let samples = try JSONDecoder().decode([Sample].self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
        let source = try Data(contentsOf: URL(fileURLWithPath: args[2]))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var results = [Result]()
        for sample in samples {
            let evaluation = try autoreleasepool { try measure(sample) }
            results.append(evaluation)
            print("\(sample.name): \(evaluation.matched ? "match" : "no match"), \(evaluation.lines) OCR lines")
            fflush(stdout)
        }
        var exact = [Double]()
        for _ in 0..<10 {
            let start = ProcessInfo.processInfo.systemUptime
            guard DemoConsentForm.exactPhoto(source) != nil else { throw CocoaError(.fileReadCorruptFile) }
            exact.append((ProcessInfo.processInfo.systemUptime-start)*1000)
        }
        struct Report: Encodable {
            let cases: [Result]
            let exactHEICLookupMs: [Double]
            let platform: String
        }
        let report = Report(cases: results, exactHEICLookupMs: exact, platform: ProcessInfo.processInfo.operatingSystemVersionString)
        try encoder.encode(report).write(to: URL(fileURLWithPath: args[3]))
    }
}
