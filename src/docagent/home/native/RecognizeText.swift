import Foundation
import Vision
import ImageIO

struct Word: Codable { let text: String; let box: [Double] }
struct Line: Codable { let text: String; let confidence: Float; let box: [Double]; let words: [Word] }
struct Boundary: Codable { let confidence: Float; let quad: [[Double]] }
struct Output: Codable { let engine: String; let width: Int; let height: Int; let lines: [Line] }

func pixels(_ box: CGRect, width: Int, height: Int) -> [Double] {
    [Double(box.minX) * Double(width), Double(1 - box.maxY) * Double(height),
     Double(box.width) * Double(width), Double(box.height) * Double(height)]
}

do {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    let supported = try request.supportedRecognitionLanguages()
    if CommandLine.arguments.contains("--languages") {
        FileHandle.standardOutput.write(try JSONEncoder().encode(supported))
        exit(0)
    }
    guard supported.contains("ko-KR") else { throw NSError(domain: "Korean OCR unavailable", code: 1) }
    request.recognitionLanguages = ["ko-KR", "en-US"]
    request.usesLanguageCorrection = false
    request.minimumTextHeight = 0.003
    let data = FileHandle.standardInput.readDataToEndOfFile()
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw NSError(domain: "Invalid image", code: 2)
    }
    if CommandLine.arguments.contains("--document") || CommandLine.arguments.contains("--rectangles") {
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        var observations: [VNRectangleObservation] = []
        if CommandLine.arguments.contains("--document") {
            let detect = VNDetectDocumentSegmentationRequest()
            try handler.perform([detect]); observations = detect.results ?? []
        } else {
            let detect = VNDetectRectanglesRequest()
            detect.minimumAspectRatio = 0.2; detect.minimumSize = 0.2
            detect.maximumObservations = 8; detect.minimumConfidence = 0.6
            try handler.perform([detect]); observations = detect.results ?? []
        }
        var boundaries: [Boundary] = []
        for observation in observations {
            let points: [CGPoint] = [observation.topLeft, observation.topRight, observation.bottomRight, observation.bottomLeft]
            var quad: [[Double]] = []
            for point in points {
                let x: Double = Double(point.x) * Double(image.width)
                let y: Double = (1.0 - Double(point.y)) * Double(image.height)
                quad.append([x, y])
            }
            boundaries.append(Boundary(confidence: observation.confidence, quad: quad))
        }
        FileHandle.standardOutput.write(try JSONEncoder().encode(boundaries)); exit(0)
    }
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    let pattern = try NSRegularExpression(pattern: "\\S+")
    let lines: [Line] = (request.results ?? []).compactMap { observation in
        guard let candidate = observation.topCandidates(1).first else { return nil }
        let text = candidate.string
        let words: [Word] = pattern.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap { match in
            guard let range = Range(match.range, in: text), let rectangle = try? candidate.boundingBox(for: range) else { return nil }
            return Word(text: String(text[range]), box: pixels(rectangle.boundingBox, width: image.width, height: image.height))
        }
        return Line(text: text, confidence: candidate.confidence,
                    box: pixels(observation.boundingBox, width: image.width, height: image.height), words: words)
    }
    FileHandle.standardOutput.write(try JSONEncoder().encode(Output(engine: "apple_vision", width: image.width, height: image.height, lines: lines)))
} catch {
    // 문서 원문이나 파일 경로를 오류 로그에 포함하지 않는다.
    FileHandle.standardError.write(Data("Local Vision OCR failed\n".utf8))
    exit(1)
}
