import CoreImage
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

/// The tracker and the field detector must see the same upright pixels and the same origin.
public struct WritingReference: @unchecked Sendable {
    public let image: CGImage
    public let jpeg: Data
    public let digest: String

    public init(photo: Data, rectify: Bool) throws {
        guard let source = CGImageSourceCreateWithData(photo as CFData, nil),
              let upright = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: 2200
              ] as CFDictionary) else { throw CocoaError(.fileReadCorruptFile) }
        let page = rectify ? Self.flatten(upright) ?? upright : upright
        for quality in [0.88, 0.68, 0.48] {
            let buffer = NSMutableData()
            guard let dest = CGImageDestinationCreateWithData(buffer, UTType.jpeg.identifier as CFString, 1, nil) else { break }
            CGImageDestinationAddImage(dest, page, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
            guard CGImageDestinationFinalize(dest), buffer.length <= 3 * 1024 * 1024 else { continue }
            let bytes = buffer as Data
            guard let encoded = CGImageSourceCreateWithData(bytes as CFData, nil),
                  let reference = CGImageSourceCreateImageAtIndex(encoded, 0, nil) else { break }
            jpeg = bytes
            image = reference
            digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            return
        }
        throw CocoaError(.fileReadCorruptFile)
    }

    private static func flatten(_ image: CGImage) -> CGImage? {
        let detect = VNDetectDocumentSegmentationRequest()
        try? VNImageRequestHandler(cgImage: image).perform([detect])
        guard let paper = detect.results?.first, paper.confidence >= 0.85,
              paper.boundingBox.width * paper.boundingBox.height > 0.18 else { return nil }
        func point(_ p: CGPoint) -> CIVector {
            CIVector(x: p.x * CGFloat(image.width), y: p.y * CGFloat(image.height))
        }
        let flat = CIImage(cgImage: image).applyingFilter("CIPerspectiveCorrection", parameters: [
            "inputTopLeft": point(paper.topLeft), "inputTopRight": point(paper.topRight),
            "inputBottomLeft": point(paper.bottomLeft), "inputBottomRight": point(paper.bottomRight)
        ])
        guard !flat.extent.isEmpty, flat.extent.width <= 4096, flat.extent.height <= 4096,
              flat.extent.width * flat.extent.height <= 8_000_000 else { return nil }
        return CIContext().createCGImage(flat, from: flat.extent)
    }
}

struct DetectedWritingFields: Decodable {
    struct Field: Decodable {
        let id: String
        let label: String
        let kind: String
        let bounds: [Double]
    }
    let imageSHA256: String
    let coordinateSpace: String
    let fields: [Field]

    func resolve(for digest: String) throws -> [WritingTarget] {
        guard imageSHA256 == digest, coordinateSpace == "upright-top-left-normalized",
              fields.count <= 100 else { throw CloudRequestError.response }
        var ids = Set<String>()
        return try fields.map { field in
            let label = field.label.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !field.id.isEmpty, ids.insert(field.id).inserted,
                  !label.isEmpty, label.count <= 240, let kind = WritingTarget.Kind(rawValue: field.kind),
                  field.bounds.count == 4, field.bounds.allSatisfy({ $0.isFinite && (0...1).contains($0) })
            else { throw CloudRequestError.response }
            let b = field.bounds
            guard b[2] > 0, b[3] > 0, b[0] + b[2] <= 1.000001, b[1] + b[3] <= 1.000001
            else { throw CloudRequestError.response }
            return WritingTarget(id: field.id, label: label, kind: kind,
                                 bounds: CGRect(x: b[0], y: b[1], width: b[2], height: b[3]))
        }
    }
}
