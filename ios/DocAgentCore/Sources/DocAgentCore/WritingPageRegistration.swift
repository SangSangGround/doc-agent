import CoreGraphics
import CoreVideo
import Foundation
import Vision
import WritingVision

/// The v7.5 reference matcher is stateful and must be called from a serial queue.
public final class WritingPageRegistration {
    public let paperCoordinates: WritingTransform?
    private let tracker: OpaquePointer

    public init?(reference: CGImage) {
        let request = VNDetectDocumentSegmentationRequest()
        try? VNImageRequestHandler(cgImage: reference).perform([request])
        let corners: [Double]
        if let observation = request.results?.first, observation.confidence >= 0.70,
           observation.boundingBox.width * observation.boundingBox.height > 0.14 {
            corners = [observation.topLeft, observation.topRight, observation.bottomRight, observation.bottomLeft]
                .flatMap { [Double($0.x), Double(1 - $0.y)] }
            var matrix = [Double](repeating: 0, count: 9)
            paperCoordinates = DAWritingPaperCoordinates(corners, &matrix) ? WritingTransform(rowMajor: matrix) : nil
        } else {
            // A photographed page may share the same desk as the next page. If the
            // page outline disappears, use its printed area rather than desk texture.
            paperCoordinates = nil
            let printed = VNDetectTextRectanglesRequest()
            try? VNImageRequestHandler(cgImage: reference).perform([printed])
            let boxes = (printed.results ?? []).map(\.boundingBox)
            guard boxes.count >= 3 else { return nil }
            let unit = CGRect(x: 0, y: 0, width: 1, height: 1)
            let b = boxes.reduce(CGRect.null) { $0.union($1) }.insetBy(dx: -0.025, dy: -0.025).intersection(unit)
            guard !b.isNull, b.width*b.height >= 0.08 else { return nil }
            corners = [b.minX, 1-b.maxY, b.maxX, 1-b.maxY, b.maxX, 1-b.minY, b.minX, 1-b.minY]
        }
        guard let gray = GrayImage(reference) else { return nil }
        let value = gray.pixels.withUnsafeBufferPointer {
            DAWritingCreate($0.baseAddress, Int32(gray.width), Int32(gray.height), gray.width, corners)
        }
        guard let value else { return nil }
        tracker = value
    }

    deinit { DAWritingDestroy(tracker) }

    public func register(_ buffer: CVPixelBuffer) -> WritingTransform? {
        if CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA {
            guard CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else { return nil }
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
            var matrix = [Double](repeating: 0, count: 9)
            guard DAWritingRegisterBGRA(tracker, base.assumingMemoryBound(to: UInt8.self),
                                        Int32(CVPixelBufferGetWidth(buffer)), Int32(CVPixelBufferGetHeight(buffer)),
                                        CVPixelBufferGetBytesPerRow(buffer), &matrix) else { return nil }
            return WritingTransform(rowMajor: matrix)
        }
        guard [kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange]
                .contains(CVPixelBufferGetPixelFormatType(buffer)), CVPixelBufferGetPlaneCount(buffer) == 2,
              CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else { return nil }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let plane = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return nil }
        return register(plane.assumingMemoryBound(to: UInt8.self),
                        width: CVPixelBufferGetWidthOfPlane(buffer, 0), height: CVPixelBufferGetHeightOfPlane(buffer, 0),
                        stride: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0))
    }

    /// Allows replay of the exact registration code on saved frames without a phone camera.
    public func register(_ image: CGImage) -> WritingTransform? {
        guard let gray = GrayImage(image) else { return nil }
        return gray.pixels.withUnsafeBufferPointer {
            register($0.baseAddress, width: gray.width, height: gray.height, stride: gray.width)
        }
    }

    private func register(_ bytes: UnsafePointer<UInt8>?, width: Int, height: Int, stride: Int) -> WritingTransform? {
        var values = [Double](repeating: 0, count: 9)
        guard DAWritingRegister(tracker, bytes, Int32(width), Int32(height), stride, &values) else { return nil }
        return WritingTransform(rowMajor: values)
    }
}

private struct GrayImage {
    var pixels: [UInt8]
    let width: Int
    let height: Int

    init?(_ image: CGImage) {
        let scale = min(1.0, 1600.0 / Double(max(image.width, image.height)))
        let w = max(1, Int((Double(image.width) * scale).rounded()))
        let h = max(1, Int((Double(image.height) * scale).rounded()))
        var out = [UInt8](repeating: 0, count: w * h)
        let ok = out.withUnsafeMutableBytes { bytes in
            guard let context = CGContext(data: bytes.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                          bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(),
                                          bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { return nil }
        width = w
        height = h
        pixels = out
    }
}
