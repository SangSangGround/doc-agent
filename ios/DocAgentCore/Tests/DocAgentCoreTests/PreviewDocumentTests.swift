import CoreGraphics
import XCTest
@testable import DocAgentCore

final class PreviewDocumentTests: XCTestCase {
    private func view(of sheet: CGRect, lightDesk: Bool = false) throws -> (page: PreviewPage?, partial: FramingCorrection?) {
        let w = 300, h = 400
        let context = try XCTUnwrap(CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                               space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                               bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.scaleBy(x: CGFloat(w), y: CGFloat(h))
        let desk = lightDesk ? (205, 206, 205) : (204, 161, 82)
        let paper = (228, 227, 225), printColor = (70, 70, 73)
        let table = CGRect(x: sheet.midX, y: sheet.midY - 0.13, width: sheet.width * 0.35, height: 0.26)
        let inside = table.insetBy(dx: 0.008, dy: 0.008)
        func fill(_ rect: CGRect, _ rgb: (Int, Int, Int)) {
            context.setFillColor(red: CGFloat(rgb.0) / 255, green: CGFloat(rgb.1) / 255, blue: CGFloat(rgb.2) / 255, alpha: 1)
            context.fill(rect)
        }
        fill(CGRect(x: 0, y: 0, width: 1, height: 1), desk)
        fill(sheet, paper)
        fill(table, printColor)
        fill(inside, paper)
        let image = try XCTUnwrap(context.makeImage())
        return PreviewDocumentDetector.observe(in: image) { p in
            let pixel = !sheet.contains(p) ? desk : (table.contains(p) && !inside.contains(p) ? printColor : paper)
            let y = 0.299 * Double(pixel.0) + 0.587 * Double(pixel.1) + 0.114 * Double(pixel.2)
            let c = max(0.564 * abs(Double(pixel.2) - y), 0.713 * abs(Double(pixel.0) - y))
            return (Int(y.rounded()), Int(c.rounded()))
        }
    }

    func testCutSheetGuidesEachDirectionEvenWithABoxInsideThePage() throws {
        let frames: [(CGRect, FramingHint)] = [
            (CGRect(x: -0.24, y: 0.18, width: 0.64, height: 0.65), .left),
            (CGRect(x: 0.61, y: 0.18, width: 0.64, height: 0.65), .right),
            (CGRect(x: 0.18, y: 0.63, width: 0.58, height: 0.65), .up),
            (CGRect(x: 0.18, y: -0.25, width: 0.58, height: 0.65), .down)
        ]
        for (bounds, direction) in frames {
            let frame = try view(of: bounds)
            XCTAssertNil(frame.page)
            XCTAssertEqual(frame.partial?.direction, direction)
        }
        let before = try XCTUnwrap(view(of: frames[0].0).partial?.fit)
        let closer = try XCTUnwrap(view(of: frames[0].0.offsetBy(dx: 0.16, dy: 0)).partial?.fit)
        XCTAssertGreaterThan(closer, before)
    }

    func testWholePaperIsRecognizedWithRealMarginAndOnLightDesk() throws {
        let nearEdge = CGRect(x: 0.025, y: 0.12, width: 0.64, height: 0.75)
        let whole = try XCTUnwrap(view(of: nearEdge).page)
        XCTAssertEqual(whole.bounds.minX, nearEdge.minX, accuracy: 0.03)
        XCTAssertEqual(whole.bounds.width, nearEdge.width, accuracy: 0.04)
        XCTAssertNotNil(try view(of: CGRect(x: 0.2, y: 0.11, width: 0.6, height: 0.78), lightDesk: true).page)
    }

    func testInsetEstimatedByVisionNeedsAnActualExteriorStrip() {
        let outline = PreviewPage([CGPoint(x: 0.007, y: 0.82), CGPoint(x: 0.4, y: 0.82),
                                   CGPoint(x: 0.4, y: 0.18), CGPoint(x: 0.007, y: 0.18)])
        XCTAssertEqual(PreviewDocumentDetector.edgeCorrection(for: outline) { _ in (210, 5) }?.direction, .left)
        XCTAssertNil(PreviewDocumentDetector.edgeCorrection(for: outline) { $0.x < 0.007 ? (172, 46) : (210, 5) })
    }
}
