import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import DocAgentCore

final class PhotoReviewTests: XCTestCase {
    private func paper(shade: CGFloat, text: String? = nil) throws -> CGImage {
        let context = try XCTUnwrap(CGContext(data: nil, width: 900, height: 520, bitsPerComponent: 8,
                                              bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.setFillColor(gray: shade, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 900, height: 520))
        if let text {
            let attrs: [NSAttributedString.Key: Any] = [
                NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("AppleSDGothicNeo-Regular" as CFString, 64, nil),
                NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 0.08, alpha: 1)
            ]
            context.textPosition = CGPoint(x: 50, y: 235)
            CTLineDraw(CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attrs)), context)
        }
        return try XCTUnwrap(context.makeImage())
    }
    private func png(_ image: CGImage) throws -> Data {
        let buffer = NSMutableData()
        let dest = try XCTUnwrap(CGImageDestinationCreateWithData(buffer, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        return buffer as Data
    }

    func testBlankRequiresImageEvidenceAndAgreement() throws {
        XCTAssertEqual(PaperAppearance.inspect(try paper(shade: 0.90)), .blankCandidate)
        XCTAssertEqual(PaperAppearance.inspect(try paper(shade: 1.0)), .lightProblem)
        XCTAssertEqual(PaperAppearance.inspect(try paper(shade: 0.04)), .lightProblem)
        XCTAssertEqual(PaperAppearance.inspect(try paper(shade: 0.90, text: "금액 12,000원")), .other)
        XCTAssertTrue(DocumentReviewer.emptyNotice(appearance: .blankCandidate).contains("뒷면"))
        for appearance in [PaperAppearance.lightProblem, .other] {
            XCTAssertFalse(DocumentReviewer.emptyNotice(appearance: appearance).contains("뒷면"))
        }
    }
    func testTwoVisionReadsRaiseScoreAndKeepOriginalText() {
        let low = PrintedLine(id: 1, text: "참가비 12,000원", confidence: 0.5)
        let again = OCRReading(text: "참가비 12,000 원", confidence: 0.5)
        let confirmed = DocumentReviewer.reviewLine(low, second: again)
        XCTAssertEqual(confirmed.confidence, PrintedLine.warningThreshold, accuracy: 0.0001)
        XCTAssertFalse(confirmed.uncertain)
        XCTAssertEqual(confirmed.spoken, low.text)
        XCTAssertEqual(confirmed.withText("12,000원").confidence, confirmed.confidence)
        XCTAssertTrue(DocumentReviewer.reviewLine(low, second: nil).uncertain)
        let weak = PrintedLine(id: 0, text: "가", confidence: 0.2)
        XCTAssertTrue(DocumentReviewer.reviewLine(weak, second: .init(text: "가", confidence: 0.2)).uncertain)
        XCTAssertTrue(PrintedLine(id: 0, text: "", confidence: .nan).uncertain)
    }

    func testConflictingReadingsKeepWarningEvenWithContextAgreement() {
        for (first, other) in [("3월 11일", "3월 17일"), ("동의 없이 신청", "동의 있어야 신청"),
                               ("1 2", "12"), ("1.5%", "15%"), ("1만 원", "1억 원")] {
            let high = PrintedLine(id: 2, text: first, confidence: 1)
            let disputed = DocumentReviewer.reviewLine(high, second: .init(text: other, confidence: 1))
            XCTAssertEqual(disputed.text, first)
            XCTAssertTrue(disputed.conflictingReadings)
            XCTAssertTrue(disputed.corroborated(by: 1).uncertain, "\(first) / \(other)")
        }
        let top = CGRect(x: 0.1, y: 0.6, width: 0.4, height: 0.05)
        XCTAssertFalse(DocumentReviewer.sameLocation(top, top.offsetBy(dx: 0, dy: 0.095)))
        XCTAssertTrue(DocumentReviewer.sameLocation(top, top.offsetBy(dx: 0.015, dy: 0.002)))
    }

    func testCameraReadsBlankAndKoreanPrintLocally() async throws {
        let reader = DocumentReviewer()
        let blank = try await reader.read(png(paper(shade: 0.91)))
        XCTAssertTrue(blank.lines.isEmpty)
        XCTAssertTrue(blank.notice.contains("뒤집어"), blank.notice)
        let printed = try await reader.read(png(paper(shade: 0.91, text: "참가비 12,000원")))
        XCTAssertTrue(printed.lines.contains { $0.text.contains("12,000") }, printed.notice)
        XCTAssertFalse(printed.notice.contains("마치지 못"), printed.notice)
        XCTAssertFalse(printed.notice.contains("뒷면"), printed.notice)
    }

}
