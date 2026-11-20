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
        XCTAssertTrue(DocumentReviewer.emptyNotice(appearance: .blankCandidate, vote: .yes).contains("뒷면"))
        for (image, vote) in [(PaperAppearance.lightProblem, VisualVote.yes), (.other, .yes), (.blankCandidate, .unsure), (.blankCandidate, .no)] {
            XCTAssertFalse(DocumentReviewer.emptyNotice(appearance: image, vote: vote).contains("뒷면"))
        }
    }
    func testRecheckKeepsNumbersAndRequiresTwoAgreements() {
        let low = PrintedLine(id: 1, text: "참가비 12,000원", uncertain: true)
        XCTAssertFalse(DocumentReviewer.reviewLine(low, candidates: [low.text], vote: .yes).uncertain)
        XCTAssertTrue(DocumentReviewer.reviewLine(low, candidates: ["참가비 12,800원"], vote: .yes).uncertain)
        XCTAssertTrue(DocumentReviewer.reviewLine(low, candidates: [low.text], vote: .unsure).uncertain)
        let high = PrintedLine(id: 2, text: "3월 11일", uncertain: false)
        let disputed = DocumentReviewer.reviewLine(high, candidates: ["3월 17일"], vote: .no)
        XCTAssertEqual(disputed.text, high.text)
        XCTAssertTrue(disputed.uncertain)
        XCTAssertFalse(DocumentReviewer.reviewLine(high, candidates: [high.text], vote: .no).uncertain)
        let top = CGRect(x: 0.1, y: 0.6, width: 0.4, height: 0.05)
        XCTAssertFalse(DocumentReviewer.sameLocation(top, top.offsetBy(dx: 0, dy: 0.095)))
        XCTAssertTrue(DocumentReviewer.sameLocation(top, top.offsetBy(dx: 0.015, dy: 0.002)))
    }

    func testBundledModelReviewsBlankAndKoreanPrintOffline() async throws {
        guard VisualReview.weights != nil, VisualReview.projector != nil else {
            throw XCTSkip("Run ios/prepare_review_model.py to test the bundled model.")
        }
        let reader = DocumentReviewer()
        let blank = try await reader.read(png(paper(shade: 0.91))) { _ in }
        XCTAssertTrue(blank.lines.isEmpty)
        XCTAssertTrue(blank.notice.contains("뒤집어"), blank.notice)
        let printed = try await reader.read(png(paper(shade: 0.91, text: "참가비 12,000원"))) { _ in }
        XCTAssertTrue(printed.lines.contains { $0.text.contains("12,000") }, printed.notice)
        XCTAssertFalse(printed.notice.contains("마치지 못"), printed.notice)
        XCTAssertFalse(printed.notice.contains("뒷면"), printed.notice)
    }

    func testVisibleKoreanAmountCanClearALowConfidenceFlag() throws {
        guard VisualReview.weights != nil, VisualReview.projector != nil else {
            throw XCTSkip("Run ios/prepare_review_model.py to test the bundled model.")
        }
        let page = try PhotoReader.scan(png(paper(shade: 0.91, text: "참가비 12,000원")))
        let located = try XCTUnwrap(page.lines.first { $0.value.text.contains("12,000") })
        let crop = try XCTUnwrap(PhotoReader.crop(page.upright, to: located.bounds.insetBy(dx: -0.012, dy: -0.012)))
        let model = try VisualReview(cancellation: ReviewCancellation())
        let vote = try model.matches([located.value.text], in: crop)
        XCTAssertEqual(vote, .yes)
        let weak = PrintedLine(id: 0, text: located.value.text, uncertain: true)
        let verified = DocumentReviewer.reviewLine(weak, candidates: [located.value.text], vote: vote)
        XCTAssertEqual(verified.text, located.value.text)
        XCTAssertFalse(verified.uncertain)
    }
}
