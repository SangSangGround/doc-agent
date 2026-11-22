import Foundation
import XCTest
@testable import DocAgentCore

final class RetrievedPageQuestionTests: XCTestCase {
    func testPermissionMatcherKeepsKoreanNegationAttachedToTheVerb() {
        XCTAssertEqual(PageRetrieval.consentCondition(in: "집주인 허락이 없어도"), false)
        XCTAssertEqual(PageRetrieval.consentCondition(in: "임대인의 동의를 받지 못한 경우"), false)
        XCTAssertEqual(PageRetrieval.consentCondition(in: "허락을 받고 수수료 없이"), true)
        XCTAssertNotEqual(PageRetrieval.consentCondition(in: "동의를 받지만 서명은 나중에 합니다"), false)
    }

    private func row(_ id: Int, _ text: String, left: CGFloat, top: CGFloat, width: CGFloat) -> PrintedLine {
        PrintedLine(id: id, text: text, uncertain: false,
                    bounds: CGRect(x: left, y: top - 0.014, width: width, height: 0.014))
    }

    func testRelevantPhraseLateOnPageKeepsThreeOriginalLinesOnEitherSide() throws {
        let captions = ["반납", "점심에는 안내 데스크를 이용하세요."] + Array(repeating: "이용 안내", count: 14) +
            ["저녁 반납", "정문으로 들어오세요.", "문이 잠겨 있다면", "무인 반납함을 이용하세요.", "건물 배치도"]
        let lines = captions.enumerated().map { PrintedLine(id: $0.offset * 2, text: $0.element, uncertain: false) }
        let passages = PageRetrieval.passages(for: "저녁에 책을 어디다 돌려줘?", in: lines)
        let first = try XCTUnwrap(passages.first)
        XCTAssertEqual(first.anchor.id, lines[16].id)
        XCTAssertEqual(first.lines.map(\.id), lines[13...19].map(\.id))
    }

    func testCloudRequestKeepsRelevantTableCellsAndOtherRowsWithCoordinates() throws {
        let lines = [
            row(0, "열람 신청 안내", left: 0.1, top: 0.90, width: 0.3),
            row(1, "복사비", left: 0.1, top: 0.65, width: 0.10),
            row(2, "수수료", left: 0.7, top: 0.65, width: 0.10),
            row(3, "장당 500원", left: 0.1, top: 0.62, width: 0.12),
            row(4, "없음", left: 0.7, top: 0.62, width: 0.08)
        ]
        for question in ["이거 보려면 돈 내야 해?", "열람 수수료가 얼마야?"] {
            let request = try CloudQuestionRequest(question: question, lines: lines, history: [])
            XCTAssertTrue(request.focusLineIDs.contains(2))
            XCTAssertTrue(request.focusLineIDs.contains(4))
            XCTAssertEqual(request.lines.map(\.id), lines.map(\.id))
            XCTAssertTrue(request.lines.allSatisfy { $0.bounds?.count == 4 })
        }
    }
}
