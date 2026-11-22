import XCTest
@testable import DocAgentCore

final class CloudQuestionTests: XCTestCase {
    let lines = [PrintedLine(id: 2, text: "신청 기한 6월 1일", confidence: 0.60),
                 PrintedLine(id: 7, text: "참가비 없음", confidence: 0.59)]
    private func response(_ json: String) throws -> CloudQuestionResponse {
        try JSONDecoder().decode(CloudQuestionResponse.self, from: Data(json.utf8))
    }

    func testThresholdIsInclusiveAndOnlyCitedRowsChangeWithoutRewritingOCR() throws {
        let request = try CloudQuestionRequest(question: "신청 기한은 언제야?", lines: lines, history: [])
        let answer = try response(#"{"kind":"document","text":"6월 1일이에요.","lineIDs":[2],"referenceID":null,"choices":[],"lineContexts":[{"id":2,"score":0.60}]}"#)
        let result = try answer.resolve(for: request, page: lines)
        XCTAssertEqual(result.lines.map(\.text), lines.map(\.text))
        XCTAssertEqual(result.reply.source, "촬영한 문서의 인식 줄 번호 3")
        XCTAssertEqual(result.reply.spoken, "지금 찍은 문서에서 읽어드릴게요. 6월 1일이에요.")
        XCTAssertFalse(result.lines[0].uncertain)
        XCTAssertTrue(result.lines[1].uncertain)
        XCTAssertTrue(lines[0].uncertain)
        let repeated = try answer.resolve(for: request, page: result.lines)
        XCTAssertEqual(repeated.lines[0].confidence, lines[0].confidence)
        XCTAssertEqual(repeated.lines[0].contextSupport, result.lines[0].contextSupport)
        XCTAssertTrue(lines[1].corroborated(by: 0.60).uncertain)
        XCTAssertTrue(lines[0].corroborated(by: 0.5999).uncertain)
        XCTAssertTrue(PrintedLine(id: 2, text: "6월", confidence: 0.10).corroborated(by: 1).uncertain)
    }

    func testMissingContextsOrDisagreementLeaveOriginalWarning() throws {
        let disputed = DocumentReviewer.reviewLine(lines[0], second: .init(text: "신청 기한 6월 7일", confidence: 1))
        let photo = [disputed, lines[1]]
        let request = try CloudQuestionRequest(question: "언제까지야?", lines: photo, history: [])
        let missing = try response(#"{"kind":"document","text":"문서에 기한이 있어요.","lineIDs":[7],"choices":[]}"#)
        XCTAssertTrue(try missing.resolve(for: request, page: photo).reply.text.contains("불확실"))
        let strong = try response(#"{"kind":"document","text":"6월 1일로 읽혔어요.","lineIDs":[2],"choices":[],"lineContexts":[{"id":2,"score":1}]}"#)
        let result = try strong.resolve(for: request, page: photo)
        XCTAssertTrue(result.lines[0].uncertain)
        XCTAssertTrue(result.reply.text.contains("불확실"))
    }

    func testAutomaticPageReviewPreservesTextAndMergesWithAnswersAlreadyReceived() throws {
        let photo = lines + [PrintedLine(id: 9, text: "기간 3년", confidence: 0.59, conflictingReadings: true)]
        let request = try CloudOCRReviewRequest(page: photo)
        let encoded = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any])
        XCTAssertEqual(Set(encoded.keys), ["lines"])
        let alreadyReviewed = [photo[0], photo[1].corroborated(by: 0.95), photo[2]]
        let review = try JSONDecoder().decode(CloudOCRReviewResponse.self, from: Data(#"{"lineContexts":[{"id":2,"score":0.60},{"id":9,"score":1}]}"#.utf8))
        let updated = try review.applying(to: alreadyReviewed, request: request)
        XCTAssertEqual(updated.map(\.text), photo.map(\.text))
        XCTAssertEqual(updated.map(\.uncertain), [false, false, true])
        XCTAssertThrowsError(try review.applying(to: [photo[0].withText("다른 문서")] + photo.dropFirst(), request: request))
    }

    func testReviewRejectsUnknownDuplicateOrInvalidScoresAndNeverScoresCertainLines() throws {
        let page = lines + [PrintedLine(id: 10, text: "서명", confidence: 0.9)]
        let request = try CloudOCRReviewRequest(page: page)
        for items in [#"[{"id":2,"score":1.1}]"#, #"[{"id":123,"score":0.8}]"#,
                      #"[{"id":2,"score":0.8},{"id":2,"score":0.7}]"#, #"[{"id":10,"score":0.8}]"#] {
            let response = try JSONDecoder().decode(CloudOCRReviewResponse.self, from: Data(("{\"lineContexts\":" + items + "}").utf8))
            XCTAssertThrowsError(try response.applying(to: page, request: request))
        }
    }

    func testInvalidEvidenceCannotBecomeAnAnswerOrChangeWarnings() throws {
        let request = try CloudQuestionRequest(question: "무슨 서류야?", lines: lines, history: [])
        for json in [
            #"{"kind":"document","text":"답","lineIDs":[8],"choices":[]}"#,
            #"{"kind":"document","text":"답","lineIDs":[2],"choices":[],"lineContexts":[{"id":7,"score":1}]}"#,
            #"{"kind":"document","text":"답","lineIDs":[2],"choices":[],"lineContexts":[{"id":2,"score":1.2}]}"#,
            #"{"kind":"example","text":"답","lineIDs":[],"referenceID":"invented","choices":[]}"#,
            #"{"kind":"term","text":"답","lineIDs":[],"referenceID":"bank-records","choices":[]}"#
        ] {
            XCTAssertThrowsError(try response(json).resolve(for: request, page: lines), json)
        }
        let stale = try response(#"{"kind":"document","text":"기한은 6월 1일이에요.","lineIDs":[2],"choices":[],"lineContexts":[{"id":2,"score":1}]}"#)
        let otherPage = [PrintedLine(id: 2, text: "기한 7월 1일", confidence: 0.60), lines[1]]
        XCTAssertThrowsError(try stale.resolve(for: request, page: otherPage))
    }

    func testLibraryAttributionAndConversationGrounding() throws {
        let request = try CloudQuestionRequest(question: "은행 계좌 조회 사례?", lines: lines, history: [])
        let answer = try response(#"{"kind":"example","text":"거래내역을 제공한 사건이에요.","lineIDs":[],"referenceID":"bank-records","choices":[]}"#)
        let result = try answer.resolve(for: request, page: lines)
        XCTAssertTrue(result.reply.source.contains("2008도3598"))
        XCTAssertEqual(result.reply.url?.host, "www.law.go.kr")
        let history = [ConsentTurn(question: request.question, reply: result.reply)]
        let followup = try CloudQuestionRequest(question: "그 사건에서 왜 문제가 됐지?", lines: lines, history: history)
        XCTAssertEqual(followup.history.first?.question, request.question)
        XCTAssertTrue(followup.history.first?.answer.contains(result.reply.text) ?? false)
        XCTAssertTrue(followup.history.first?.answer.contains(result.reply.source) ?? false)
        XCTAssertTrue(followup.references.contains { $0.id == "bank-records" && $0.kind == "example" })
        let clarification = try response(#"{"kind":"notice","text":"어느 기간을 물어보시나요?","lineIDs":[],"referenceID":null,"choices":["보유 기간","신청 기한"]}"#)
        let reply = try clarification.resolve(for: followup, page: lines).reply
        XCTAssertEqual(reply.choices, ["보유 기간", "신청 기한"])
        XCTAssertTrue(reply.spoken.contains("2번, 신청 기한"))
    }
}
