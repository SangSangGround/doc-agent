import Foundation
import XCTest
@testable import DocAgentCore

final class PageQuestionTests: XCTestCase {
    private func page(_ text: [String]) -> [PrintedLine] {
        text.enumerated().map { PrintedLine(id: $0.offset, text: $0.element, uncertain: false) }
    }
    private var libraryNotice: [PrintedLine] {
        page(["<붙임 3>", "도서관 임시 휴관 안내문", "시설 점검을 위해 6월 12일 도서관 문을 닫습니다.",
              "책은 정문 옆 무인 반납함에 넣어 주세요.", "문의 전화 02-1234-5678"])
    }

    func testAttachmentColumnKeepsWrappedConditionsAndExcludesAdjacentFee() async throws {
        func row(_ id: Int, _ text: String, _ x: Double, _ y: Double, _ w: Double, uncertain: Bool = false) -> PrintedLine {
            PrintedLine(id: id, text: text, uncertain: uncertain, bounds: CGRect(x: x, y: y, width: w, height: 0.012))
        }
        // Observation order can put the field name and fee in the middle of a numbered list.
        let lines = [row(0, "1. 참가 확인서", 0.20, 0.54, 0.30),
                     row(1, "(단체 참가자만 제출)", 0.22, 0.52, 0.45, uncertain: true),
                     row(2, "수수료", 0.84, 0.51, 0.05),
                     row(3, "첨부 서류", 0.10, 0.50, 0.08),
                     row(4, "2. 활동 기록 사본 1부", 0.20, 0.50, 0.45),
                     row(5, "없음", 0.85, 0.49, 0.04),
                     row(6, "3. 신청인 확인 자료", 0.20, 0.48, 0.40),
                     row(7, "담당 공무원 확인사항", 0.09, 0.455, 0.10),
                     row(8, "공부 등본", 0.20, 0.455, 0.15)]
        let reader = DocumentReviewer()
        for q in ["첨부 서류에 뭐가 있어?", "어떤 서류를 제출해야 해?", "뭘 같이 내야 돼?"] {
            let answer = try await reader.answer(q, lines: lines, history: [])
            XCTAssertEqual(answer, .page([lines[0], lines[1], lines[4], lines[6]]), q)
        }
    }

    func testAttachmentsInReadingOrderAndMissingContents() {
        func answer(_ rows: [String]) -> ConsentReply? {
            AttachmentSection.answer(in: page(rows))
        }
        let list = answer(["서식", "구비서류: 1. 확인서", "2. 신청서 사본", "(해당자만 제출)", "유의사항", "기한은 5월 1일"])
        XCTAssertEqual(list?.text, "1. 확인서\n2. 신청서 사본\n(해당자만 제출)")
        XCTAssertEqual(list?.source, "촬영한 문서의 인식 줄 번호 2, 3, 4")
        XCTAssertEqual(answer(["첨부", "서류", "없음", "수수료", "500원"])?.text, "없음")
        XCTAssertEqual(answer(["첨부서류", "수수료", "없음"])?.kind, .notice)
        XCTAssertNil(answer(["신청서", "수수료", "없음"]))
        XCTAssertFalse(AttachmentSection.requested(.init("첨부서류는 어디서 발급받아?")))
    }

    func testIdentityAndSummaryRouting() throws {
        for q in ["이거 어떤 서류야?", "무슨 문서예요?", "이게 뭐야?"] {
            guard case .direct(let answer) = ConsentConversation.plan(q, lines: libraryNotice, history: []) else { return XCTFail(q) }
            XCTAssertEqual(answer.text, "도서관 임시 휴관 안내문")
            XCTAssertTrue(answer.source.contains("2"))
        }
        for q in ["한 줄로 요약해줘", "이 문서가 무슨 내용이야?", "핵심만 알려줘"] {
            guard case .page(let request) = ConsentConversation.plan(q, lines: libraryNotice, history: []) else { return XCTFail(q) }
            XCTAssertEqual(request.intent, .summary, q)
        }
        guard case .page(let required) = ConsentConversation.plan("어떤 서류를 가져가야 해?", lines: libraryNotice, history: []) else { return XCTFail() }
        XCTAssertEqual(required.intent, .detail)
        let whoSigned = ConsentConversation.plan("이 동의서는 누가 작성했어?", lines: page(["개인정보 동의서", "작성자: 김하늘"]), history: [])
        guard case .page(let author) = whoSigned else { return XCTFail() }
        XCTAssertEqual(author.intent, .detail)
        let unreadable = PageQuestion.plan(.identity, question: "무슨 서류야?", lines: [])
        guard case .direct(let notice) = unreadable else { return XCTFail() }
        XCTAssertEqual(notice.kind, .notice)
    }

    func testClarificationAndFollowingChoice() throws {
        let consent = page(["개인정보 제공 동의서", "대부 신청 기한: 3월 1일", "개인정보 보유 기간: 2년"])
        let question = "언제까지야?"
        guard case .direct(let ask) = ConsentConversation.plan(question, lines: consent, history: []) else { return XCTFail() }
        XCTAssertEqual(ask.choices, ["신청 기한", "개인정보 보유 기간"])
        let history = [ConsentTurn(question: question, reply: ask)]
        for input in ["2번", "개인정보 보유 기간"] {
            guard case .search(let search) = ConsentConversation.plan(input, lines: consent, history: history) else { return XCTFail(input) }
            XCTAssertTrue(search.certainMatch?.text.contains("2년") == true)
        }
        guard case .direct(let clarification) = ConsentConversation.plan("그거 알려줘", lines: libraryNotice, history: []) else { return XCTFail() }
        let next = ConsentConversation.plan("첫 번째", lines: libraryNotice, history: [.init(question: "그거 알려줘", reply: clarification)])
        guard case .direct(let title) = next else { return XCTFail() }
        XCTAssertEqual(title.text, libraryNotice[1].text)
    }

    func testFreeQuestionsOnPrintedNoticeUsingBundledModel() async throws {
        guard DocumentLanguageModel.weights != nil else { throw XCTSkip("Bundled Qwen required.") }
        let reader = DocumentReviewer()
        for (q, expected, line) in [("빌린 책 어디다 돌려줘?", "무인 반납함", "4"),
                                     ("전화하려면 어디로 걸어?", "02-1234-5678", "5"),
                                     ("왜 쉬는 거지?", "시설 점검", "3")] {
            let answer = try await reader.answer(q, lines: libraryNotice, history: [])
            XCTAssertEqual(answer.kind, .document, q + " / " + answer.text)
            XCTAssertTrue(answer.text.contains(expected), q + " / " + answer.text)
            XCTAssertTrue(answer.source.hasSuffix(line), answer.source)
        }
        let summary = try await reader.answer("한 줄로 요약해줘", lines: libraryNotice, history: [])
        XCTAssertEqual(summary.kind, .document, summary.text)
        XCTAssertTrue(["문을 닫", "휴관", "시설 점검"].contains(where: summary.text.contains), summary.text)
        XCTAssertFalse(summary.text.contains("\n"), summary.text)
        XCTAssertNotEqual(summary.text, libraryNotice[1].text)
        for q in ["주차비는 얼마야?", "몇 시에 열어?"] {
            let missing = try await reader.answer(q, lines: libraryNotice, history: [])
            XCTAssertEqual(missing.kind, .notice, q + " / " + missing.text)
        }
    }
}
