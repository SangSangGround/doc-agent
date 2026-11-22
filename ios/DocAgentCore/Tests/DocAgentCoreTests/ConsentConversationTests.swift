import Foundation
import XCTest
@testable import DocAgentCore

final class ConsentConversationTests: XCTestCase {
    // A short rephrased fixture with two different retention conditions, as on the loan consent form.
    private let training = [
        "개인정보의 수집·이용·제공 동의서", "직업훈련생계비 대부 신청", "① 수집·이용에 관한 사항",
        "가. 수집·이용 목적: 대부 대상자 선발 및 훈련 계속 참여 여부 확인",
        "나. 수집·이용할 항목", "고용보험 가입 이력과 직업훈련 출석·수료 정보",
        "다. 보유·이용기간: 이용 목적을 달성할 때까지", "상환조건에 따라 최대 10년",
        "② 제공에 관한 사항", "가. 제공받는 자", "고용노동부 및 한국고용정보원, 사업평가 업무 수행기관",
        "나. 제공받는 자의 이용 목적: 정부재정사업 이중수혜방지 및 사업평가",
        "제공할 개인정보의 항목: 고유식별정보, 인적사항, 수혜내역 등",
        "제공받은 자의 개인정보 보유·이용 기간: 법령상 보존 의무가 있는 경우를 제외하고",
        "이용 목적을 달성할 때까지 보유 및 이용합니다.", "본인은 위 목적의 제공에 동의합니다.",
        "3. 동의를 거부할 권리 및 동의를 거부할 경우의 불이익", "거부할 수 있습니다. 다만 대부 신청에 필수여서",
        "위 사항에 동의하셔야만 직업훈련생계비 대부 신청이 가능합니다.", "이름: (서명 또는 인)"
    ]

    private var lines: [PrintedLine] {
        training.enumerated().map { PrintedLine(id: $0.offset, text: $0.element, uncertain: $0.offset == 10) }
    }

    private func reply(_ question: String, _ history: [ConsentTurn] = [], page: [PrintedLine]? = nil) -> ConsentReply {
        switch ConsentConversation.plan(question, lines: page ?? lines, history: history) {
        case .direct(let r): return r
        case .search(let s): return s.certainMatch ?? s.missing
        case .page(let page): return page.fallback
        }
    }

    func testLoanFieldsSourcesAndRefusal() {
        let who = reply("내 정보를 어느 기관에 줘?")
        XCTAssertEqual(who.kind, .document)
        XCTAssertTrue(who.text.contains("한국고용정보원"))
        XCTAssertTrue(who.text.contains("인식 불확실"))
        XCTAssertFalse(who.text.contains("수혜내역"))
        XCTAssertNil(who.url)
        XCTAssertTrue(reply("왜 다른 기관에 제공하는 거야?").text.contains("이중수혜"))
        XCTAssertTrue(reply("어떤 항목을 제공해?").text.contains("고유식별"))
        XCTAssertTrue(reply("동의를 거절하면 어떻게 돼?").text.contains("동의하셔야만"))
        let different = [PrintedLine(id: 9, text: "제공받는 자: 가상 시험기관", uncertain: false)]
        XCTAssertTrue(reply("누구한테 줘?", page: different).text.contains("가상 시험기관"))
        XCTAssertFalse(reply("누구한테 줘?", page: different).text.contains("한국고용정보원"))
        XCTAssertEqual(reply("누구한테 줘?", page: Array(lines.prefix(7))).kind, .notice)
        XCTAssertEqual(reply("누구한테 줘?", page: [PrintedLine(id: 0, text: "제공받는 자", uncertain: false)]).kind, .notice)
    }

    func testDifferentRetentionRulesAndFollowingUpAfterExternalAnswer() {
        let q = "공단은 내 정보를 얼마나 오래 보관해?"
        let first = reply(q)
        XCTAssertTrue(first.text.contains("최대 10년"))
        let next = reply("그럼 제공받은 곳은?", [.init(question: q, reply: first)])
        XCTAssertTrue(next.text.contains("목적을 달성할 때까지"))
        XCTAssertFalse(next.text.contains("10년"))
        let again = reply("그럼 공단은?", [.init(question: q, reply: first), .init(question: "그럼 제공받은 곳은?", reply: next)])
        XCTAssertTrue(again.text.contains("최대 10년"))
        let onlyCollection = Array(lines.prefix(8)) + [PrintedLine(id: 8, text: "10년간 보관합니다.", uncertain: false)]
        XCTAssertEqual(reply("그럼 제공받은 곳은?", [.init(question: q, reply: first)], page: onlyCollection).kind, .notice)
        for question in ["대부는 몇 년 안에 갚아야 해?", "대부 한도는 얼마야?", "신청은 언제까지야?", "금리는 몇 퍼센트야?"] {
            XCTAssertEqual(reply(question).kind, .notice, question)
        }
        let external = reply("계좌 조회와 관련된 실제 사례가 있어?")
        let refusal = reply("그럼 내가 찍은 동의서에서 거부하면?", [.init(question: "계좌 조회 사례", reply: external)])
        XCTAssertEqual(refusal.kind, .document)
        XCTAssertTrue(refusal.text.contains("대부 신청"))
    }

    func testRetrievesDifferentCasesVerifiableSourcesAndTopicReplay() throws {
        let examples = ConsentEntry.library.filter { $0.kind == .example }
        XCTAssertGreaterThanOrEqual(examples.count, 3)
        XCTAssertTrue(examples.allSatisfy { $0.url.scheme == "https" && $0.source.contains("대법원") })
        let first = reply("은행이 계좌 거래내역을 조회해 문제가 된 판례 있어?")
        XCTAssertTrue(first.source.contains("2008도3598"))
        let call = reply("대출 상담 전화 사칭 사건도 있어?")
        XCTAssertTrue(call.source.contains("2016도11138"))
        var history = [ConsentTurn(question: "계좌 거래내역 조회 사례", reply: first),
                       ConsentTurn(question: "대출 상담 전화 사칭 사건", reply: call)]
        XCTAssertTrue(reply("출처도 읽어줘", history).text.contains("2016도11138"))
        XCTAssertEqual(reply("아까 계좌 거래내역 사례 다시 들려줘", history), first)
        XCTAssertNotEqual(reply("다른 사례도 있어?", history).entryID, call.entryID)
        let documentTurn = ConsentTurn(question: "누구에게 제공해?", reply: reply("누구에게 제공해?"))
        history.append(documentTurn)
        XCTAssertEqual(reply("방금 뭐라고 했어?", history), documentTurn.reply)
        XCTAssertEqual(reply("방금 뭐라고 했어?", []).kind, .notice)
        XCTAssertEqual(reply("암호화폐 거래소 해킹 사례가 있어?").kind, .notice)
        XCTAssertEqual(reply("암호화폐 거래소 대출 사례가 있어?").kind, .notice)
        XCTAssertEqual(reply("근로복지공단에서 이런 유출 사건이 났어?").kind, .notice)
        XCTAssertEqual(reply("이중 수혜가 무슨 뜻이야?").kind, .term)
        XCTAssertEqual(reply("수집 항목이 뭐야?").kind, .document)
        XCTAssertEqual(reply("고유식별정보가 뭔가요?").kind, .term)
        XCTAssertEqual(reply("분할 대부가 뭐야?").entryID, "instalments")
        XCTAssertTrue(reply("대출 개인정보가 잘못 쓰인 실제 사례가 있어?").kind == .example)
        let term = reply("고유식별정보가 뭔가요?")
        XCTAssertTrue(reply("그게 무슨 뜻이야?", [.init(question: "고유식별정보가 뭔가요?", reply: term)]).text.contains("운전면허"))
        XCTAssertEqual(reply("아까 비밀번호 변경 사례 다시 들려줘", []).kind, .notice)
    }

    func testBundledQwenSelectsRetrievedEvidenceOffline() throws {
        guard DocumentLanguageModel.weights != nil else { throw XCTSkip("Bundled Qwen weights required for inference.") }
        let model = try DocumentLanguageModel(cancellation: ReviewCancellation())
        let group = try XCTUnwrap(ConsentEntry.library.first { $0.id == "bank-records" })
        let other = try XCTUnwrap(ConsentEntry.library.first { $0.id == "card-purpose" })
        let search = ConsentSearch(candidates: [other.answer(), group.answer()], certainMatch: nil, missing: ConsentConversation.noCase)
        let selected = try DocumentReviewer.selectAnswer("은행 계좌 거래내역을 서면 동의 없이 조회한 대출회사 사건을 알려줘.", search: search, model: model)
        XCTAssertEqual(selected.entryID, group.id, selected.spoken)
    }

    func testPhotographedLoanConsentWhenFixtureProvided() async throws {
        guard let path = ProcessInfo.processInfo.environment["DOCAGENT_CONSENT_PHOTO"] else {
            throw XCTSkip("Optional image of the consent page for OCR replay.")
        }
        let photo = try Data(contentsOf: URL(fileURLWithPath: path))
        let lines = try PhotoReader.read(photo)
        XCTAssertGreaterThan(lines.count, 15)
        let reader = DocumentReviewer()
        let identity = try await reader.answer("이거 어떤 서류야?", lines: lines, history: [])
        XCTAssertEqual(identity.kind, .document)
        XCTAssertTrue(identity.text.contains("동의서"), identity.text)
        XCTAssertTrue(identity.text.contains("대부 신청"), identity.text)
        let overview = try await reader.answer("한 줄로 요약해줘", lines: lines, history: [])
        XCTAssertEqual(overview.kind, .document, overview.text)
        XCTAssertTrue(ConsentQuery(overview.text).has(["개인정보", "대부", "동의"]), overview.text)
        XCTAssertTrue(overview.source.contains("줄 번호"))
        let first = try await reader.answer("내 정보는 누구에게 제공돼?", lines: lines, history: [])
        XCTAssertEqual(first.kind, .document)
        XCTAssertTrue(ConsentQuery(first.text).has(["한국고용정보원"]), first.text)
        let q = "대출 상담을 사칭한 실제 사례가 있어?"
        let external = try await reader.answer(q, lines: lines, history: [.init(question: "누구에게 제공해?", reply: first)])
        XCTAssertTrue(external.source.contains("2016도11138"), external.text)
        let history = [ConsentTurn(question: q, reply: external)]
        let back = try await reader.answer("그럼 내 동의서에서 동의를 거부하면?", lines: lines, history: history)
        XCTAssertEqual(back.kind, .document)
        XCTAssertTrue(ConsentQuery(back.text).has(["필수", "가능"]), back.text)
        let collection = reply("수집 항목이 뭐야?", page: lines)
        XCTAssertTrue(ConsentQuery(collection.text).has(["고용보험"]), collection.text)
        let retentionQuestion = "공단은 얼마나 오래 보관해?"
        let internalRetention = reply(retentionQuestion, page: lines)
        XCTAssertTrue(ConsentQuery(internalRetention.text).has(["최대 10년"]), internalRetention.text)
        let publicRetention = reply("그럼 제공받은 곳은?", [.init(question: retentionQuestion, reply: internalRetention)], page: lines)
        XCTAssertTrue(ConsentQuery(publicRetention.text).has(["목적을 달성"]), publicRetention.text)
        XCTAssertFalse(publicRetention.text.contains("10년"), publicRetention.text)
        XCTAssertEqual(reply("그럼 내가 몇 년 안에 갚아야 해?", page: lines).kind, .notice)
    }
}
