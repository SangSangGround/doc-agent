import CryptoKit
import Foundation

/// Available only with a verified match. Every prepared answer still needs its evidence in this photo.
public enum DemoDocumentAnswers {
    public static func prepared(_ question: String, match: DemoConsentForm.Match, page: [PrintedLine]) -> ConsentReply? {
        func compact(_ text: String) -> String {
            String(text.precomposedStringWithCanonicalMapping.filter { $0.isLetter || $0.isNumber })
        }
        let query = compact(question)
        let phrases: [String]
        let answer: String
        switch query {
        case "이거어떤서류야", "이문서는어떤서류야", "한줄로요약해줘":
            phrases = ["개인정보의수집이용제공동의서", "직업훈련생계비대부신청"]
            answer = "직업훈련생계비 대부를 신청할 때 근로복지공단이 개인정보를 수집·이용하고 제3자에게 제공하는 데 동의를 구하는 서류예요."
        case "내정보는누구에게제공돼", "누가내정보를받아", "개인정보를제공받는기관은어디야":
            phrases = ["제공받는자", "고용노동부및한국고용정보원", "사업평가업무수행기관"]
            answer = "제공받는 곳으로 고용노동부 및 한국고용정보원, 사업평가 업무 수행기관이 적혀 있어요."
        case "동의를거부하면어떻게돼", "동의를거부할수있어":
            phrases = ["거부할수있습니다", "동의하셔야만직업훈련생계비대부신청이가능합니다"]
            answer = "개인정보 수집·이용·제공을 거부할 수 있어요. 이 서식에는 해당 사항에 동의해야 직업훈련생계비 대부를 신청할 수 있다고 적혀 있어요."
        default:
            return nil
        }
        guard page.count == match.lines.count,
              zip(page, match.lines).allSatisfy({ $0.id == $1.id && $0.text == $1.text }) else { return nil }
        var cited = [PrintedLine]()
        for phrase in phrases {
            guard let line = page.first(where: { !$0.uncertain && compact($0.text).contains(phrase) }) else { return nil }
            if !cited.contains(where: { $0.id == line.id }) { cited.append(line) }
        }
        return ConsentReply(kind: .document, text: answer,
                            source: "시연 양식의 저장 답변. " + ConsentReply.page(cited).source,
                            url: nil, entryID: nil)
    }
}

/// Exact document, prompt, confidence flags, source library and conversational context must match.
public struct SessionAnswerCache {
    private var answers: [Data: ConsentReply] = [:]
    private var oldestFirst = [Data]()
    public init() {}

    /// A just-answered, explicitly named document clause can be replayed once the new turn
    /// has been appended. “그럼?”, “다른 사례는?” and revision requests still need full dialogue.
    public static func permitsImmediateRepeat(_ question: String) -> Bool {
        let compact = question.filter { $0.isLetter || $0.isNumber }
        let clauses = ["보유기간", "보관기간", "수집목적", "제공목적", "제공기관", "제공받는자", "거부할권리", "수집항목"]
        let dependsOnTurn = ["그", "이전", "다른", "다시", "비교", "방금", "더", "전보다", "이번", "예시"]
        return clauses.contains(where: compact.contains) && !dependsOnTurn.contains(where: compact.contains)
    }

    public static func key(for request: CloudQuestionRequest) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return Data(SHA256.hash(data: try encoder.encode(request)))
    }

    public func lookup(_ key: Data) -> ConsentReply? {
        guard let reply = answers[key] else { return nil }
        return ConsentReply(kind: reply.kind, text: reply.text, source: "이 문서에서 저장한 답변. " + reply.source,
                            url: reply.url, entryID: reply.entryID, choices: reply.choices)
    }

    public mutating func record(_ reply: ConsentReply, for key: Data) {
        guard reply.kind != .notice else { return }
        if answers[key] == nil { oldestFirst.append(key) }
        answers[key] = reply
        if oldestFirst.count > 16 { answers.removeValue(forKey: oldestFirst.removeFirst()) }
    }
}
