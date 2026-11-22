import Foundation

/// Read fields from the photographed page. The case library never supplies missing form values.
enum ConsentField: CaseIterable {
    case refusal, retention, purpose, disclosed, recipient, collected

    var headings: [String] {
        switch self {
        case .refusal: return ["동의를 거부할 권리", "동의를 거부할 경우", "거부할 권리", "동의 거부", "거부권리", "동의거부권리 및 불이익"]
        case .retention: return ["개인정보 보유 이용 기간", "보유 및 이용 기간", "보유 이용 기간", "보유기간", "보존기간", "보관 기간"]
        case .purpose: return ["수집 이용 목적", "수집목적", "제공받는 자의 이용 목적", "이용 목적", "제공 목적", "제공목적", "조회 목적"]
        case .disclosed: return ["제공할 개인정보의 항목", "제공하는 개인정보의 항목", "제공되는 개인정보 항목", "제공하는 항목", "제공 항목", "제공할 개인정보", "제공하는 개인정보"]
        case .recipient: return ["개인정보를 제공받는 자", "제공받는 자", "제공 받는 기관", "제공 대상자", "제공 대상 기관", "제공 기관"]
        case .collected: return ["수집 이용할 항목", "수집 이용 항목", "수집하려는 개인정보의 항목", "수집하는 개인정보 항목", "수집하는 개인정보의 항목", "수집 항목", "개인정보 항목"]
        }
    }

    static func requested(_ q: ConsentQuery) -> Self? {
        if q.has(["거절", "거부", "불이익", "필수", "안 하면", "안 해도", "안하면", "동의하지", "동의를 안", "동의 안", "동의해야", "꼭 동의"]) { return .refusal }
        if q.has(["보유", "보관", "보존", "파기", "몇 년", "언제까지", "얼마 동안", "얼마나 오래", "언제 지", "언제 없"]) { return .retention }
        if q.has(["왜", "목적", "어디에 쓰", "어디에 써", "어디다 쓰", "어디에 사용", "뭐에 쓰", "뭐에 써", "어디에 이용"]) { return .purpose }
        if q.has(["누구", "누가", "어디로", "어디에 넘", "어디에 제", "어디에 주", "어디에 줘", "어디에 보", "어디에 가", "어디가 받", "어느 곳", "어떤 곳", "어느 기관", "어떤 기관", "어느 회사", "어떤 회사", "받는 곳", "받는 자", "받는 사람", "받는 기관"]) { return .recipient }
        if q.has(["넘어", "넘겨", "제공 항목", "제공하는 정보", "어떤 정보가 제", "어떤 정보를 제", "어떤 항목을 제", "전달", "공유", "공개", "뭘 제공", "보내는 정보"]) { return .disclosed }
        if q.has(["어떤 정보", "무슨 정보", "수집", "수집 항목", "가져가", "모아", "모으", "어떤 항목", "받아가", "뭘 받", "무엇을 받"]) { return .collected }
        return nil
    }

    static func heading(_ text: String) -> (field: Self, key: String)? {
        for field in allCases {
            for key in field.headings.map(ConsentQuery.compact) {
                guard let range = text.range(of: key), text.distance(from: text.startIndex, to: range.lowerBound) <= 18 else { continue }
                let suffix = text[range.upperBound...]
                if suffix.hasPrefix("을달성") || suffix.hasPrefix("이달성") || suffix.hasPrefix("으로") { continue }
                return (field, key)
            }
        }
        return nil
    }
}

struct ConsentDocument {
    private enum Scope { case collection, provision }
    private struct Block {
        let field: ConsentField
        let scope: Scope?
        let key: String
        var lines: [PrintedLine]

        var hasValue: Bool {
            if lines.count > 1 { return true }
            let text = ConsentQuery.compact(lines[0].text)
            guard let end = text.range(of: key)?.upperBound else { return false }
            let rest = String(text[end...])
            return rest.count >= 2 && !["사항", "단체", "기관", "이용기간", "불이익"].contains(rest)
        }
    }

    private static func scope(_ q: ConsentQuery) -> Scope? {
        if q.has(["제3자", "제공", "넘겨", "넘어", "전달", "받은 곳", "받는 곳", "받는 기관", "받은 기관", "받는 쪽", "그 기관", "거기서"]) { return .provision }
        if q.has(["수집", "공단", "신청 기관", "처음 모"]) { return .collection }
        return nil
    }

    static func answeredField(_ turn: ConsentTurn) -> ConsentField? {
        guard turn.reply.kind == .document, turn.question != "문서 원문 읽기" else { return nil }
        for line in turn.reply.text.split(separator: "\n") {
            if let heading = ConsentField.heading(ConsentQuery.compact(String(line))) { return heading.field }
        }
        return ConsentField.requested(ConsentQuery(turn.question))
    }

    private static func blocks(_ lines: [PrintedLine]) -> [Block] {
        var blocks: [Block] = []
        var active: Block?
        var section: Scope?
        for line in lines {
            let s = ConsentQuery.compact(line.text)
            let sectionStart = s.hasSuffix("관한사항") || s.hasSuffix("제공동의") || s.hasSuffix("수집이용동의")
            if sectionStart || s.hasPrefix("기타고지") || s.hasPrefix("본인은") || s.hasPrefix("동의함") ||
                s.hasPrefix("이름") || s.hasPrefix("이동의는") || s.hasPrefix("년월일") {
                if let active { blocks.append(active) }
                active = nil
                if sectionStart { section = scope(ConsentQuery(line.text)) }
                continue
            }
            if let heading = ConsentField.heading(s) {
                if let active { blocks.append(active) }
                let local = scope(ConsentQuery(line.text.components(separatedBy: CharacterSet(charactersIn: ":：")).first ?? line.text))
                active = Block(field: heading.field, scope: local ?? section, key: heading.key, lines: [line])
            } else if active != nil {
                active?.lines.append(line)
            }
        }
        if let active { blocks.append(active) }
        return blocks.filter(\.hasValue)
    }

    static func plan(_ q: ConsentQuery, lines: [PrintedLine], history: [ConsentTurn]) -> ConsentPlan {
        let missing = ConsentReply.notice("촬영한 이 면에서 질문에 답할 문구를 확인하기 어려워요. 다른 면에 적혀 있다면 그 면을 촬영해 주세요.")
        guard !lines.isEmpty else { return .direct(missing) }
        // A consent can mention a number of years for retaining data without stating
        // the borrower's repayment period. Require evidence about the loan term itself.
        let contents = ConsentQuery(lines.map(\.text).joined(separator: " "))
        let consentPage = contents.has(["개인정보", "제공받는 자", "동의서", "동의 거부", "제공받은 기관"])
        let loanPage = contents.has(["대부", "대출", "융자"])
        if ["기간", "기간은", "기간은요", "언제까지", "언제까지야", "언제까지예요"].contains(q.compact) {
            var choices: [String] = []
            if contents.has(["신청 기한", "신청 기간", "접수 기간", "신청 마감"]) { choices.append("신청 기한") }
            if contents.has(["보유", "보관"]) { choices.append("개인정보 보유 기간") }
            if contents.has(["납부 기한", "납부 마감", "납부 기간"]) { choices.append("납부 기한") }
            if contents.has(["상환 기간", "거치 기간"]) { choices.append("대부 상환 기간") }
            if choices.count > 1 { return .direct(.clarification("어떤 기간을 말씀하시는 건가요?", choices: choices)) }
        }
        if ["얼마", "얼마야", "얼마예요", "금액은", "비용은", "비용은얼마야"].contains(q.compact) {
            let amounts = ["신청비", "수강료", "재료비", "참가비", "납부 금액", "환급 금액"].filter { contents.has([$0]) }
            if amounts.count > 1 { return .direct(.clarification("어떤 금액이 궁금하세요?", choices: Array(amounts.prefix(4)))) }
        }
        if loanPage, let loanTopic = loanTopic(q) {
            let passages = lines.filter { ConsentQuery($0.text).has(loanTopic) }
            let candidates = passages.prefix(3).map { ConsentReply.page([$0]) }
            return candidates.isEmpty ? .direct(missing) : .search(.init(candidates: candidates, certainMatch: nil, missing: missing))
        }
        let previous = history.last { $0.reply.kind == .document }
        let previousQuestion = previous.map { ConsentQuery($0.question) }
        let lastField = previous.flatMap(answeredField)
        let scopeOnly = ["제공받은곳은", "제공받는곳은", "제공받은기관은", "제공받는기관은", "제3자는", "공단은", "수집쪽은", "제공쪽은"].contains { q.compact.hasSuffix($0) || q.compact.hasSuffix($0 + "요") }
        let field = scopeOnly && lastField != nil ? lastField : ConsentField.requested(q) ?? (q.has(["그럼", "그건", "그거", "그쪽", "그곳"]) ? lastField : nil)
        let privacyQuestion = q.has(["동의", "개인정보", "보유", "보관", "파기", "수집", "제공", "거부", "불이익",
                                     "내 정보", "정보를", "어느 기관", "어떤 기관", "누구한테 줘", "누구에게 줘", "누구한테 가", "넘겨", "넘어가"])
        let asksAboutOtherRoles = q.has(["작성자", "작성했", "작성한", "발행", "발급", "서명해야", "서명하는", "신청 자격", "신청할 수", "문의", "준비물"])
        if let field, consentPage && !asksAboutOtherRoles && (privacyQuestion || scopeOnly || (q.has(["그럼", "거기"]) && lastField != nil)) {
            var matches = blocks(lines).filter { $0.field == field }
            let preferred = scope(q) ?? (q.has(["그럼", "거기", "그쪽", "그곳"]) ? previousQuestion.flatMap(scope) : nil)
            if preferred == nil, field == .retention, matches.contains(where: { $0.scope == .collection }),
               matches.contains(where: { $0.scope == .provision }) {
                return .direct(.clarification("어느 쪽의 보유 기간이 궁금하세요?", choices: ["수집 기관의 보유 기간", "제공받은 기관의 보유 기간"]))
            }
            if let preferred, matches.contains(where: { $0.scope == preferred }) {
                matches = matches.filter { $0.scope == preferred }
            } else if !matches.isEmpty && preferred != nil && [.retention, .purpose].contains(field) {
                matches = matches.filter { $0.scope == nil }
                if matches.isEmpty { return .direct(missing) }
            }
            if !matches.isEmpty {
                // Two retention clauses may state different limits. Read both instead of
                // silently attributing the lender's limit to a recipient, or vice versa.
                let answer = ConsentReply.page(matches.flatMap(\.lines))
                return .search(ConsentSearch(candidates: [answer], certainMatch: answer, missing: missing))
            }
            // Do not substitute, for example, collected items for a missing recipient.
            // Also support prose-only consents where a label is missing but the line
            // explicitly addresses the requested field.
            let plain = lines.filter { line in
                let content = ConsentQuery(line.text)
                guard ConsentField.heading(content.compact) == nil else { return false }
                switch field {
                case .recipient: return content.has(["에게 제공합니다", "에 제공합니다", "에 제공됩니다"])
                case .retention: return content.has(["보관합니다", "보유합니다", "후 파기"])
                case .refusal: return content.has(["신청이 불가", "거절하면", "거부하면"])
                case .purpose: return content.has(["위하여 사용", "위해 사용", "확인을 위해"])
                case .disclosed: return content.has(["항목을 전달", "정보를 제공합니다"])
                case .collected: return content.has(["을 수집합니다", "를 수집합니다"])
                }
            }
            if plain.isEmpty { return .direct(missing) }
            return .search(ConsentSearch(candidates: plain.prefix(3).map { .page([$0]) }, certainMatch: nil, missing: missing))
        }
        let continuing = q.has(["그럼", "그러면", "거기는", "거기서", "그곳", "그날", "그때", "그건"])
        if continuing, let last = history.last, last.reply.kind == .document && last.question != "문서 원문 읽기" {
            let contextual = "앞서 물은 내용: " + String(last.question.prefix(80)) + "\n현재 질문: " + q.raw
            return PageQuestion.plan(.detail, question: contextual, lines: lines)
        }
        return PageQuestion.plan(.detail, question: q.raw, lines: lines)
    }

    private static func loanTopic(_ query: ConsentQuery) -> [String]? {
        if query.has(["보유", "보관", "파기", "보존"]) { return nil }
        let repayment = ["상환 기간", "거치 기간", "상환 기한", "상환 기일", "분할 상환", "갚아야", "갚는 기간"]
        if query.has(repayment + ["갚아", "갚는", "언제 갚", "상환은", "상환하"]) { return repayment }
        let amount = ["대부 한도", "대출 한도", "대부 금액", "대출 금액", "융자 한도", "1인당 한도", "월 한도"]
        if query.has(amount + ["얼마 빌", "얼마나 빌", "얼마까지", "얼마 받을", "빌릴 수 있는 금액"]) { return amount }
        let rate = ["대부 금리", "대출 금리", "이자율", "대부 이율", "연이율", "대부 이자"]
        if query.has(["이자", "이율", "금리"]) { return rate }
        let deadline = ["신청 기한", "신청 마감", "접수 마감", "접수 기간", "신청 기간"]
        if query.has(deadline + ["언제까지 신청", "신청은 언제까지"]) { return deadline }
        return nil
    }
}
