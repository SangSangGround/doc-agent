import CoreGraphics
import Foundation

/// On-device lexical retrieval. Search aliases improve recall; only photographed rows are evidence.
enum PageRetrieval {
    struct Passage {
        let anchor: PrintedLine
        let lines: [PrintedLine]
    }

    // The first form is the hint sent to the model when the user used a different expression.
    private static let aliases = [
        ["임대인", "집주인", "건물주", "집 주인"],
        ["임차인", "세입자", "세든 사람"],
        ["동의", "허락", "승낙"],
        ["열람", "보려면", "보려면은", "볼 수", "볼 때", "보려고"],
        ["수수료", "비용", "돈 내", "돈을 내", "돈이 들어", "돈 들어", "돈 드나", "유료", "무료"],
        ["제출서류", "첨부서류", "구비서류", "준비물", "첨부", "제출", "챙겨", "챙길", "가져가", "뭐가 필요", "뭘 준비"],
        ["연락처", "전화번호", "문의", "전화하려", "전화 걸", "전화하면", "연락하려"],
        ["처리기간", "처리 시간", "소요 시간", "소요 기간", "얼마나 기다", "얼마나 걸", "며칠 걸", "언제 나와", "언제 나올"],
        ["기한", "마감", "언제까지", "마지막 날", "기한이 언제"],
        ["보관", "보유", "보존", "저장"],
        ["반납", "돌려줘", "돌려주", "돌려줄", "반납함"],
        ["휴관", "휴업", "휴무", "쉬는", "문을 닫", "문 닫"],
        ["서명", "사인", "싸인"],
        ["신분증", "신분을 증명", "본인 확인 서류"],
        ["신청 자격", "지원 대상", "신청 대상", "신청할 수 있는 사람"],
        ["납부", "납입", "내야 하는 금액"]
    ]
    private static let consentNouns = ["동의", "허락", "승낙"]
    private static let permissionAbsent = ["없이", "없을", "없어", "없는", "없다면", "안받", "받지않", "받지못",
                                           "하지않", "못받", "하지말", "안해", "안하"]
    private static let permissionPresent = ["받아", "받았", "받으면", "받은", "받고", "있으면", "있는", "있어", "했"]

    /// true = with consent, false = without, nil = neither explicit nor unambiguous.
    /// Negation must modify permission itself ("동의를 받고 수수료 없이" is still with permission).
    static func consentCondition(in text: String) -> Bool? {
        let compact = ConsentQuery.compact(text)
        func mentions(_ qualifiers: [String]) -> Bool {
            consentNouns.contains { noun in
                ["", "이", "가", "을", "를"].contains { particle in
                    qualifiers.contains { compact.contains(noun + particle + $0) }
                }
            }
        }
        let positive = mentions(permissionPresent)
        let negative = mentions(permissionAbsent)
        return positive == negative ? nil : positive
    }
    private static let stopWords: Set<String> = ["어떤", "무엇", "어디", "어디다", "어디로", "언제", "관련", "알고", "싶어",
                                               "있으면", "하는", "해야", "하면", "하려면", "여기", "그때", "현재", "질문", "앞서", "물은"]

    private struct Concept {
        let label: String
        let variants: [String]
    }

    private static func concepts(for query: ConsentQuery) -> [Concept] {
        var groups = aliases.filter { query.has($0) }
        // "집주인이 허락 안 해 줬어" and "집주인이 허락했어" must stay distinguishable.
        if let permission = consentCondition(in: query.raw) {
            if permission {
                groups.append(["동의를 받아", "동의를 받은", "동의를 받고"])
            } else {
                groups.append(["동의 없이", "없이", "받지 않고"])
            }
        }
        if query.has(["주차", "재료비", "수강료", "교통비", "배송비", "보증금"]), !query.has(["수수료"]) {
            groups.removeAll { $0.first == "수수료" }
        }
        return groups.map { Concept(label: $0[0], variants: $0.map(ConsentQuery.compact)) }
    }

    static func searchTerms(for question: String) -> [String] {
        let q = ConsentQuery(question)
        return concepts(for: q).map(\.label).filter { !q.has([$0]) }
    }

    // A mention of an object or its location doesn't establish that it can be borrowed.
    // Use request forms of the verb; the past modifier in "빌린 책 어디다 돌려줘?" is not a loan request.
    static func supportsRequestedAction(question: String, evidence: String) -> Bool {
        let q = ConsentQuery(question), source = ConsentQuery(evidence)
        if q.has(["빌려줘", "빌려주", "빌려쓸", "빌릴", "빌리려", "대여하려", "대여할", "대여해", "대출받으려"]) {
            return source.has(["대여", "대출", "대부", "빌리", "빌려", "빌릴"])
        }
        if q.has(["돌려줘", "돌려주려", "반납하려", "반납할"]) {
            return source.has(["반납", "돌려주", "반환", "반송", "회수함"])
        }
        return true
    }

    static func passages(for question: String, in lines: [PrintedLine]) -> [Passage] {
        guard !lines.isEmpty else { return [] }
        let query = ConsentQuery(question)
        let permission = consentCondition(in: question)
        var terms = concepts(for: query)
        for word in query.words.sorted() where !stopWords.contains(word) {
            // Dictated inflections such as "참가하려는데" can refer to a printed noun "참가".
            let end = word.range(of: #"^.{2,}?(?=하려|했는데|하는데|하고싶|할건데|했을)"#, options: .regularExpression)
            let key = end.map { String(word[$0]) } ?? word
            if !terms.contains(where: { $0.variants.contains(key) }) {
                terms.append(Concept(label: key, variants: [key]))
            }
        }
        let text = lines.map { ConsentQuery.compact($0.text) }
        let hits = text.map { row in
            Set(terms.indices.filter { terms[$0].variants.contains(where: row.contains) })
        }
        var frequencies = Array(repeating: 0, count: terms.count)
        for row in hits {
            for term in row { frequencies[term] += 1 }
        }
        // Terms that occur in nearly every field are weaker anchors than distinctive subjects.
        let weights = frequencies.map { 1 + log(Double(lines.count + 1) / Double($0 + 1)) }
        var ranked: [(index: Int, score: Double, context: [Int])] = []
        for index in lines.indices where !hits[index].isEmpty {
            let context = neighborhood(around: index, in: lines)
            var scores: [Int: Double] = [:]
            let position = context.firstIndex(of: index) ?? 0
            for (offset, neighbor) in context.enumerated() {
                let distance = abs(position - offset)
                let factor = distance == 0 ? 1.0 : 0.58 / Double(distance)
                for term in hits[neighbor] {
                    scores[term] = max(scores[term] ?? 0, weights[term] * factor)
                }
            }
            var score = scores.values.sorted().reduce(0, +)
            if hits[index].contains(where: { terms[$0].variants.contains(text[index]) }) {
                score += 1  // A printed field label is a stronger anchor than a passing mention.
            }
            if let permission {
                // OCR can split "임대인의 동의" and "없이 ..." across two lines.
                let start = max(0, position - 1)
                let clause = context[start...position].map { text[$0] }.joined()
                if let match = consentCondition(in: text[index]) ?? consentCondition(in: clause) {
                    score *= match == permission ? 1.6 : 0.5
                }
            }
            ranked.append((index, score, context))
        }
        ranked.sort { $0.score == $1.score ? $0.index < $1.index : $0.score > $1.score }
        var result: [Passage] = []
        var covered = Set<Int>()
        var anchoredTerms = Set<Int>()
        for candidate in ranked {
            // Different fields can have overlapping context. Preserve the new anchor when it
            // covers a distinct subject, even if a previous window included its label.
            if hits[candidate.index].isSubset(of: anchoredTerms), covered.contains(candidate.index),
               candidate.context.filter({ !covered.contains($0) }).count <= 2 { continue }
            result.append(Passage(anchor: lines[candidate.index], lines: candidate.context.map { lines[$0] }))
            covered.formUnion(candidate.context)
            anchoredTerms.formUnion(hits[candidate.index])
            if result.count == 3 { break }
        }
        return result
    }

    private static func neighborhood(around index: Int, in lines: [PrintedLine]) -> [Int] {
        let linear = Array(max(0, index - 3)...min(lines.count - 1, index + 3))
        guard let anchor = lines[index].bounds, anchor.height > 0 else { return linear }
        // Vision may emit every label before any of its corresponding values. Prefer the
        // physical column and the nearest right-hand cell on the same baseline.
        var above: [Int] = [], below: [Int] = []
        var beside: Int?
        let shortLabel = lines[index].text.count <= 18 && anchor.width <= 0.35
        for i in lines.indices where i != index {
            guard let box = lines[i].bounds else { continue }
            let height = max(anchor.height, box.height)
            let dy = box.midY - anchor.midY
            let overlap = min(box.maxX, anchor.maxX) - max(box.minX, anchor.minX)
            if shortLabel, abs(dy) <= 0.65 * height, box.minX >= anchor.maxX - 0.15 * height,
               beside.map({ box.minX < (lines[$0].bounds?.minX ?? 1) }) ?? true {
                beside = i
            } else if abs(dy) > 0.45 * height, abs(dy) <= 8 * height,
                      overlap >= 0.4 * max(box.width, anchor.width) || abs(box.minX - anchor.minX) < 2.5 * height {
                if dy > 0 { above.append(i) } else { below.append(i) }
            }
        }
        func nearest(_ candidates: [Int]) -> [Int] {
            candidates.sorted { a, b in
                let da = abs((lines[a].bounds?.midY ?? 0) - anchor.midY)
                let db = abs((lines[b].bounds?.midY ?? 0) - anchor.midY)
                return da == db ? a < b : da < db
            }
        }
        // Prefer contiguous source order when no spatial neighbor was found.
        if above.isEmpty && below.isEmpty && beside == nil { return linear }
        var after = beside.map { [$0] } ?? []
        after += nearest(below).prefix(3 - after.count)
        return Array(nearest(above).prefix(3).reversed()) + [index] + after
    }
}
