import Foundation

public struct ConsentReply: Sendable, Equatable {
    public enum Kind: String, Sendable { case document, example, term, notice }
    public let kind: Kind
    public let text: String
    public let source: String
    public let url: URL?
    let entryID: String?

    public var spoken: String {
        switch kind {
        case .document: return "지금 찍은 문서에서 읽어드릴게요. " + text
        case .example: return "다른 사건의 사례예요. " + text
        case .term: return "용어 설명이에요. " + text
        case .notice: return text
        }
    }

    static func notice(_ message: String) -> Self {
        Self(kind: .notice, text: message, source: "", url: nil, entryID: nil)
    }

    public static func page(_ lines: [PrintedLine]) -> Self {
        let numbers = lines.map { String($0.id + 1) }
        let reference = numbers.count <= 8 ? "촬영한 문서의 인식 줄 번호 " + numbers.joined(separator: ", ") : "촬영한 문서의 원문"
        return Self(kind: .document, text: lines.map(\.spoken).joined(separator: "\n"),
                    source: reference, url: nil, entryID: nil)
    }
}

public struct ConsentTurn: Identifiable, Sendable {
    public let id = UUID()
    public let question: String
    public let reply: ConsentReply
    public init(question: String, reply: ConsentReply) {
        self.question = question
        self.reply = reply
    }
}

struct ConsentEntry: Decodable {
    enum Kind: String, Decodable { case example, term }
    let id: String
    let kind: Kind
    let title: String
    let keywords: [String]
    let summary: String
    let detail: String
    let source: String
    let url: URL

    func answer(detailOnly: Bool = false) -> ConsentReply {
        let body = detailOnly ? detail : summary
        return ConsentReply(kind: kind == .example ? .example : .term, text: title + ". " + body,
                            source: source, url: url, entryID: id)
    }

    static let library: [ConsentEntry] = {
        guard let file = Bundle.module.url(forResource: "consent-library", withExtension: "json"),
              let data = try? Data(contentsOf: file),
              let entries = try? JSONDecoder().decode([ConsentEntry].self, from: data) else { return [] }
        return entries
    }()
}

/// Korean OCR and dictation often disagree about spacing. Match topic phrases across spaces,
/// but retain the original evidence (including numbers and uncertain lines) for speech.
struct ConsentQuery {
    let raw: String
    let compact: String
    let words: Set<String>

    init(_ text: String) {
        raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalized = text.precomposedStringWithCanonicalMapping.lowercased()
        compact = Self.compact(normalized)
        words = Set(normalized.split { !$0.isLetter && !$0.isNumber }.map(String.init)
            .compactMap(Self.keyword))
    }

    static func compact(_ text: String) -> String {
        text.precomposedStringWithCanonicalMapping.lowercased().filter { $0.isLetter || $0.isNumber }
            .replacingOccurrences(of: "제삼자", with: "제3자")
    }

    private static func keyword(_ word: String) -> String? {
        let endings = ["에서는요", "에서는", "에게도", "이라는", "에서도", "이라고", "이랑", "에게", "이나", "에는",
                       "에서", "으로", "하고", "까지", "한테", "처럼", "하면", "라는", "은", "는", "이", "가", "을", "를", "에", "도", "와", "과"]
        var stem = word
        if let end = endings.first(where: { stem.hasSuffix($0) && stem.count - $0.count >= 2 }) {
            stem = String(stem.dropLast(end.count))
        }
        let noise: Set<String> = ["이거", "이런", "그런", "저런", "그럼", "그건", "그러면", "혹시", "나도", "내가", "우리",
                                  "대한", "관한", "관련", "내용", "동의서", "문서", "알려", "알려줘", "말해줘", "읽어줘",
                                  "있어", "있나요", "있어요", "어떻게", "적혀", "뭐야", "다시", "아까", "정보", "개인정보"]
        return stem.count >= 2 && !noise.contains(stem) ? stem : nil
    }

    func has(_ phrases: [String]) -> Bool { phrases.contains { compact.contains(Self.compact($0)) } }

    func overlap(with text: String) -> Int {
        let target = Self.compact(text)
        return words.reduce(0) { $0 + (target.contains(Self.compact($1)) ? min(5, $1.count) : 0) }
    }

    func matches(_ keywords: [String]) -> Int {
        Set(keywords.map(Self.compact)).reduce(0) { $0 + (compact.contains($1) ? min(6, $1.count) : 0) }
    }

    var asksSource: Bool { has(["출처", "사건번호", "근거 어디", "어디 자료", "몇 번째 줄", "몇 번 줄", "몇 줄에"]) }
    var asksReplay: Bool { has(["다시 들려", "다시 읽", "다시 말", "다시 알려", "다시 듣", "반복", "뭐라고 했", "방금 뭐", "아까 뭐"]) }
    var asksExamples: Bool { has(["사례", "사건", "판례", "범죄", "실제로 문제", "문제가 된 적", "피해 본 적", "피해가 난 적", "사고가 난", "처벌받은 적"]) }
    var asksDefinition: Bool { has(["무슨 뜻", "어떤 뜻", "뜻이", "뜻은", "뜻 좀", "뜻 알려", "무슨 말", "무슨 의미", "란 뭐", "쉽게 설명", "용어 설명", "뭐를 말", "라는 게 뭐"]) }
    var explicitPage: Bool { has(["이 문서", "이 동의서", "내 문서", "내 동의서", "찍은 동의서", "지금 찍", "촬영한", "내가 받은", "지금 서류", "내 서류", "이번 신청서", "내 신청서", "이 신청서"]) }

    func directlyNames(_ entry: ConsentEntry) -> Bool {
        let endings = ["뭐야", "뭔데", "뭔가요", "뭐예요", "뭐에요", "무엇인가요"]
        let particles = ["", "이", "가", "은", "는", "란", "이란"]
        return entry.keywords.contains { key in
            let phrase = Self.compact(key)
            guard compact.hasPrefix(phrase) else { return false }
            let rest = String(compact.dropFirst(phrase.count))
            return particles.contains { particle in endings.contains { rest == particle + $0 } }
        }
    }

    static func promptData(_ text: String) -> String {
        text.replacingOccurrences(of: "<", with: "＜").replacingOccurrences(of: ">", with: "＞")
            .replacingOccurrences(of: "\0", with: "")
    }
}

/// Only the retrieved candidates can be spoken. Even a confident but invalid model output
/// cannot invent an amount, a legal finding, or a source.
struct ConsentSearch {
    let candidates: [ConsentReply]
    let certainMatch: ConsentReply?
    let missing: ConsentReply
}

enum ConsentPlan {
    case direct(ConsentReply)
    case search(ConsentSearch)
}
