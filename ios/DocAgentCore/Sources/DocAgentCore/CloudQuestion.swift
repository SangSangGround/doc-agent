import Foundation

/// Only text and normalized OCR coordinates cross the network. IDs match PrintedLine.id.
public struct CloudQuestionRequest: Encodable, Sendable {
    public struct Line: Encodable, Sendable {
        public let id: Int
        public let text: String
        public let uncertain: Bool
        public let bounds: [Double]?
    }
    public struct Turn: Encodable, Sendable {
        public let question: String
        public let answer: String
    }
    public struct Reference: Encodable, Sendable {
        public let id: String
        public let kind: String
        public let text: String
    }

    public let question: String
    public let lines: [Line]
    public let focusLineIDs: [Int]
    public let history: [Turn]
    public let references: [Reference]

    static func makeLines(_ original: [PrintedLine]) throws -> [Line] {
        guard !original.isEmpty, original.count <= 500, original.allSatisfy({ $0.text.unicodeScalars.count <= 2000 }) else {
            throw CloudQuestionError.pageLength
        }
        return original.map { row in
            let unit = CGRect(x: 0, y: 0, width: 1, height: 1)
            let box = row.bounds.map { $0.intersection(unit) }
            let coordinates: [Double]? = box.flatMap { rect in
                guard !rect.isNull, !rect.isEmpty else { return nil }
                return [Double(rect.minX), Double(rect.minY), Double(rect.width), Double(rect.height)]
            }
            return Line(id: row.id, text: row.text, uncertain: row.uncertain, bounds: coordinates)
        }
    }

    public init(question: String, lines original: [PrintedLine], history previous: [ConsentTurn]) throws {
        self.question = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !self.question.isEmpty, self.question.unicodeScalars.count <= 2000 else { throw CloudQuestionError.questionLength }
        lines = try Self.makeLines(original)
        history = previous.filter { $0.question != "문서 원문 읽기" }.suffix(8).map {
            Turn(question: String($0.question.unicodeScalars.prefix(2000)),
                 answer: String(($0.reply.spoken + ($0.reply.source.isEmpty ? "" : "\n근거: " + $0.reply.source)).unicodeScalars.prefix(4000)))
        }
        var seen = Set<Int>()
        focusLineIDs = PageRetrieval.passages(for: self.question, in: original)
            .flatMap { $0.lines.map(\.id) }.filter { seen.insert($0).inserted }
        references = ConsentEntry.library.map {
            Reference(id: $0.id, kind: $0.kind.rawValue,
                      text: [$0.title, $0.summary, $0.detail, $0.source].joined(separator: "\n"))
        }
        let characters = self.question.unicodeScalars.count + lines.reduce(0) { $0 + $1.text.unicodeScalars.count }
            + history.reduce(0) { $0 + $1.question.unicodeScalars.count + $1.answer.unicodeScalars.count }
            + references.reduce(0) { $0 + $1.text.unicodeScalars.count }
        guard characters <= 100_000 else { throw CloudQuestionError.pageLength }
    }
}

public enum CloudQuestionError: LocalizedError {
    case questionLength, pageLength, invalidEvidence

    public var errorDescription: String? {
        switch self {
        case .questionLength: return "질문을 2,000자 이내로 입력해 주세요."
        case .pageLength: return "한 번에 확인할 수 있는 원문 분량을 벗어났어요. 질문할 부분을 다시 촬영해 주세요."
        case .invalidEvidence: return "답변의 근거를 확인하지 못했어요. 다시 질문해 주세요."
        }
    }
}

public struct CloudQuestionResponse: Decodable, Sendable {
    public struct LineContext: Decodable, Sendable {
        public let id: Int
        public let score: Float
    }
    public let kind: ConsentReply.Kind
    public let text: String
    public let lineIDs: [Int]
    public let referenceID: String?
    public let choices: [String]
    public let lineContexts: [LineContext]?

    /// Resolve citations against the exact page the caller sent; source URLs come from the bundled library.
    public func resolve(for request: CloudQuestionRequest, page: [PrintedLine]) throws -> (reply: ConsentReply, lines: [PrintedLine]) {
        let body = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !body.isEmpty, page.count == request.lines.count,
              zip(page, request.lines).allSatisfy({ $0.id == $1.id && $0.text == $1.text })
        else { throw CloudQuestionError.invalidEvidence }
        switch kind {
        case .document:
            let cited = Set(lineIDs)
            let submitted = Set(request.lines.map(\.id))
            guard !cited.isEmpty, cited.count == lineIDs.count, cited.isSubset(of: submitted),
                  referenceID == nil, choices.isEmpty else { throw CloudQuestionError.invalidEvidence }
            var supports: [Int: Float] = [:]
            for context in lineContexts ?? [] {
                guard cited.contains(context.id), (0...1).contains(context.score), supports[context.id] == nil
                else { throw CloudQuestionError.invalidEvidence }
                supports[context.id] = context.score
            }
            let updated = page.map { line in supports[line.id].map { line.corroborated(by: $0) } ?? line }
            let evidence = updated.filter { cited.contains($0.id) }
            guard evidence.count == cited.count else { throw CloudQuestionError.invalidEvidence }
            let warning = evidence.contains(where: \.uncertain) ? "근거 중 인식이 불확실한 문구가 있어요. " : ""
            let reply = ConsentReply(kind: .document, text: warning + body, source: ConsentReply.page(evidence).source,
                                     url: nil, entryID: nil)
            return (reply, updated)
        case .example, .term:
            guard lineIDs.isEmpty, (lineContexts ?? []).isEmpty, choices.isEmpty,
                  let id = referenceID,
                  request.references.contains(where: { $0.id == id && $0.kind == kind.rawValue }),
                  let entry = ConsentEntry.library.first(where: { $0.id == id && $0.kind.rawValue == kind.rawValue })
            else { throw CloudQuestionError.invalidEvidence }
            return (.init(kind: kind, text: body, source: entry.source, url: entry.url, entryID: entry.id), page)
        case .notice:
            guard lineIDs.isEmpty, referenceID == nil, (lineContexts ?? []).isEmpty,
                  choices.count <= 4, choices.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })
            else { throw CloudQuestionError.invalidEvidence }
            return (choices.isEmpty ? .notice(body) : .clarification(body, choices: choices), page)
        }
    }
}

/// The proactive text-only review has no question, audio, reference library or conversation.
public struct CloudOCRReviewRequest: Encodable, Sendable {
    public let lines: [CloudQuestionRequest.Line]

    public init(page: [PrintedLine]) throws {
        lines = try CloudQuestionRequest.makeLines(page)
        guard lines.reduce(0, { $0 + $1.text.unicodeScalars.count }) <= 100_000 else {
            throw CloudQuestionError.pageLength
        }
    }
}

public struct CloudOCRReviewResponse: Decodable, Sendable {
    public let lineContexts: [CloudQuestionResponse.LineContext]

    public func applying(to page: [PrintedLine], request: CloudOCRReviewRequest) throws -> [PrintedLine] {
        guard page.count == request.lines.count,
              zip(page, request.lines).allSatisfy({ $0.id == $1.id && $0.text == $1.text }) else {
            throw CloudQuestionError.invalidEvidence
        }
        let targets = Set(request.lines.filter(\.uncertain).map(\.id))
        var scores = [Int: Float]()
        for item in lineContexts {
            guard targets.contains(item.id), (0...1).contains(item.score), scores[item.id] == nil else {
                throw CloudQuestionError.invalidEvidence
            }
            scores[item.id] = item.score
        }
        return page.map { line in scores[line.id].map { line.corroborated(by: $0) } ?? line }
    }
}
