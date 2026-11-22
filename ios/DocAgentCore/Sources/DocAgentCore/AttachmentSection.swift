import Foundation

/// Reads a printed list with its conditions, including items that wrap onto another OCR line.
enum AttachmentSection {
    private static let headings = ["첨부서류", "구비서류", "제출서류", "신청인제출서류", "민원인제출서류", "첨부문서", "붙임서류"]
    private static let nextFields = ["수수료", "담당공무원확인", "담당자확인사항", "행정정보공동이용", "작성방법",
                                     "작성요령", "유의사항", "처리절차", "처리기간", "접수처", "제출처"]
    private static func endsSection(_ text: String) -> Bool {
        let s = ConsentQuery.compact(text)
        return nextFields.contains(where: s.hasPrefix) && !isItem(text)
    }

    static func requested(_ q: ConsentQuery) -> Bool {
        if q.asksExamples || q.asksDefinition || q.has(["왜", "이유", "언제", "어디", "누가", "몇 개", "몇 장", "발급", "대신", "대체"]) { return false }
        let namesField = q.has(headings + ["준비할 서류", "필요한 서류", "내야 할 서류", "챙길 서류"])
        let asksList = q.has(["어떤", "무슨", "뭐", "무엇", "목록", "뭘", "어떠", "뭔", "읽어", "알려", "전부"])
        if PageRetrieval.consentCondition(in: q.raw) != nil && asksList && q.has(["필요", "챙", "준비", "내야"]) { return true }
        if namesField { return asksList || headings.contains { label in
            ["", "는", "가있어", "가있나요", "는있어"].contains { q.compact == label + $0 }
        } }
        return asksList && q.has(["첨부", "구비", "제출", "같이 내", "챙겨야", "가져가야", "서류를 준비", "뭘 내야"])
    }

    static func answer(in lines: [PrintedLine], for question: ConsentQuery? = nil) -> ConsentReply? {
        for (index, line) in lines.enumerated() {
            let compact = ConsentQuery.compact(line.text)
            let inline = line.text.range(of: #"^\s*(?:(?:신청인|민원인)\s*)?(?:첨부|구비|제출|붙임)\s*(?:서류|문서)\s*(?:[:：]\s*|(?=\d+[.)])|\s+(?=\S))"#,
                                         options: .regularExpression)
            let split = ["첨부", "구비", "제출", "붙임"].contains(compact) && index + 1 < lines.count &&
                ConsentQuery.compact(lines[index + 1].text) == "서류"
            guard headings.contains(compact) || split || inline != nil else { continue }
            let labelEnd = split ? index + 1 : index
            let header = lines[labelEnd]
            if inline == nil, let cell = adjacentCell(to: header, on: lines) {
                return listReply(cell, for: question)
            }
            var evidence: [PrintedLine] = []
            if let inline {
                let value = String(line.text[inline.upperBound...])
                if !ConsentQuery.compact(value).isEmpty {
                    evidence.append(line.withText(value))
                }
            }
            for row in lines.dropFirst(labelEnd + 1) {
                if endsSection(row.text) { break }
                if !evidence.isEmpty, !isItem(row.text), row.text.count < 20,
                   ConsentQuery.compact(row.text).hasSuffix("동의서"), evidence.contains(where: { isItem($0.text) }) { break }
                if headings.contains(ConsentQuery.compact(row.text)) { continue }
                evidence.append(row)
            }
            return listReply(evidence, for: question)
        }
        // A prose-only page can still answer through ordinary evidence retrieval.
        return nil
    }

    private static let unreadable = ConsentReply.notice("첨부서류 제목은 보이지만 항목을 확인하기 어려워요. 첨부서류 칸 전체가 나오게 다시 찍어 주세요.")

    private static func listReply(_ lines: [PrintedLine], for question: ConsentQuery?) -> ConsentReply? {
        guard let first = lines.first else { return unreadable }
        guard let question, let condition = PageRetrieval.consentCondition(in: question.raw) else { return .page(lines) }
        // A condition can wrap to a second OCR row. Only split at printed item boundaries.
        guard isItem(first.text) else { return nil }
        var items: [[PrintedLine]] = []
        for line in lines {
            if isItem(line.text) { items.append([]) }
            items[items.count - 1].append(line)
        }
        let roles = [["임대인", "집주인"], ["임차인", "세입자"], ["보호자", "부모"], ["대리인"], ["신청인"]]
        let subject = roles.filter { question.has($0) }
        var applicable: [PrintedLine] = []
        var foundCondition = false
        for item in items {
            let content = ConsentQuery(item.map(\.text).joined(separator: " "))
            if let printed = PageRetrieval.consentCondition(in: content.raw) {
                guard printed == condition, subject.allSatisfy(content.has) else { continue }
                foundCondition = true
            }
            applicable += item
        }
        return foundCondition ? .page(applicable) : nil
    }

    private static func isItem(_ text: String) -> Bool {
        text.range(of: #"^\s*(?:[0-9가-하]+[.)]|[①-⑳•●▪-])"#, options: .regularExpression) != nil
    }

    // A left-hand field label can sit halfway down a multi-line value. Find the adjoining
    // column first, then read outward vertically. Vision observation order can mix columns.
    private static func adjacentCell(to header: PrintedLine, on lines: [PrintedLine]) -> [PrintedLine]? {
        guard let label = header.bounds else { return nil }
        let right = lines.compactMap { line -> (line: PrintedLine, box: CGRect)? in
            guard let box = line.bounds, box.minX >= label.maxX - label.height * 0.3 else { return nil }
            return (line, box)
        }
        let band = right.filter { abs($0.box.midY - label.midY) <= 1.7 * max(label.height, $0.box.height) }
        guard let neighbor = band.min(by: { $0.box.minX < $1.box.minX }) else { return nil }
        let left = neighbor.box.minX
        let height = max(neighbor.box.height, label.height)
        let nextColumn = band.filter { endsSection($0.line.text) }.map(\.box.minX).min() ?? 1
        guard left < nextColumn else { return [] }
        // Bound this row by other labels in the same left column. Small line spacing
        // alone does not separate adjacent rows in a compact form.
        let labels = lines.compactMap(\.bounds).filter {
            $0 != label && $0.minX < label.maxX && $0.midX < left && $0.maxX < left + height
        }
        let above = labels.filter { $0.minY > label.maxY }.map(\.minY).min() ?? 1
        let below = labels.filter { $0.maxY < label.minY }.map(\.maxY).max() ?? 0
        let column = right.filter {
            abs($0.box.minX - left) <= 4 * height && $0.box.midX < nextColumn &&
                $0.box.midY < above && $0.box.midY > below
        }.sorted { $0.box.midY > $1.box.midY }
        guard let center = column.firstIndex(where: { $0.line.id == neighbor.line.id }) else { return [] }
        var top = center, bottom = center
        func connected(_ a: Int, _ b: Int) -> Bool {
            let gap = column[a].box.minY - column[b].box.maxY
            return gap < 1.4 * max(column[a].box.height, column[b].box.height)
        }
        while top > 0, connected(top - 1, top), !endsSection(column[top - 1].line.text) { top -= 1 }
        while bottom + 1 < column.count, connected(bottom, bottom + 1), !endsSection(column[bottom + 1].line.text) { bottom += 1 }
        return column[top...bottom].map(\.line)
    }
}
