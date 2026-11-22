import Foundation

public enum QuestionVoiceCommand {
    public enum ConfirmationReply: Equatable {
        case yes, no, clarify
        case question(String)
    }

    private static let acknowledgments = ["네", "예", "응", "어"]
    private static let affirmative: Set<String> = [
        "네", "예", "응", "네네", "응응", "ㅇㅇ", "오케이", "그래", "그래요", "좋아", "좋아요",
        "맞아", "맞아요", "맞습니다", "맞네", "맞네요", "보내", "보내줘", "보내줘요", "보내주세요",
        "질문해", "질문해줘", "질문해주세요", "그렇게해", "그렇게해줘", "그렇게해주세요"
    ]
    private static let refusals: Set<String> = [
        "보내지마", "보내지마요", "보내지마세요", "보내지말아줘", "보내지말아주세요", "안보내", "안보낼래",
        "하지마", "하지마세요", "안돼", "안돼요", "취소", "질문취소", "취소해", "취소해줘", "취소해주세요",
        "잠깐", "잠깐만", "잠깐만요", "멈춰", "멈춰줘", "기다려", "기다려줘"
    ]
    private static let correctionLeads = [
        "아니에요", "아니예요", "아니야", "아니요", "아니", "아뇨", "아냐",
        "틀렸어요", "틀렸어", "틀렸", "틀려", "다시"
    ]
    private static let rephraseFillers: Set<String> = [
        "", "요", "야", "에요", "예요", "어", "었어", "어요", "말할게", "말할게요", "말할래",
        "말해줘", "할게", "할게요", "그거아니야", "그게아니야"
    ]
    private static let stops: Set<String> = [
        "끝", "끝내", "끝내자", "끝낼게", "끝낼게요", "끝내줘", "끝이야", "대화끝",
        "그만", "그만해", "그만해요", "그만하자", "그만할게", "그만할게요", "그만할래", "그만해줘", "이제그만", "이제그만할게",
        "종료", "종료해", "종료해줘", "종료할게", "대화종료",
        "됐어", "됐어요", "이제됐어", "이제됐어요", "고마워", "고마워요", "고맙습니다", "감사합니다", "됐어고마워", "알겠어고마워", "고마워그만"
    ]

    // Strip a single sentence terminator. Preserve hesitation ("어..."), questions and numbers.
    private static func commandText(_ text: String) -> String {
        var compact = text.filter { !$0.isWhitespace }
        if let last = compact.last, ".!。！".contains(last) { compact.removeLast() }
        return compact
    }

    public static func confirmationReply(_ text: String) -> ConfirmationReply {
        let command = commandText(text)
        guard !command.isEmpty else { return .clarify }
        if affirmative.contains(command) { return .yes }
        if refusals.contains(command) { return .no }

        if let lead = acknowledgments.first(where: { command.hasPrefix($0) }) {
            // Recognize the entire second phrase in e.g. "네, 맞아요" or "응, 보내지 마".
            var suffix = command.dropFirst(lead.count)
            if let separator = suffix.first, ",，.".contains(separator) { suffix = suffix.dropFirst() }
            let remainder = String(suffix)
            if affirmative.contains(remainder) { return .yes }
            if refusals.contains(remainder) { return .no }
            if remainder.hasPrefix("맞") || remainder.hasPrefix("보내") { return .clarify }
        }
        if let lead = correctionLeads.first(where: { command.hasPrefix($0) }) {
            var rest = text[...]
            for _ in lead {
                rest = rest.drop(while: { $0.isWhitespace }).dropFirst()
            }
            rest = rest.drop { $0.isWhitespace || ",，:：".contains($0) }
            let correction = String(rest)
            let compactCorrection = commandText(correction)
            return rephraseFillers.contains(compactCorrection) || refusals.contains(compactCorrection) ? .no : .question(correction)
        }

        // Mixed or questioning replies retain the pending question for explicit confirmation.
        let firstWord = text.split { $0.isWhitespace || $0.isPunctuation }.first.map(String.init)
        let uncertainStarts = ["글쎄", "잘몰라", "잘모르겠", "모르겠", "확실하지않", "맞는지모", "맞지않"]
        if firstWord == "어" || firstWord.map(affirmative.contains) == true || command.hasPrefix("맞아") ||
            uncertainStarts.contains(where: { command.hasPrefix($0) }) {
            return .clarify
        }
        return .question(text)
    }

    public static func isStopCommand(_ text: String) -> Bool {
        stops.contains(commandText(text))
    }
}
