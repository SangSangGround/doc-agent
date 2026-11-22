import XCTest
@testable import DocAgentCore

final class QuestionVoiceCommandTests: XCTestCase {
    func testOnlyCompleteAffirmativeRepliesAuthorizeSending() {
        for reply in ["네", "보내 주세요.", "맞아요!", "네, 맞아요.", "응 보내줘", "어, 맞아", "네 맞습니다", "예. 질문해 주세요"] {
            XCTAssertEqual(QuestionVoiceCommand.confirmationReply(reply), .yes, reply)
        }
    }

    func testAmbiguousAndMixedRepliesAskAgainWithoutReplacingTheQuestion() {
        for reply in ["네, 맞지 않아요", "네, 맞지 않아", "맞아요? 아니요", "네, 맞는지 몰라",
                      "네맞지않아요", "네?", "맞아요?", "네? 맞아요", "네 3", "네, 보내면 안 되지?",
                      "어", "어.", "어…", "어...", "네... 맞아요", "잘 모르겠는데요", "모르겠어요", ""] {
            XCTAssertEqual(QuestionVoiceCommand.confirmationReply(reply), .clarify, reply)
        }
    }

    func testRefusalsDoNotSendAndDocumentQuestionsAboutCancellationArePreserved() {
        for reply in ["아니", "아니요.", "틀렸어요", "다시 말할게요", "네, 보내지 마", "아니요, 보내지 마",
                      "응 취소", "질문 취소", "잠깐만", "멈춰"] {
            XCTAssertEqual(QuestionVoiceCommand.confirmationReply(reply), .no, reply)
        }
        for question in ["동의 취소 방법은?", "어떤 정보야?", "종료일 알려줘", "그만두면 어떻게 돼?"] {
            XCTAssertEqual(QuestionVoiceCommand.confirmationReply(question), .question(question), question)
        }
    }

    func testCorrectionsKeepDigitsUnitsAndQuestionMarksEvenInShortQuestions() {
        let examples = [
            ("아니, 3년?", "3년?"), ("아니, 30만 원?", "30만 원?"),
            ("아니, 3번 항목 알려줘", "3번 항목 알려줘"), ("아 니, 2026년 기준인가요?", "2026년 기준인가요?"),
            ("아니요, -3%?", "-3%?"), ("아니, 보유 기간 알려줘", "보유 기간 알려줘")
        ]
        for (spoken, expected) in examples {
            XCTAssertEqual(QuestionVoiceCommand.confirmationReply(spoken), .question(expected), spoken)
        }
    }

    func testOnlyStandaloneStopCommandsEndAConversation() {
        for reply in ["그만", "그만할게요.", "대화 종료", "됐어 고마워"] {
            XCTAssertTrue(QuestionVoiceCommand.isStopCommand(reply), reply)
        }
        for question in ["종료일 알려줘", "그만두면 어떻게 돼?", "끝까지 읽어줘", "종료 2", "종료?"] {
            XCTAssertFalse(QuestionVoiceCommand.isStopCommand(question), question)
        }
    }
}
