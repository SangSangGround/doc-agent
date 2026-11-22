import XCTest
@testable import DocAgentCore

final class ConversationMicrophoneTurnTests: XCTestCase {
    @MainActor func testSwitchingOffBeforeAutomaticListeningPreventsDelayedStart() async {
        let turn = ConversationMicrophoneTurn()
        let pending = turn.schedule()

        XCTAssertFalse(turn.cancel(), "The hardware has not started yet")
        XCTAssertFalse(turn.beginScheduled(pending))
        XCTAssertFalse(turn.finishScheduled(pending))
    }

    @MainActor func testSwitchingOffDuringAutomaticListeningStopsRecordingAndDropsLateTranscript() async {
        let turn = ConversationMicrophoneTurn()
        let active = turn.schedule()
        XCTAssertTrue(turn.beginScheduled(active))

        XCTAssertTrue(turn.cancel())
        XCTAssertFalse(turn.finishScheduled(active), "A late ‘네’ must never reach the confirmation handler")
        XCTAssertFalse(turn.cancel(), "The same automatic recording is stopped only once")
    }

    @MainActor func testReenablingDoesNotAcceptCallbacksFromBeforeSwitchOff() async {
        let turn = ConversationMicrophoneTurn()
        let old = turn.schedule()
        XCTAssertTrue(turn.beginScheduled(old))
        XCTAssertTrue(turn.cancel())

        let current = turn.schedule()
        XCTAssertFalse(turn.beginScheduled(old))
        XCTAssertTrue(turn.beginScheduled(current))
        XCTAssertFalse(turn.finishScheduled(old))
        XCTAssertTrue(turn.finishScheduled(current))
    }

    @MainActor func testOnlyLatestAutomaticTurnStartsAndDeliversOnce() async {
        let turn = ConversationMicrophoneTurn()
        let superseded = turn.schedule()
        let current = turn.schedule()

        XCTAssertFalse(turn.beginScheduled(superseded))
        XCTAssertFalse(turn.finishScheduled(current), "A pending recording cannot deliver a result")
        XCTAssertTrue(turn.beginScheduled(current))
        XCTAssertFalse(turn.beginScheduled(current))
        XCTAssertTrue(turn.finishScheduled(current))
        XCTAssertFalse(turn.finishScheduled(current))
    }

    @MainActor func testDisablingAutomaticListeningLeavesManualRecordingAlone() async {
        let turn = ConversationMicrophoneTurn()

        // A user-initiated recording does not schedule an automatic turn.
        XCTAssertFalse(turn.cancel(), "No automatic recording owns the microphone")
    }

    @MainActor func testCancellingAndReschedulingBeforeYieldStartsOnlyTheNewRecording() async {
        let turn = ConversationMicrophoneTurn()
        let restarted = expectation(description: "The newly scheduled microphone starts")

        turn.schedule(after: .zero) { _ in XCTFail("A cancelled turn must not reopen the microphone") }
        XCTAssertFalse(turn.cancel())
        turn.schedule(after: .zero) { id in
            XCTAssertTrue(turn.finishScheduled(id))
            restarted.fulfill()
        }

        await fulfillment(of: [restarted], timeout: 3)
    }
}
