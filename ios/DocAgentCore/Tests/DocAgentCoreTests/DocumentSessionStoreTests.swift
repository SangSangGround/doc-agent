import Foundation
import XCTest
@testable import DocAgentCore

final class DocumentSessionStoreTests: XCTestCase {
    private let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    private var store: DocumentSessionStore { DocumentSessionStore(directory: directory) }
    private let photo = Data([0x11, 0x23, 0x57])

    override func tearDownWithError() throws {
        if FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
        try super.tearDownWithError()
    }

    func testColdRestorePreservesDocumentAndUnsentWorkWithoutRewritingThePhoto() throws {
        let rect = CGRect(x: 0.2, y: 0.3, width: 0.05, height: 0.05)
        let line = PrintedLine(id: 9, text: "신청인 서명", confidence: 0.58,
                               bounds: rect, contextSupport: 0.9)
        let target = WritingTarget(id: "applicant", label: "서명", kind: .signature, bounds: rect)
        var checkpoint = DocumentCheckpoint()
        checkpoint.recognitionPending = true
        try store.replace(with: checkpoint, photo: photo)
        XCTAssertTrue(try XCTUnwrap(store.load()).checkpoint.recognitionPending)
        let photoFile = directory.appendingPathComponent(checkpoint.id.uuidString + ".photo")
        let firstInode = try FileManager.default.attributesOfItem(atPath: photoFile.path)[.systemFileNumber] as? NSNumber
        XCTAssertNotNil(firstInode)

        checkpoint.recognitionPending = false
        checkpoint.lines = [line]
        checkpoint.demoForm = .init(lines: [line], targets: [target])
        let reply = ConsentReply(kind: .example, text: "답변", source: "문서 10번 줄",
                                 url: URL(string: "https://example.org/evidence"), entryID: "source-id", choices: ["선택"])
        checkpoint.turns = [.init(question: "질문", reply: reply)]
        checkpoint.questionDraft = "작성 중인 글"
        checkpoint.questionToConfirm = "보내기 전 질문"
        checkpoint.readingIndex = 4
        checkpoint.screen = .writing
        checkpoint.writing = .init(referenceDigest: "verified-jpeg", fields: [target])
        checkpoint.selectedTargetID = target.id
        try store.save(checkpoint)

        let loaded = try XCTUnwrap(DocumentSessionStore(directory: directory).load())
        let restored = loaded.checkpoint
        XCTAssertEqual(loaded.photo, photo)
        XCTAssertEqual(restored.id, checkpoint.id)
        XCTAssertFalse(restored.recognitionPending)
        XCTAssertEqual(restored.lines.first?.bounds, rect)
        XCTAssertEqual(restored.lines.first?.contextSupport, 0.9)
        XCTAssertEqual(restored.lines.first?.uncertain, false)
        XCTAssertEqual(restored.demoForm?.targets, [target])
        XCTAssertEqual(restored.turns.first?.id, checkpoint.turns.first?.id)
        XCTAssertEqual(restored.turns.first?.reply, reply)
        XCTAssertEqual(restored.questionDraft, checkpoint.questionDraft)
        XCTAssertEqual(restored.questionToConfirm, checkpoint.questionToConfirm)
        XCTAssertEqual(restored.readingIndex, 4)
        XCTAssertEqual(restored.screen, .writing)
        XCTAssertEqual(restored.writing?.referenceDigest, "verified-jpeg")
        XCTAssertEqual(restored.writing?.fields, [target])
        XCTAssertEqual(restored.selectedTargetID, target.id)
        XCTAssertEqual(firstInode, try FileManager.default.attributesOfItem(atPath: photoFile.path)[.systemFileNumber] as? NSNumber)
        let values = try directory.resourceValues(forKeys: [.isExcludedFromBackupKey])
        XCTAssertEqual(values.isExcludedFromBackup, true)
    }

    func testFailedReplacementKeepsCommittedDocumentAndLateUpdatesCannotRestoreDeletedData() throws {
        XCTAssertNil(try store.load())
        try store.clear()
        var first = DocumentCheckpoint()
        first.questionDraft = "첫 번째 사진에 대한 질문"
        try store.replace(with: first, photo: photo)
        var second = DocumentCheckpoint()
        // Throw while encoding the new metadata, after the new photo was staged.
        second.lines = [PrintedLine(id: 1, text: "invalid", confidence: 1, contextSupport: .infinity)]
        XCTAssertThrowsError(try store.replace(with: second, photo: Data([0x99])))
        XCTAssertEqual(try store.load()?.checkpoint.id, first.id)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path).count, 3)

        second.lines = []
        try store.replace(with: second, photo: Data([0x98]))
        XCTAssertThrowsError(try store.save(first))
        XCTAssertNil(try store.load()?.checkpoint.questionToConfirm)
        XCTAssertEqual(try store.load()?.checkpoint.questionDraft, "")
        XCTAssertEqual(try store.load()?.photo, Data([0x98]))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path).count, 3)
        try store.clear()
        XCTAssertThrowsError(try store.save(second))
        XCTAssertNil(try store.load())
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
    }

    func testRefusesToAssociateCachedFieldsWithDifferentPhoto() throws {
        let checkpoint = DocumentCheckpoint()
        try store.replace(with: checkpoint, photo: photo)
        XCTAssertThrowsError(try store.replace(with: checkpoint, photo: Data([0x44])))
        XCTAssertEqual(try store.load()?.photo, photo)
        let url = directory.appendingPathComponent(checkpoint.id.uuidString + ".photo")
        try Data([0xff]).write(to: url)
        XCTAssertThrowsError(try store.load())
        try store.clear()
        XCTAssertNil(try store.load())
    }
}
