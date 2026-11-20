import AVFoundation
import DocAgentCore
import PhotosUI
import SwiftUI
import UIKit

@MainActor
final class HomeModel: ObservableObject {
    enum Phase { case idle, recognizing, reviewing, finished }
    @Published private(set) var phase = Phase.idle
    @Published var cameraPresented = false
    @Published var questionsPresented = false
    @Published private(set) var turns: [ConsentTurn] = []
    @Published private(set) var pendingQuestion: String?
    private var answerTask: Task<Void, Never>?
    private var questionRevision = 0
    var busy: Bool { phase == .recognizing || phase == .reviewing }
    @Published private(set) var lines: [PrintedLine] = []
    @Published private(set) var notice = "문서를 준비한 뒤 카메라를 열어 주세요."
    @Published var automaticSpeech = true
    @Published var readingRate = 1.0
    let synthesizer = AVSpeechSynthesizer()
    private let reviewer = DocumentReviewer()
    private var recognition: Task<Void, Never>?
    private var revision = 0
    private var announcement: String?
    private var announcementDeadline = Date.distantPast
    private var readingDocument = false
    var guidanceSpeaking: Bool { synthesizer.isSpeaking || Date() < announcementDeadline }

    func speak(_ text: String, reading: Bool = false) {
        guard reading || automaticSpeech else { return }
        stopSpeech()
        readingDocument = reading
        if UIAccessibility.isVoiceOverRunning && !reading {
            announcement = text
            announcementDeadline = Date().addingTimeInterval(min(18, Double(text.count) / 5 + 1))
            UIAccessibility.post(notification: .announcement, argument: text)
        } else {
            let utterance = AVSpeechUtterance(string: text)
            utterance.voice = AVSpeechSynthesisVoice(language: "ko-KR")
            utterance.rate = AVSpeechUtteranceDefaultSpeechRate * Float(readingRate)
            synthesizer.speak(utterance)
        }
    }
    func announcementFinished(_ notification: Notification) {
        if let text = notification.userInfo?[UIAccessibility.announcementStringValueUserInfoKey] as? String,
           text == announcement {
            announcement = nil
            announcementDeadline = .distantPast
        }
    }
    func stopSpeech() {
        synthesizer.stopSpeaking(at: .immediate)
        readingDocument = false
        announcement = nil
        announcementDeadline = .distantPast
    }

    func beginCamera() {
        stopSpeech()
        cameraPresented = true
    }

    func readLines(_ selected: [PrintedLine]) {
        guard !selected.isEmpty else { return }
        let reply = ConsentReply.page(selected)
        remember(ConsentTurn(question: "문서 원문 읽기", reply: reply))
        speak(reply.text, reading: true)
    }

    func openQuestions() {
        guard !lines.isEmpty else { return }
        stopSpeech()
        questionsPresented = true
    }

    func ask(_ raw: String) {
        let question = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard questionsPresented, pendingQuestion == nil, !lines.isEmpty, !question.isEmpty else { return }
        stopSpeech()
        if phase == .reviewing {
            // Give the question the shared model immediately; preserve the existing OCR confidence.
            revision += 1
            recognition?.cancel()
            recognition = nil
            phase = .finished
            notice = "사진 대조를 멈추고 지금 읽은 원문으로 질문에 답해요."
        }
        pendingQuestion = question
        questionRevision += 1
        let ticket = questionRevision
        let original = lines, previous = turns
        answerTask = Task {
            do {
                let answer = try await reviewer.answer(question, lines: original, history: previous)
                guard ticket == questionRevision, questionsPresented else { return }
                remember(ConsentTurn(question: question, reply: answer))
                speak(answer.spoken, reading: true)
            } catch {
                guard ticket == questionRevision, !Task.isCancelled else { return }
                speak("답변을 마치지 못했어요. 다시 질문해 주세요.", reading: true)
            }
            if ticket == questionRevision {
                pendingQuestion = nil
                answerTask = nil
            }
        }
    }

    private func remember(_ turn: ConsentTurn) {
        turns.append(turn)
        if turns.count > 24 { turns.removeFirst(turns.count - 24) }
    }

    func cancelQuestion() {
        questionRevision += 1
        answerTask?.cancel()
        answerTask = nil
        pendingQuestion = nil
        stopSpeech()
    }

    func capture(_ data: Data) {
        cameraPresented = false
        recognize { data }
    }
    func importPhoto(_ item: PhotosPickerItem) {
        recognize {
            guard let data = try await item.loadTransferable(type: Data.self) else { throw CocoaError(.fileReadCorruptFile) }
            return data
        }
    }
    private func recognize(load: @escaping () async throws -> Data) {
        clear()
        let generation = revision
        phase = .recognizing
        notice = "사진의 글자를 읽고 있어요."
        speak(notice)
        recognition = Task {
            do {
                let data = try await load()
                try Task.checkCancellation()
                let output = try await reviewer.read(data) { [weak self] originals in
                    await self?.receivedOCR(originals, generation: generation)
                }
                guard revision == generation else { return }
                lines = output.lines
                notice = output.notice
            } catch {
                guard revision == generation else { return }
                notice = "사진을 읽지 못했어요. 다시 촬영하거나 다른 사진을 선택해 주세요."
            }
            phase = .finished
            recognition = nil
            if !questionsPresented && !(readingDocument && synthesizer.isSpeaking) { speak(notice) }
        }
    }
    private func receivedOCR(_ originals: [PrintedLine], generation: Int) {
        guard generation == revision else { return }
        phase = .reviewing
        lines = originals
        notice = originals.isEmpty ? "사진에서 종이와 글자 흔적을 확인하고 있어요." :
            "사진과 인식한 글자를 대조하고 있어요. 지금도 원문을 들을 수 있어요."
        speak(notice)
    }

    func cancelReview() {
        let originals = lines
        clear()
        lines = originals
        if !originals.isEmpty { phase = .finished }
        notice = originals.isEmpty ? "확인을 취소했어요. 다른 사진을 촬영할 수 있어요." :
            "재확인을 멈췄어요. 먼저 인식한 원문을 들을 수 있어요."
        speak(notice)
    }
    func clear() {
        questionsPresented = false
        cancelQuestion()
        turns = []
        revision += 1
        recognition?.cancel()
        recognition = nil
        stopSpeech()
        lines = []
        phase = .idle
        notice = "문서를 준비한 뒤 카메라를 열어 주세요."
    }
}
