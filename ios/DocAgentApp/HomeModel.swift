import AVFoundation
import DocAgentCore
import ImageIO
import PhotosUI
import SwiftUI
import UIKit

@MainActor
final class HomeModel: ObservableObject {
    enum Phase { case idle, recognizing, finished }
    @Published private(set) var phase = Phase.idle
    @Published var cameraPresented = false
    @Published private var screen = DocumentCheckpoint.Screen.result
    var writingPresented: Bool {
        get { screen == .writing }
        set { present(.writing, isPresented: newValue) }
    }
    var readingPresented: Bool {
        get { screen == .reading }
        set { present(.reading, isPresented: newValue) }
    }
    var questionsPresented: Bool {
        get { screen == .questions }
        set { present(.questions, isPresented: newValue) }
    }
    @Published var readingIndex = 0 {
        didSet { if readingIndex != oldValue { saveProgress() } }
    }
    @Published var questionDraft = "" {
        didSet { if questionDraft != oldValue { saveProgress() } }
    }
    @Published var writingTargetID: String? {
        didSet { if writingTargetID != oldValue { saveProgress() } }
    }
    private(set) var writingPhoto: Data?
    private(set) var demoForm: DemoConsentForm.Match?
    private(set) var writingPage: WritingPage?
    var writingFields: [WritingTarget]? { checkpoint?.writing?.fields }
    var writingReferenceDigest: String? { checkpoint?.writing?.referenceDigest }
    var documentID: UUID? { checkpoint?.id }
    @Published private(set) var turns: [ConsentTurn] = []
    @Published private(set) var questionToConfirm: String?
    @Published private(set) var localAnswer: ConsentReply?
    private var answerCache = SessionAnswerCache()
    @Published private(set) var pendingQuestion: String?
    @Published private(set) var questionError: String?
    private var answerTask: Task<Void, Never>?
    private var questionRevision = 0
    var busy: Bool { phase == .recognizing || importing }
    @Published private(set) var lines: [PrintedLine] = []
    @Published private(set) var notice = "문서를 준비한 뒤 카메라를 열어 주세요."
    @Published var automaticSpeech = true
    @Published var readingRate = 1.0
    /// 질문 확인과 답을 끝까지 들은 뒤 마이크를 다시 켜서 말로 이어서 대화한다.
    @Published var continuousConversation = true
    /// 사용자에게 차례를 넘기는 말(질문 확인, 답)을 중간에 멈추지 않고 끝까지 읽을 때마다 1씩 늘어난다.
    @Published private(set) var handBackCount = 0
    private var handBack: UUID?
    private var handBackWatch: Task<Void, Never>?
    let synthesizer = AVSpeechSynthesizer()
    private let reviewer = DocumentReviewer()
    private var recognition: Task<Void, Never>?
    private var contextReview: Task<Void, Never>?
    @Published private(set) var ocrReviewStatus: String?
    @Published private(set) var canRetryOCRReview = false
    private var revision = 0
    private var announcement: String?
    private var announcementDeadline = Date.distantPast
    private var readingDocument = false
    var guidanceSpeaking: Bool { synthesizer.isSpeaking || Date() < announcementDeadline }
    private let store: DocumentSessionStore
    private var checkpoint: DocumentCheckpoint?
    private var savedPhotoID: UUID?
    private var needsRestore = true
    private var applyingState = false
    private var suspended = false
    @Published private(set) var saveError: String?
    @Published private var importing = false
    private var importTask: Task<Void, Never>?

    init(store: DocumentSessionStore = DocumentSessionStore()) {
        self.store = store
        restoreIfNeeded()
    }

    private func present(_ destination: DocumentCheckpoint.Screen, isPresented: Bool) {
        guard (screen == destination) != isPresented else { return }
        screen = isPresented ? destination : .result
        saveProgress()
    }

    private func restoreIfNeeded() {
        guard needsRestore else { return }
        do {
            let saved = try store.load()
            needsRestore = false
            saveError = nil
            guard let saved else { return }
            applyingState = true
            defer { applyingState = false }
            let work = saved.checkpoint
            checkpoint = work
            savedPhotoID = work.id
            writingPhoto = saved.photo
            demoForm = work.demoForm
            lines = work.lines
            turns = work.turns
            notice = work.notice
            phase = work.recognitionPending ? .recognizing : .finished
            readingIndex = min(max(0, work.readingIndex), max(0, lines.count - 1))
            questionDraft = work.questionDraft
            questionToConfirm = work.questionToConfirm
            localAnswer = work.questionToConfirm.flatMap(preparedAnswer)
            writingTargetID = work.selectedTargetID
            canRetryOCRReview = work.canRetryOCRReview
            ocrReviewStatus = work.ocrReviewStatus
            screen = lines.isEmpty ? .result : work.screen
        } catch {
            saveError = UIApplication.shared.isProtectedDataAvailable
                ? "저장한 문서를 열지 못했어요. 다시 열어 보거나 저장된 작업을 지울 수 있어요."
                : "아이폰 잠금을 풀면 저장한 문서를 다시 열어요."
        }
    }

    private func saveProgress() {
        guard !applyingState, var work = checkpoint, let photo = writingPhoto else { return }
        work.lines = lines
        work.turns = turns
        work.demoForm = demoForm
        work.notice = notice
        work.readingIndex = readingIndex
        work.questionDraft = questionDraft
        work.questionToConfirm = questionToConfirm ?? pendingQuestion
        work.screen = screen
        work.selectedTargetID = writingTargetID
        // An interrupted online review may be retried explicitly when the user returns.
        work.canRetryOCRReview = canRetryOCRReview || contextReview != nil
        work.ocrReviewStatus = contextReview == nil ? ocrReviewStatus : "문맥 검증을 다시 시도할 수 있어요."
        checkpoint = work
        do {
            if savedPhotoID == work.id {
                try store.save(work)
            } else {
                try store.replace(with: work, photo: photo)
                savedPhotoID = work.id
            }
            saveError = nil
        } catch {
            saveError = "진행 상황을 아이폰에 저장하지 못했어요. 잠금 상태와 저장 공간을 확인해 주세요."
        }
    }

    func storeWritingPage(_ page: WritingPage) {
        writingPage = page
        checkpoint?.writing = WritingCheckpoint(referenceDigest: page.reference.digest, fields: page.fields)
        saveProgress()
    }

    func suspend() {
        suspended = true
        cameraPresented = false
        stopSpeech()
        recognition?.cancel()
        recognition = nil
        if let pendingQuestion {
            questionRevision += 1
            answerTask?.cancel()
            answerTask = nil
            self.pendingQuestion = nil
            questionToConfirm = pendingQuestion
        }
        if contextReview != nil {
            contextReview?.cancel()
            contextReview = nil
            canRetryOCRReview = lines.contains(where: \.canReviewContext)
            ocrReviewStatus = canRetryOCRReview ? "문맥 검증을 다시 시도할 수 있어요." : nil
            notice = DocumentReviewer.countNotice(lines)
        }
        saveProgress()
    }

    func activate() {
        suspended = false
        restoreIfNeeded()
        if saveError != nil { saveProgress() }
        if checkpoint?.recognitionPending == true, recognition == nil, let photo = writingPhoto {
            readPhoto(photo, automaticReview: false)
        }
    }

    /// - Parameter handsBack: 끝까지 읽으면 `handBackCount`를 올려 질문 화면이 마이크를 다시 켤 수 있게 한다.
    func speak(_ text: String, reading: Bool = false, force: Bool = false, handsBack: Bool = false) {
        guard !suspended, reading || force || automaticSpeech else { return }
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
        if handsBack { watchHandBack() }
    }

    /// 원문 읽기 화면이 synthesizer.delegate를 쓰므로 delegate 대신 읽기 상태를 짧게 확인한다.
    /// stopSpeech()로 멈추면 handBack이 지워져 차례를 넘기지 않는다.
    private func watchHandBack() {
        let id = UUID()
        handBack = id
        handBackWatch?.cancel()
        handBackWatch = Task { [weak self] in
            do {
                try await Task.sleep(for: .milliseconds(300))
                while let model = self, model.handBack == id, model.guidanceSpeaking {
                    try await Task.sleep(for: .milliseconds(150))
                }
            } catch { return }
            guard let self, self.handBack == id else { return }
            self.handBack = nil
            self.handBackCount += 1
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
        handBack = nil
        handBackWatch?.cancel()
        handBackWatch = nil
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
        if selected.count == 1, let index = lines.firstIndex(where: { $0.id == selected[0].id }) {
            readingIndex = index
        }
        let reply = ConsentReply.page(selected)
        remember(ConsentTurn(question: "문서 원문 읽기", reply: reply))
        saveProgress()
        speak(reply.text, reading: true)
    }

    func openQuestions() {
        guard !lines.isEmpty else { return }
        stopSpeech()
        questionsPresented = true
    }

    func stageQuestion(_ raw: String) {
        let question = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard questionsPresented, pendingQuestion == nil, !lines.isEmpty, !question.isEmpty else { return }
        guard question.unicodeScalars.count <= 2000 else {
            questionError = CloudQuestionError.questionLength.localizedDescription
            speak(CloudQuestionError.questionLength.localizedDescription, force: true)
            return
        }
        questionError = nil
        localAnswer = preparedAnswer(question)
        questionToConfirm = question
        saveProgress()
        repeatQuestion()
    }

    private func preparedAnswer(_ question: String) -> ConsentReply? {
        guard let demoForm else { return nil }
        if let answer = DemoDocumentAnswers.prepared(question, match: demoForm, page: lines) { return answer }
        guard let request = try? CloudQuestionRequest(question: question, lines: lines, history: turns),
              let key = try? SessionAnswerCache.key(for: request) else { return nil }
        return answerCache.lookup(key)
    }

    func repeatQuestion() {
        guard let question = questionToConfirm else { return }
        speak("“" + question + "” 이렇게 질문할까요?", force: true, handsBack: true)
    }

    /// 확인 질문에 "아니"라고 답했을 때 질문을 지우고 다시 말하도록 차례를 넘긴다.
    func askToRephrase() {
        discardQuestion()
        speak("다시 말씀해 주세요.", force: true, handsBack: true)
    }

    func discardQuestion() {
        questionToConfirm = nil
        localAnswer = nil
        questionError = nil
        stopSpeech()
        saveProgress()
    }

    func confirmQuestion() {
        guard questionsPresented, let question = questionToConfirm, pendingQuestion == nil, !lines.isEmpty else { return }
        if let localAnswer {
            remember(ConsentTurn(question: question, reply: localAnswer))
            discardQuestion()
            speak(localAnswer.spoken, reading: true, handsBack: true)
            return
        }
        guard CloudWiFiState.shared.available, CloudDocumentAPI.shared.isConfigured else {
            let message = "질문은 보관하고 있어요. 같은 Wi-Fi의 맥을 연결한 다음 다시 확인해 주세요."
            questionError = message
            speak(message, force: true)
            return
        }
        questionToConfirm = nil
        stopSpeech()
        questionError = nil
        pendingQuestion = question
        saveProgress()
        questionRevision += 1
        let ticket = questionRevision
        let original = lines, previous = turns
        answerTask = Task {
            do {
                let request = try CloudQuestionRequest(question: question, lines: original, history: previous)
                let response = try await CloudDocumentAPI.shared.answer(request)
                guard ticket == questionRevision, questionsPresented else { return }
                let result = try response.resolve(for: request, page: lines)
                if demoForm != nil, let key = try? SessionAnswerCache.key(for: request) {
                    answerCache.record(result.reply, for: key)
                }
                lines = result.lines
                if canRetryOCRReview && !lines.contains(where: \.canReviewContext) {
                    canRetryOCRReview = false
                    ocrReviewStatus = nil
                }
                notice = DocumentReviewer.countNotice(lines)
                remember(ConsentTurn(question: question, reply: result.reply))
                if demoForm != nil, SessionAnswerCache.permitsImmediateRepeat(question),
                   let repeatRequest = try? CloudQuestionRequest(question: question, lines: lines, history: turns),
                   let key = try? SessionAnswerCache.key(for: repeatRequest) {
                    answerCache.record(result.reply, for: key)
                }
                speak(result.reply.spoken, reading: true, handsBack: true)
            } catch {
                guard ticket == questionRevision, !Task.isCancelled else { return }
                questionToConfirm = question
                let message = (error as? LocalizedError)?.errorDescription ?? "답변을 받지 못했어요. Wi-Fi 연결을 확인하고 다시 질문해 주세요."
                questionError = message
                speak(message, reading: true)
            }
            if ticket == questionRevision {
                pendingQuestion = nil
                answerTask = nil
                saveProgress()
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
        questionToConfirm = nil
        localAnswer = nil
        pendingQuestion = nil
        questionError = nil
        stopSpeech()
        saveProgress()
    }

    func capture(_ data: Data) {
        cameraPresented = false
        acceptPhoto(data)
    }
    func importPhoto(_ item: PhotosPickerItem) {
        importTask?.cancel()
        importing = true
        if checkpoint == nil { phase = .recognizing }
        importTask = Task {
            do {
                guard let data = try await item.loadTransferable(type: Data.self) else { throw CocoaError(.fileReadCorruptFile) }
                guard !Task.isCancelled else { return }
                acceptPhoto(data)
            } catch {
                guard !Task.isCancelled else { return }
                importing = false
                importTask = nil
                reportInvalidPhoto()
            }
        }
    }

    private func reportInvalidPhoto() {
        if checkpoint == nil { phase = .finished }
        notice = "사진을 읽지 못했어요. 다시 촬영하거나 다른 사진을 선택해 주세요."
        speak(notice)
    }

    private func acceptPhoto(_ data: Data) {
        importTask?.cancel()
        importTask = nil
        importing = false
        guard let source = CGImageSourceCreateWithData(data as CFData, nil), CGImageSourceGetCount(source) > 0 else {
            reportInvalidPhoto()
            return
        }
        resetInMemory()
        needsRestore = false
        var work = DocumentCheckpoint()
        work.recognitionPending = true
        checkpoint = work
        writingPhoto = data
        phase = .recognizing
        notice = "사진의 글자를 읽고 있어요."
        saveProgress()
        guard !suspended else { return }
        speak(notice)
        readPhoto(data, automaticReview: true)
    }

    private func readPhoto(_ data: Data, automaticReview: Bool) {
        let generation = revision
        recognition = Task {
            do {
                try Task.checkCancellation()
                let exact = await Task.detached(priority: .userInitiated) { DemoConsentForm.exactPhoto(data) }.value
                let page: [PrintedLine]
                let message: String
                let match: DemoConsentForm.Match?
                if let exact {
                    match = exact
                    page = exact.lines
                    message = DocumentReviewer.countNotice(page)
                } else {
                    let output = try await reviewer.read(data)
                    match = await Task.detached(priority: .userInitiated) {
                        DemoConsentForm.match(output.lines, imageData: data)
                    }.value
                    page = match?.lines ?? output.lines
                    message = match == nil ? output.notice : DocumentReviewer.countNotice(page)
                }
                guard revision == generation, !Task.isCancelled else { return }
                demoForm = match
                lines = page
                notice = message
                if automaticReview {
                    retryOCRReview()
                } else {
                    canRetryOCRReview = lines.count > 1 && lines.contains(where: \.canReviewContext)
                    ocrReviewStatus = canRetryOCRReview ? "문맥 검증을 다시 시도할 수 있어요." : nil
                }
            } catch {
                guard revision == generation, !Task.isCancelled else { return }
                notice = "사진을 읽지 못했어요. 다시 촬영하거나 다른 사진을 선택해 주세요."
            }
            checkpoint?.recognitionPending = false
            phase = .finished
            recognition = nil
            saveProgress()
            if !cameraPresented && !questionsPresented && !(readingDocument && synthesizer.isSpeaking) { speak(notice) }
        }
    }

    func retryOCRReview() {
        guard contextReview == nil else { return }
        guard lines.count > 1, lines.contains(where: \.canReviewContext) else {
            canRetryOCRReview = false
            ocrReviewStatus = nil
            return
        }
        guard CloudWiFiState.shared.available && CloudDocumentAPI.shared.isConfigured else {
            ocrReviewStatus = "같은 Wi-Fi의 맥을 연결하면 글자를 문맥과 대조할 수 있어요."
            canRetryOCRReview = true
            return
        }
        let generation = revision
        let page = lines
        canRetryOCRReview = false
        ocrReviewStatus = "OpenAI가 불확실한 줄을 주변 문맥과 대조하고 있어요."
        notice = "\(lines.count)줄을 읽었어요. 문맥을 확인하는 동안에도 원문을 들을 수 있어요."
        contextReview = Task {
            do {
                let request = try CloudOCRReviewRequest(page: page)
                let response = try await CloudDocumentAPI.shared.review(request)
                guard generation == revision, !Task.isCancelled else { return }
                lines = try response.applying(to: lines, request: request)
                ocrReviewStatus = "문맥 대조를 마쳤어요."
            } catch {
                guard generation == revision, !Task.isCancelled else { return }
                canRetryOCRReview = lines.contains(where: \.canReviewContext)
                ocrReviewStatus = canRetryOCRReview ? "문맥 대조를 마치지 못했어요. " + error.localizedDescription : nil
            }
            notice = DocumentReviewer.countNotice(lines)
            contextReview = nil
            saveProgress()
        }
    }

    func cancelReview() {
        if importing && checkpoint != nil {
            importTask?.cancel()
            importTask = nil
            importing = false
            notice = "새 사진 불러오기를 취소했어요."
            speak(notice)
            return
        }
        guard clear() else { return }
        notice = "읽기를 취소했어요. 다른 사진을 촬영할 수 있어요."
        speak(notice)
    }
    @discardableResult func clear() -> Bool {
        do {
            try store.clear()
        } catch {
            saveError = "저장한 문서를 지우지 못했어요. 아이폰 잠금을 풀고 다시 시도해 주세요."
            return false
        }
        needsRestore = false
        resetInMemory()
        return true
    }

    private func resetInMemory() {
        applyingState = true
        defer { applyingState = false }
        importTask?.cancel()
        importTask = nil
        importing = false
        checkpoint = nil
        savedPhotoID = nil
        saveError = nil
        screen = .result
        readingIndex = 0
        questionDraft = ""
        writingTargetID = nil
        writingPage = nil
        writingPhoto = nil
        demoForm = nil
        answerCache = SessionAnswerCache()
        cancelQuestion()
        turns = []
        revision += 1
        recognition?.cancel()
        recognition = nil
        contextReview?.cancel()
        contextReview = nil
        ocrReviewStatus = nil
        canRetryOCRReview = false
        stopSpeech()
        lines = []
        phase = .idle
        notice = "문서를 준비한 뒤 카메라를 열어 주세요."
    }
}
