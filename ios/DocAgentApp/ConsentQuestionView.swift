import AudioToolbox
import SwiftUI
import UIKit
import DocAgentCore

@MainActor
struct ConsentQuestionView: View {
    @ObservedObject var model: HomeModel
    @ObservedObject private var wifi = CloudWiFiState.shared
    @ObservedObject private var relay = CloudDocumentAPI.shared
    @StateObject private var mic = QuestionMicrophone()
    @FocusState private var typing: Bool
    /// 말로 질문한 뒤 질문 확인과 답이 끝날 때마다 마이크를 다시 켜는 중인지.
    @State private var conversing = false
    /// 답을 기다리는 동안 "찾아보고 있어요"와 틱 소리로 진행 중임을 알리는 작업.
    @State private var waitingSignal: Task<Void, Never>?
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.dismiss) private var dismiss

    private var dialogue: [ConsentTurn] { model.turns.filter { $0.question != "문서 원문 읽기" } }
    private var canSend: Bool { model.localAnswer != nil || (wifi.available && relay.isConfigured) }
    private var microphoneLabel: String {
        switch mic.phase {
        case .idle: return "말로 질문하기"
        case .preparing: return "마이크 준비 취소"
        case .listening: return "말하기 마치고 질문 확인"
        case .finishing: return "아이폰에서 받아쓰는 중"
        }
    }

    var body: some View {
        NavigationStack {
            ScrollViewReader { scroll in
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        if let text = model.questionToConfirm {
                            confirmation(text).id("confirmation")
                        } else if let pending = model.pendingQuestion {
                            VStack(alignment: .leading, spacing: 12) {
                                Text(pending).font(.headline).accessibilityAddTraits(.isHeader)
                                ProgressView("답변을 준비하고 있어요.")
                                Button("질문 취소") { stopConversation(); mic.cancel(); model.cancelQuestion() }.frame(minHeight: 44)
                            }.padding(18).frame(maxWidth: .infinity, alignment: .leading).documentCard().id("pending")
                        } else {
                            ForEach(dialogue) { turn in answer(turn).id(turn.id) }
                            composer.id("compose")
                        }
                        if let message = model.questionError { Text(message).foregroundStyle(DocumentStyle.caution) }
                    }.padding(20).frame(maxWidth: 580).frame(maxWidth: .infinity)
                }
                .onChange(of: model.questionToConfirm) { _, value in
                    if value != nil { scroll.scrollTo("confirmation", anchor: .top) }
                }
                .onChange(of: model.pendingQuestion) { previous, value in
                    if value != nil { scroll.scrollTo("pending", anchor: .top) }
                    else if previous != nil, let turn = dialogue.last { scroll.scrollTo(turn.id, anchor: .top) }
                }
                .onChange(of: typing) { _, focused in
                    if focused { scroll.scrollTo("questionField", anchor: .center) }
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                // 대화 중에는 질문 확인 카드 아래에서도 마이크 상태와 녹음 취소를 보여준다.
                if (model.questionToConfirm == nil && model.pendingQuestion == nil && !typing) || mic.phase != .idle {
                    micDock
                        .padding(.horizontal, 20).padding(.vertical, 8)
                        .frame(maxWidth: 580).frame(maxWidth: .infinity)
                        .background(DocumentStyle.card.ignoresSafeArea(edges: .bottom))
                }
            }
            .background(DocumentStyle.canvas).foregroundStyle(DocumentStyle.ink)
            .toolbarColorScheme(.light, for: .navigationBar)
            .toolbarBackground(DocumentStyle.gold, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .navigationTitle("문서에 질문하기").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("닫기") { dismiss() }.foregroundStyle(DocumentStyle.darkInk) } }
        }
        .tint(DocumentStyle.accent)
        .onChange(of: mic.error) { _, message in
            if message != nil { stopConversation() }
            if scenePhase == .active, let message { model.speak(message, reading: true) }
        }
        .onChange(of: model.questionError) { _, message in if message != nil { stopConversation() } }
        .onChange(of: model.handBackCount) { _, _ in listenAgain() }
        .onChange(of: model.pendingQuestion) { _, pending in
            if pending == nil { stopWaitingSignal() } else { startWaitingSignal() }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background || (phase == .inactive && mic.phase == .listening) {
                stopConversation()
                if !mic.transcript.isEmpty { model.questionDraft = mic.transcript }
                mic.cancel()
            }
        }
        .onDisappear { stopWaitingSignal(); stopConversation(); mic.cancel() }
    }

    @ViewBuilder private var micDock: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                switch mic.phase {
                case .idle: startRecording()
                case .preparing: stopConversation(); mic.cancel()
                case .listening: mic.finish()
                case .finishing: break
                }
            } label: {
                Label(microphoneLabel, systemImage: mic.phase == .listening ? "stop.circle.fill" : "mic.fill")
            }
            .buttonStyle(DocumentButtonStyle()).disabled(mic.phase == .finishing)
            .accessibilityHint("말을 마치고 잠시 쉬면 질문을 읽어줍니다. 확인 후 전송합니다.")
            Toggle("말로 이어서 대화하기", isOn: $model.continuousConversation)
                .frame(minHeight: 44)
                .onChange(of: model.continuousConversation) { _, on in if !on { stopConversation() } }
                .accessibilityHint("켜면 질문 확인과 답이 끝난 뒤 띵 소리와 함께 마이크가 다시 켜집니다. 맞아 또는 아니로 확인하고, 그만이라고 말하면 끝납니다.")
            if conversing {
                Text("대화 중이에요. 띵 소리가 나면 바로 말하세요. 질문 확인에는 \"맞아\" 또는 \"아니\", 끝내려면 \"그만\".")
                    .font(.subheadline).fixedSize(horizontal: false, vertical: true)
            }
            if mic.phase == .listening || mic.phase == .finishing {
                if mic.phase == .finishing { ProgressView("아이폰이 질문을 받아쓰고 있어요.") }
                else { Text("듣고 있어요.").font(.callout) }
                Button(mic.phase == .finishing ? "받아쓰기 취소" : "녹음 취소") { stopConversation(); mic.cancel() }.frame(minHeight: 44)
            }
            if let error = mic.error { Text(error).foregroundStyle(DocumentStyle.caution) }
        }
    }

    @ViewBuilder private var composer: some View {
        Text(model.demoForm == nil
             ? "아이폰에서 받아쓴 질문을 확인한 뒤 보내세요. OpenAI에는 질문과 문서 원문, 관련 대화가 전달돼요."
             : "시연 양식의 준비된 질문은 저장된 답을 읽어요. 다른 질문은 확인 후 OpenAI에 보내요.")
            .font(.subheadline).foregroundStyle(DocumentStyle.muted)
        VStack(alignment: .leading, spacing: 10) {
            TextField("질문을 입력하세요", text: $model.questionDraft, axis: .vertical)
                .lineLimit(1...5).focused($typing).padding(12).frame(minHeight: 48)
                .background(DocumentStyle.canvas, in: RoundedRectangle(cornerRadius: 10))
                .accessibilityLabel("질문 입력").id("questionField")
            Button("입력한 질문 확인하기") { stopConversation(); stage(model.questionDraft) }
                .buttonStyle(DocumentButtonStyle(prominent: false))
                .disabled(model.questionDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }.padding(16).documentCard().disabled(mic.phase != .idle)

        VStack(alignment: .leading, spacing: 2) {
            Text("예시 질문").font(.headline).padding(.bottom, 6).accessibilityAddTraits(.isHeader)
            ScrollView(.horizontal) {
                HStack(spacing: 8) {
                    ForEach(["이거 어떤 서류야?", "한 줄로 요약해줘", "내 정보는 누구에게 제공돼?", "동의를 거부하면 어떻게 돼?", "대출 개인정보가 잘못 쓰인 실제 사례가 있어?"], id: \.self) { text in
                        Button { stopConversation(); stage(text) } label: {
                            Text(text).padding(.horizontal, 12).frame(minHeight: 44)
                                .background(DocumentStyle.raised, in: Capsule())
                        }.buttonStyle(.plain)
                    }
                }
            }.scrollIndicators(.automatic)
        }.padding(16).documentCard().disabled(mic.phase != .idle)
    }

    private func confirmation(_ text: String) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("이 질문이 맞나요?", systemImage: "text.bubble.fill")
                .font(.title2.bold()).accessibilityAddTraits(.isHeader)
            Text(text).font(.title3.weight(.medium)).fixedSize(horizontal: false, vertical: true)
                .padding(14).frame(maxWidth: .infinity, alignment: .leading)
                .background(DocumentStyle.canvas, in: RoundedRectangle(cornerRadius: 12))
            if !canSend {
                Label(wifi.available ? "답변을 받으려면 홈의 설정에서 맥을 연결해 주세요." : "답변을 받으려면 맥과 같은 Wi-Fi에 연결해 주세요.",
                      systemImage: wifi.available ? "network" : "wifi.slash")
                    .font(.subheadline).foregroundStyle(DocumentStyle.caution)
            }
            // 대화 중에는 마이크가 자동으로 켜져 있을 수 있다. 버튼을 누르면 먼저 녹음을 멈춰
            // 버튼을 누른 뒤 들어온 말이 새 질문이나 대답으로 처리되지 않게 한다.
            Button { mic.cancel(); model.confirmQuestion() } label: { Label(model.localAnswer == nil ? "맞아요, 질문 보내기" : "맞아요, 저장된 답 듣기", systemImage: "checkmark") }
                .buttonStyle(DocumentButtonStyle()).disabled(!canSend)
            Button { mic.cancel(); model.repeatQuestion() } label: { Label("질문 다시 듣기", systemImage: "speaker.wave.2.fill") }
                .buttonStyle(DocumentButtonStyle(prominent: false))
            Button { model.discardQuestion(); startRecording() } label: { Label("다시 말하기", systemImage: "mic.fill") }
                .buttonStyle(DocumentButtonStyle(prominent: false))
            Button {
                stopConversation()
                mic.cancel()
                model.discardQuestion()
                model.questionDraft = text
                Task { @MainActor in
                    await Task.yield()
                    typing = true
                }
            } label: { Label("글자로 수정하기", systemImage: "pencil") }
            .buttonStyle(DocumentButtonStyle(prominent: false))
            Button("질문 취소") { stopConversation(); mic.cancel(); model.discardQuestion() }.frame(maxWidth: .infinity, minHeight: 44)
        }.padding(18).documentCard()
    }

    private func answer(_ turn: ConsentTurn) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(turn.question).font(.headline).foregroundStyle(.white)
                .padding(13).background(DocumentStyle.action, in: RoundedRectangle(cornerRadius: 16))
                .frame(maxWidth: .infinity, alignment: .trailing)
                .accessibilityLabel("내 질문. " + turn.question).accessibilityAddTraits(.isHeader)
            VStack(alignment: .leading, spacing: 9) {
                Text(turn.reply.spoken).fixedSize(horizontal: false, vertical: true)
                ForEach(turn.reply.choices, id: \.self) { choice in
                    Button(choice) { stage(choice) }.frame(minHeight: 44)
                }
                if !turn.reply.source.isEmpty {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(turn.reply.source).font(.subheadline).foregroundStyle(DocumentStyle.ink)
                        Button { model.speak(turn.reply.source, reading: true) } label: { Label("출처 듣기", systemImage: "speaker.wave.2") }
                            .frame(minHeight: 44)
                        if let url = turn.reply.url {
                            Link("공식 원문 열기", destination: url).frame(minHeight: 44)
                                .accessibilityHint("공식 기관 웹페이지를 엽니다.")
                        }
                    }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
                    .background(DocumentStyle.raised, in: RoundedRectangle(cornerRadius: 12))
                }
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 14) { listeningActions(turn) }
                    VStack(alignment: .leading, spacing: 2) { listeningActions(turn) }
                }
            }.padding(15).frame(maxWidth: .infinity, alignment: .leading).documentCard()
        }.disabled(mic.phase != .idle)
    }

    @ViewBuilder private func listeningActions(_ turn: ConsentTurn) -> some View {
        Button { model.speak(turn.reply.spoken, reading: true) } label: {
            Label("답 다시 듣기", systemImage: "speaker.wave.2")
        }.frame(minHeight: 44)
        Button("읽기 멈추기") { stopConversation(); model.stopSpeech() }.frame(minHeight: 44)
    }

    /// 마이크를 켜고 한 번 듣는다. 대화 중이면 말이 없을 때 오류 대신 대화를 끝낸다.
    private func startRecording() {
        guard scenePhase == .active else { return }
        typing = false
        model.stopSpeech()
        let silence: (() -> Void)? = conversing
            ? { endConversation("대화를 마칠게요. 다시 질문하려면 말로 질문하기를 눌러 주세요.") }
            : nil
        mic.begin(onSilence: silence, onQuestion: heard)
    }

    /// 말로 들은 내용을 처리한다. 질문 확인 중이면 "맞아/아니"로 보고, 아니면 새 질문으로 확인을 받는다.
    private func heard(_ text: String) {
        // 답을 기다리는 중에 늦게 받아쓴 말은 질문이나 대답으로 쓰지 않는다.
        guard model.pendingQuestion == nil else { return }
        if Self.isStopCommand(text) {
            endConversation("대화를 마칠게요.")
            return
        }
        conversing = model.continuousConversation
        if model.questionToConfirm != nil {
            switch Self.confirmationReply(text) {
            case .yes: model.confirmQuestion()
            case .no: model.askToRephrase()
            case .question(let corrected): stage(corrected)
            }
        } else {
            stage(text)
        }
    }

    /// 질문 확인이나 답을 끝까지 읽었을 때 불린다. 띵 소리와 진동으로 알린 뒤 마이크를 다시 켠다.
    private func listenAgain() {
        guard conversing, model.continuousConversation, mic.phase == .idle, model.pendingQuestion == nil else { return }
        AudioServicesPlaySystemSound(1113)
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        Task {
            // 띵 소리가 마이크에 질문으로 들어가지 않도록 잠시 기다린다.
            try? await Task.sleep(for: .milliseconds(450))
            guard conversing, mic.phase == .idle, model.pendingQuestion == nil, !model.guidanceSpeaking else { return }
            startRecording()
        }
    }

    private func stopConversation() { conversing = false }

    private func endConversation(_ message: String) {
        stopConversation()
        model.speak(message, reading: true)
    }

    /// 답이 0.6초 안에 나오면 아무것도 하지 않는다. 늦어지면 한 번 말하고, 2초부터 1.5초마다 틱 소리와 진동을 준다.
    private func startWaitingSignal() {
        stopWaitingSignal()
        waitingSignal = Task {
            do {
                try await Task.sleep(for: .milliseconds(600))
                guard model.pendingQuestion != nil else { return }
                model.speak("찾아보고 있어요.")
                try await Task.sleep(for: .milliseconds(1400))
                let haptic = UIImpactFeedbackGenerator(style: .soft)
                while model.pendingQuestion != nil {
                    if !model.synthesizer.isSpeaking { AudioServicesPlaySystemSound(1104) }
                    haptic.impactOccurred(intensity: 0.6)
                    try await Task.sleep(for: .milliseconds(1500))
                }
            } catch {}
        }
    }

    private func stopWaitingSignal() {
        waitingSignal?.cancel()
        waitingSignal = nil
    }

    enum ConfirmationReply: Equatable { case yes, no, question(String) }

    /// "응", "맞아", "보내줘"는 확인, "아니", "다시"는 다시 말하기로 본다.
    /// "아니 보유 기간 알려줘"처럼 고친 질문을 바로 말하면 그 질문으로 다시 확인을 받는다.
    /// 확인은 질문과 문서를 밖으로 보내는 일이라, 애매하면 확인으로 보지 않는다.
    static func confirmationReply(_ text: String) -> ConfirmationReply {
        let words = text.filter { $0.isLetter }
        // "네, 보내지 마", "응 취소", "잠깐만"처럼 보내지 말라는 짧은 대답은 "네"로 시작해도 확인이 아니다.
        // 긴 문장의 "취소"("동의 취소는 어떻게 해?")는 질문일 수 있어 8글자 이하만 본다.
        let refusals = ["지마", "지말", "말아", "말고", "안돼", "안보내", "취소", "잠깐", "멈춰", "기다려"]
        if words.count <= 8, refusals.contains(where: { words.contains($0) }) { return .no }
        let negative = ["아니", "아냐", "틀려", "틀렸", "다시"]
        if let lead = negative.first(where: { words.hasPrefix($0) }) {
            // 앞의 "아니,"만 떼고 고친 질문의 숫자("3번", "2024년")와 끝의 물음표는 남긴다.
            let rest = String(Self.dropLetters(lead.count, from: text).drop { $0.isWhitespace || $0.isPunctuation })
            let restLetters = rest.filter { $0.isLetter }
            let filler = ["요", "야", "에요", "예요", "어", "었어", "어요", "말할게", "말할게요", "말할래", "말해줘", "할게", "할게요", "그거아니야", "그게아니야"]
            return restLetters.count <= 2 || filler.contains(restLetters) ? .no : .question(rest)
        }
        // "어떤 정보야?"가 "어"로 시작하듯 질문 첫 글자와 겹치므로 짧은 대답은 통째로 비교한다.
        let positive: Set<String> = ["응", "어", "네", "예", "네네", "응응", "ㅇㅇ", "오케이", "그래", "그래요", "좋아", "좋아요",
                                     "보내", "보내줘", "보내줘요", "보내주세요", "질문해", "질문해줘", "질문해주세요",
                                     "그렇게해", "그렇게해줘", "그렇게해주세요", "맞습니다", "맞네", "맞네요"]
        let agrees = ["맞아", "응맞", "어맞", "네맞", "예맞", "응보내", "네보내"]
        if positive.contains(words) || (words.count <= 6 && agrees.contains(where: { words.hasPrefix($0) })) { return .yes }
        return .question(text)
    }

    /// 앞에서부터 글자 `count`개(공백·문장부호 포함)까지 잘라낸 나머지. "아 니, 3번"처럼 띄어 말해도 맞게 자른다.
    private static func dropLetters(_ count: Int, from text: String) -> Substring {
        var seen = 0
        var index = text.startIndex
        while index < text.endIndex, seen < count {
            if text[index].isLetter { seen += 1 }
            index = text.index(after: index)
        }
        return text[index...]
    }

    /// "그만", "끝", "종료", "고마워"처럼 대화를 끝내는 말인지.
    /// "종료일 알려줘", "그만두면 어떻게 돼", "끝까지 읽어줘"처럼 같은 말로 시작하는 질문과 헷갈리지 않게
    /// 끝내는 말만 통째로 비교한다.
    static func isStopCommand(_ text: String) -> Bool {
        let words = text.filter { $0.isLetter }
        let stops: Set<String> = [
            "끝", "끝내", "끝내자", "끝낼게", "끝낼게요", "끝내줘", "끝이야", "대화끝",
            "그만", "그만해", "그만해요", "그만하자", "그만할게", "그만할게요", "그만할래", "그만해줘", "이제그만", "이제그만할게",
            "종료", "종료해", "종료해줘", "종료할게", "대화종료",
            "됐어", "됐어요", "이제됐어", "이제됐어요",
            "고마워", "고마워요", "고맙습니다", "감사합니다", "됐어고마워", "알겠어고마워", "고마워그만"
        ]
        return stops.contains(words)
    }

    private func stage(_ text: String) {
        typing = false
        model.stageQuestion(text)
        if model.questionToConfirm != nil { model.questionDraft = "" }
    }
}
