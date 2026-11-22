import SwiftUI
import DocAgentCore

@MainActor
struct ConsentQuestionView: View {
    @ObservedObject var model: HomeModel
    @ObservedObject private var wifi = CloudWiFiState.shared
    @ObservedObject private var relay = CloudDocumentAPI.shared
    @StateObject private var mic = QuestionMicrophone()
    @FocusState private var typing: Bool
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
                                Button("질문 취소") { model.cancelQuestion() }.frame(minHeight: 44)
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
                if model.questionToConfirm == nil && model.pendingQuestion == nil && !typing {
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
            if scenePhase == .active, let message { model.speak(message, reading: true) }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background || (phase == .inactive && mic.phase == .listening) {
                if !mic.transcript.isEmpty { model.questionDraft = mic.transcript }
                mic.cancel()
            }
        }
        .onDisappear { mic.cancel() }
    }

    @ViewBuilder private var micDock: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                switch mic.phase {
                case .idle: startRecording()
                case .preparing: mic.cancel()
                case .listening: mic.finish()
                case .finishing: break
                }
            } label: {
                Label(microphoneLabel, systemImage: mic.phase == .listening ? "stop.circle.fill" : "mic.fill")
            }
            .buttonStyle(DocumentButtonStyle()).disabled(mic.phase == .finishing)
            .accessibilityHint("말을 마치고 잠시 쉬면 질문을 읽어줍니다. 확인 후 전송합니다.")
            if mic.phase == .listening || mic.phase == .finishing {
                if mic.phase == .finishing { ProgressView("아이폰이 질문을 받아쓰고 있어요.") }
                else { Text("듣고 있어요.").font(.callout) }
                Button(mic.phase == .finishing ? "받아쓰기 취소" : "녹음 취소") { mic.cancel() }.frame(minHeight: 44)
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
            Button("입력한 질문 확인하기") { stage(model.questionDraft) }
                .buttonStyle(DocumentButtonStyle(prominent: false))
                .disabled(model.questionDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }.padding(16).documentCard().disabled(mic.phase != .idle)

        VStack(alignment: .leading, spacing: 2) {
            Text("예시 질문").font(.headline).padding(.bottom, 6).accessibilityAddTraits(.isHeader)
            ScrollView(.horizontal) {
                HStack(spacing: 8) {
                    ForEach(["이거 어떤 서류야?", "한 줄로 요약해줘", "내 정보는 누구에게 제공돼?", "동의를 거부하면 어떻게 돼?", "대출 개인정보가 잘못 쓰인 실제 사례가 있어?"], id: \.self) { text in
                        Button { stage(text) } label: {
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
            Button { model.confirmQuestion() } label: { Label(model.localAnswer == nil ? "맞아요, 질문 보내기" : "맞아요, 저장된 답 듣기", systemImage: "checkmark") }
                .buttonStyle(DocumentButtonStyle()).disabled(!canSend)
            Button { model.repeatQuestion() } label: { Label("질문 다시 듣기", systemImage: "speaker.wave.2.fill") }
                .buttonStyle(DocumentButtonStyle(prominent: false))
            Button { model.discardQuestion(); startRecording() } label: { Label("다시 말하기", systemImage: "mic.fill") }
                .buttonStyle(DocumentButtonStyle(prominent: false))
            Button {
                model.discardQuestion()
                model.questionDraft = text
                Task { @MainActor in
                    await Task.yield()
                    typing = true
                }
            } label: { Label("글자로 수정하기", systemImage: "pencil") }
            .buttonStyle(DocumentButtonStyle(prominent: false))
            Button("질문 취소") { model.discardQuestion() }.frame(maxWidth: .infinity, minHeight: 44)
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
        Button("읽기 멈추기") { model.stopSpeech() }.frame(minHeight: 44)
    }

    private func startRecording() {
        guard scenePhase == .active else { return }
        typing = false
        model.stopSpeech()
        mic.begin(onQuestion: stage)
    }

    private func stage(_ text: String) {
        typing = false
        model.stageQuestion(text)
        if model.questionToConfirm != nil { model.questionDraft = "" }
    }
}
