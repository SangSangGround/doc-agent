import SwiftUI
import DocAgentCore

@MainActor
struct ConsentQuestionView: View {
    @ObservedObject var model: HomeModel
    @StateObject private var mic = QuestionMicrophone()
    @State private var draft = ""
    @FocusState private var typing: Bool
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.dismiss) private var dismiss

    private var lastTurn: ConsentTurn? { model.turns.last { $0.question != "문서 원문 읽기" } }
    private var microphoneLabel: String {
        switch mic.phase {
        case .idle: return "말로 질문하기"
        case .preparing: return "마이크 준비 취소"
        case .listening: return "말하기 끝내고 질문 보내기"
        case .finishing: return "말한 내용 확인 중"
        }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Button {
                        switch mic.phase {
                        case .idle:
                            typing = false
                            model.stopSpeech()
                            mic.begin { model.ask($0) }
                        case .preparing: mic.cancel()
                        case .listening: mic.finish()
                        case .finishing: break
                        }
                    } label: { Label(microphoneLabel, systemImage: mic.phase == .listening ? "stop.circle.fill" : "mic.fill") }
                        .buttonStyle(DocumentButtonStyle())
                        .disabled(model.pendingQuestion != nil || mic.phase == .finishing)
                        .accessibilityHint("한 번 누르고 질문하세요. 말을 마치고 잠시 쉬거나 다시 누르면 전송합니다.")
                    if mic.phase == .listening || mic.phase == .finishing {
                        Text(mic.transcript.isEmpty ? "듣고 있어요." : mic.transcript)
                        Button("녹음 취소") { mic.cancel() }.frame(minHeight: 44)
                    }
                    if let error = mic.error { Text(error).foregroundStyle(DocumentStyle.clay) }
                    if let pending = model.pendingQuestion {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(pending).font(.headline)
                            ProgressView("근거를 확인하고 있어요.")
                            Button("질문 취소") { model.cancelQuestion() }.frame(minHeight: 44)
                        }.padding().frame(maxWidth: .infinity, alignment: .leading).documentCard()
                    } else if let turn = lastTurn {
                        VStack(alignment: .leading, spacing: 12) {
                            Text(turn.question).font(.headline).accessibilityAddTraits(.isHeader)
                            Text(turn.reply.spoken).fixedSize(horizontal: false, vertical: true)
                            Button { model.speak(turn.reply.spoken, reading: true) } label: {
                                Label("답 다시 듣기", systemImage: "speaker.wave.2")
                            }.frame(minHeight: 44)
                            if !turn.reply.source.isEmpty {
                                Text(turn.reply.source).font(.subheadline)
                                Button("출처 듣기") { model.speak(turn.reply.source, reading: true) }.frame(minHeight: 44)
                                if let url = turn.reply.url {
                                    Link("공식 원문 열기", destination: url)
                                        .frame(minHeight: 44).accessibilityHint("공식 기관 웹페이지를 엽니다.")
                                }
                            }
                            Button("읽기 멈추기") { model.stopSpeech() }.frame(minHeight: 44)
                        }.padding().frame(maxWidth: .infinity, alignment: .leading).documentCard()
                        .disabled(mic.phase != .idle)
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        TextField("질문 입력", text: $draft, axis: .vertical)
                            .textFieldStyle(.roundedBorder).focused($typing)
                            .accessibilityLabel("질문 입력").lineLimit(1...4)
                        Button("입력한 질문 보내기") { submit(draft) }
                            .frame(minHeight: 44).disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }.disabled(mic.phase != .idle || model.pendingQuestion != nil)

                    VStack(alignment: .leading, spacing: 4) {
                        Text("예시 질문").font(.headline).accessibilityAddTraits(.isHeader)
                        ForEach(["내 정보는 누구에게 제공돼?", "동의를 거부하면 어떻게 돼?", "대출 개인정보가 잘못 쓰인 실제 사례가 있어?"], id: \.self) { text in
                            Button(text) { submit(text) }.multilineTextAlignment(.leading).frame(minHeight: 44)
                        }
                    }.disabled(mic.phase != .idle || model.pendingQuestion != nil)
                }.padding(20).frame(maxWidth: 580)
            }
            .background(DocumentStyle.canvas).foregroundStyle(DocumentStyle.ink)
            .navigationTitle("문서에 질문하기").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("닫기") { dismiss() } } }
        }
        .tint(DocumentStyle.accent)
        .onChange(of: mic.error) { _, message in if let message { model.speak(message, reading: true) } }
        .onChange(of: scenePhase) { _, phase in if phase == .background { mic.cancel(); draft = "" } }
        .onDisappear { mic.cancel() }
    }

    private func submit(_ text: String) {
        typing = false
        draft = ""
        model.ask(text)
    }
}
