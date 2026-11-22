import AudioToolbox
import SwiftUI
import UIKit
import DocAgentCore

@MainActor
struct ConsentQuestionView: View {
    @ObservedObject var model: HomeModel
    @StateObject private var mic = QuestionMicrophone()
    @State private var draft = ""
    /// 말로 질문한 뒤, 답이 끝날 때마다 마이크를 다시 켜는 중인지.
    @State private var conversing = false
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
                        case .idle: listen()
                        case .preparing: stopConversation(); mic.cancel()
                        case .listening: mic.finish()
                        case .finishing: break
                        }
                    } label: { Label(microphoneLabel, systemImage: mic.phase == .listening ? "stop.circle.fill" : "mic.fill") }
                        .buttonStyle(DocumentButtonStyle())
                        .disabled(model.pendingQuestion != nil || mic.phase == .finishing)
                        .accessibilityHint("한 번 누르고 질문하세요. 말을 마치고 잠시 쉬거나 다시 누르면 전송합니다.")
                    Toggle("답을 듣고 바로 이어서 말하기", isOn: $model.continuousConversation)
                        .frame(minHeight: 44)
                        .onChange(of: model.continuousConversation) { _, on in if !on { stopConversation() } }
                        .accessibilityHint("켜면 답이 끝난 뒤 띵 소리와 함께 마이크가 다시 켜집니다. 그만이라고 말하면 끝납니다.")
                    if conversing {
                        Text("대화 중이에요. 답이 끝나고 띵 소리가 나면 바로 말하세요. \"그만\"이라고 하면 끝나요.")
                            .font(.subheadline).fixedSize(horizontal: false, vertical: true)
                    }
                    if mic.phase == .listening || mic.phase == .finishing {
                        Text(mic.transcript.isEmpty ? "듣고 있어요." : mic.transcript)
                        Button("녹음 취소") { stopConversation(); mic.cancel() }.frame(minHeight: 44)
                    }
                    if let error = mic.error { Text(error).foregroundStyle(DocumentStyle.clay) }
                    if let pending = model.pendingQuestion {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(pending).font(.headline)
                            ProgressView("근거를 확인하고 있어요.")
                            Button("질문 취소") { stopConversation(); model.cancelQuestion() }.frame(minHeight: 44)
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
                            Button("읽기 멈추기") { stopConversation(); model.stopSpeech() }.frame(minHeight: 44)
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
        .onChange(of: mic.error) { _, message in
            if let message { stopConversation(); model.speak(message, reading: true) }
        }
        .onChange(of: model.answerSpokenCount) { _, _ in listenAgain() }
        .onChange(of: scenePhase) { _, phase in if phase == .background { stopConversation(); mic.cancel(); draft = "" } }
        .onDisappear { stopConversation(); mic.cancel() }
    }

    private func submit(_ text: String) {
        // 글로 질문하면 마이크를 자동으로 켜지 않는다.
        stopConversation()
        typing = false
        draft = ""
        model.ask(text)
    }

    /// 마이크를 켜고 질문 한 개를 듣는다. 대화 중이면 말이 없을 때 오류 대신 대화를 끝낸다.
    private func listen() {
        typing = false
        model.stopSpeech()
        mic.begin(waitSeconds: conversing ? 8 : 12,
                  onSilence: conversing ? { endConversation("대화를 마칠게요. 다시 질문하려면 말로 질문하기를 눌러 주세요.") } : nil) { question in
            if Self.isStopCommand(question) {
                endConversation("대화를 마칠게요.")
            } else {
                conversing = model.continuousConversation
                model.ask(question)
            }
        }
    }

    /// 답을 끝까지 읽었을 때 불린다. 띵 소리와 진동으로 알린 뒤 마이크를 다시 켠다.
    private func listenAgain() {
        guard conversing, model.continuousConversation, mic.phase == .idle, model.pendingQuestion == nil else { return }
        AudioServicesPlaySystemSound(1113)
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        Task {
            // 띵 소리가 마이크에 질문으로 들어가지 않도록 잠시 기다린다.
            try? await Task.sleep(for: .milliseconds(450))
            guard conversing, mic.phase == .idle, model.pendingQuestion == nil, !model.synthesizer.isSpeaking else { return }
            listen()
        }
    }

    private func stopConversation() { conversing = false }

    private func endConversation(_ message: String) {
        stopConversation()
        model.speak(message, reading: true)
    }

    /// "그만", "끝", "종료", "됐어", "고마워"처럼 짧게 대화를 끝내는 말인지.
    static func isStopCommand(_ text: String) -> Bool {
        let words = text.filter { $0.isLetter }
        guard !words.isEmpty, words.count <= 8 else { return false }
        // "어떻게 됐어?" 같은 질문과 헷갈리지 않게 "됐어"는 그 말만 했을 때만 끝낸다.
        return words.contains("그만") || words.hasPrefix("끝") || words.hasPrefix("종료")
            || ["됐어", "됐어요", "이제됐어", "이제됐어요"].contains(words) || words.contains("고마워")
    }
}
