import AVFoundation
import Combine
import DocAgentCore
import SwiftUI

struct WritingPage {
    let reference: WritingReference
    let fields: [WritingTarget]
    let readings: [String: CheckboxReading]

    init(reference: WritingReference, fields: [WritingTarget]) {
        self.reference = reference
        self.fields = fields
        readings = Dictionary(uniqueKeysWithValues: fields.filter { $0.kind == .checkbox }
            .map { ($0.id, CheckboxInspection.read($0, in: reference.image)) })
    }
}

@MainActor
struct WritingGuideSheet: View {
    @ObservedObject var model: HomeModel
    let retake: () -> Void
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase
    @State private var reference: WritingReference?
    @State private var page: WritingPage?
    @State private var guiding = false
    @State private var permissionToSend = false
    @State private var attempt = 0
    @State private var status = "작성 위치를 준비하고 있어요."
    @State private var error: String?
    @State private var working = true
    @AccessibilityFocusState private var heading: Bool

    private var selected: WritingTarget? {
        page?.fields.first { $0.id == model.writingTargetID }
    }

    var body: some View {
        NavigationStack {
            Group {
                if guiding, let selected, let page {
                    LiveWritingView(model: model, page: page, field: selected, retake: retake)
                        .id(selected.id)
                } else {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 16) {
                            Text("체크·서명 안내").font(.largeTitle.bold())
                                .accessibilityAddTraits(.isHeader).accessibilityFocused($heading)
                            if let page {
                                if let selected {
                                    Text("선택한 작성란 · \(selected.label)").font(.title2.bold())
                                    Button("이 칸으로 손가락 안내 이어가기") { guiding = true }
                                        .buttonStyle(DocumentButtonStyle())
                                        .accessibilityHint("후면 카메라를 켜고 선택한 칸까지 검지를 안내합니다.")
                                } else if page.fields.isEmpty {
                                    Text("안전하게 안내할 수 있는 작성란을 찾지 못했어요. 손을 치우고 문서 전체를 다시 촬영해 주세요.")
                                } else {
                                    Text("작성할 곳을 고르면 카메라가 열려요. 체크 상태는 마지막에 촬영한 사진에서 읽었어요.")
                                    ForEach(page.fields) { field in
                                        Button {
                                            model.stopSpeech()
                                            model.writingTargetID = field.id
                                            guiding = true
                                        } label: {
                                            DocumentRow(title: field.label, detail: detail(field, page: page),
                                                        symbol: symbol(field), tint: DocumentStyle.ink, fill: DocumentStyle.raised)
                                        }.buttonStyle(.plain).accessibilityHint("아이폰 후면 카메라로 이 칸까지 검지를 안내합니다.")
                                    }
                                }
                                Button("새로 촬영해 체크 상태 확인", action: retake)
                                    .buttonStyle(DocumentButtonStyle(prominent: false))
                                    .accessibilityHint("새 사진으로 현재 읽던 문서를 교체합니다.")
                            } else if working {
                                ProgressView(status)
                            } else {
                                if let error { Text(error).foregroundStyle(DocumentStyle.caution) }
                                if reference != nil && model.demoForm == nil {
                                    Text("이 사진에서 작성란을 찾으려면 사진 한 장을 같은 Wi-Fi의 맥을 거쳐 Google Gemini에 보냅니다. 이름이나 이미 적은 내용도 포함될 수 있어요. 이후 손끝 영상은 아이폰에서 처리해요.")
                                    Button(error == nil ? "이 사진을 보내 작성란 찾기" : "이 사진으로 다시 요청하기") {
                                        permissionToSend = true
                                        attempt += 1
                                    }.buttonStyle(DocumentButtonStyle())
                                        .accessibilityLabel("Google Gemini에 이 문서 사진 한 장을 보내 작성란 찾기")
                                    Button("사진을 다시 촬영", action: retake)
                                        .buttonStyle(DocumentButtonStyle(prominent: false))
                                        .accessibilityHint("새 사진으로 현재 문서를 교체하고 아이폰에서 양식을 확인합니다.")
                                } else if error != nil {
                                    Button("다시 시도") { attempt += 1 }.buttonStyle(DocumentButtonStyle())
                                }
                            }
                        }.padding(20).frame(maxWidth: 580).frame(maxWidth: .infinity)
                    }.background(DocumentStyle.canvas)
                }
            }
            .foregroundStyle(DocumentStyle.ink)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(selected == nil ? "읽기 결과로" : "다른 칸 선택") {
                        model.stopSpeech()
                        if selected != nil {
                            guiding = false
                            model.writingTargetID = nil
                            heading = true
                        } else {
                            dismiss()
                        }
                    }
                }
            }
            .navigationTitle(selected?.label ?? "작성 안내").navigationBarTitleDisplayMode(.inline)
        }
        .task(id: attempt) {
            // Only the explicit send button grants one request, including after a failed upload.
            let send = permissionToSend
            permissionToSend = false
            await prepare(sendAfterConsent: send)
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active {
                // System permission prompts can temporarily deactivate the app before capture starts.
                if phase == .background || AVCaptureDevice.authorizationStatus(for: .video) != .notDetermined {
                    guiding = false
                }
                permissionToSend = false
            }
        }
        .onDisappear { model.stopSpeech() }
    }

    private func symbol(_ field: WritingTarget) -> String {
        switch field.kind {
        case .checkbox: return "square"
        case .signature: return "signature"
        case .input: return "pencil"
        }
    }

    private func detail(_ field: WritingTarget, page: WritingPage) -> String {
        switch field.kind {
        case .checkbox: return "체크박스 · " + (page.readings[field.id] ?? .unclear).rawValue
        case .signature: return "서명할 위치 안내"
        case .input: return "직접 작성할 위치 안내"
        }
    }

    private func prepare(sendAfterConsent: Bool) async {
        guard let photo = model.writingPhoto, let documentID = model.documentID else { return }
        if let existing = model.writingPage {
            page = existing
            working = false
            return
        }
        working = true
        error = nil
        let known = model.demoForm?.targets
        do {
            let source: WritingReference
            if let reference { source = reference }
            else {
                source = try await Task.detached(priority: .userInitiated) {
                    try WritingReference(photo: photo, rectify: known == nil)
                }.value
                try Task.checkCancellation()
                guard model.documentID == documentID else { return }
                reference = source
            }
            let fields: [WritingTarget]
            let sameReference = model.writingReferenceDigest == source.digest
            if sameReference, let saved = model.writingFields { fields = saved }
            else if let known { fields = known }
            else {
                if !sameReference && model.writingReferenceDigest != nil {
                    model.writingTargetID = nil
                    error = "사진에서 작성 위치를 다시 확인해 주세요."
                }
                guard sendAfterConsent, UIApplication.shared.applicationState == .active else {
                    working = false
                    return
                }
                status = "Gemini가 이 사진의 작성란을 찾고 있어요."
                fields = try await CloudDocumentAPI.shared.detectFieldsAfterConsent(source)
            }
            try Task.checkCancellation()
            guard model.documentID == documentID else { return }
            let prepared = await Task.detached(priority: .userInitiated) {
                WritingPage(reference: source, fields: fields)
            }.value
            try Task.checkCancellation()
            guard model.documentID == documentID else { return }
            model.storeWritingPage(prepared)
            page = prepared
            working = false
            heading = true
            if !sameReference && UIApplication.shared.applicationState == .active {
                model.speak(fields.isEmpty ? "안내할 작성란을 찾지 못했어요." : "작성 위치 \(fields.count)곳을 찾았어요. 원하는 칸을 선택해 주세요.")
            }
        } catch {
            guard !Task.isCancelled, model.documentID == documentID else { return }
            working = false
            self.error = error.localizedDescription
            if UIApplication.shared.applicationState == .active { model.speak(error.localizedDescription) }
        }
    }
}

@MainActor
private struct LiveWritingView: View {
    @ObservedObject var model: HomeModel
    @StateObject private var camera: WritingCameraController
    @StateObject private var sound = CaptureSound()
    @State private var lastSpeech = Date.distantPast
    @State private var spoken = ""
    @State private var lastBeep = Date.distantPast
    @State private var atTarget = false
    @Environment(\.scenePhase) private var scenePhase
    private let cueTimer = Timer.publish(every: 0.08, on: .main, in: .common).autoconnect()
    let field: WritingTarget
    let retake: () -> Void

    init(model: HomeModel, page: WritingPage, field: WritingTarget, retake: @escaping () -> Void) {
        self.model = model
        self.field = field
        self.retake = retake
        _camera = StateObject(wrappedValue: WritingCameraController(reference: page.reference.image, target: field, fields: page.fields))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                GeometryReader { geo in
                    let width = min(geo.size.width, geo.size.height * camera.frameAspect)
                    let height = width / camera.frameAspect
                    ZStack {
                        WritingPreview(session: camera.session)
                        Canvas { ctx, sz in
                            func point(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x * sz.width, y: p.y * sz.height) }
                            let outline = camera.feedback.targetOutline
                            if let first = outline.first {
                                var box = Path()
                                box.move(to: point(first))
                                for p in outline.dropFirst() { box.addLine(to: point(p)) }
                                box.closeSubpath()
                                ctx.stroke(box, with: .color(.yellow), lineWidth: 3)
                            }
                            if let tip = camera.feedback.fingertip {
                                let center = point(tip)
                                ctx.fill(Path(ellipseIn: CGRect(x: center.x-5, y: center.y-5, width: 10, height: 10)), with: .color(.cyan))
                            }
                        }.allowsHitTesting(false)
                    }.frame(width: width, height: height).position(x: geo.size.width/2, y: geo.size.height/2)
                }.aspectRatio(0.9, contentMode: .fit).background(.black)
                    .clipShape(RoundedRectangle(cornerRadius: 18)).accessibilityHidden(true)
                Text(camera.feedback.message).font(.title2.bold()).accessibilityAddTraits(.isHeader)
                Button("방향 안내 다시 듣기") { model.speak(camera.feedback.message, force: true) }
                    .buttonStyle(DocumentButtonStyle(prominent: false))
                Text("종이는 움직이지 않게 두세요. 도착 소리는 손끝의 위치를 알려줍니다. 종이에 표시한 뒤 새로 촬영하면 체크 상태를 다시 읽어요.")
                    .foregroundStyle(DocumentStyle.muted)
                Button("작성 후 새로 촬영", action: retake).buttonStyle(DocumentButtonStyle())
                    .accessibilityHint("새 사진으로 읽던 문서를 교체합니다.")
            }.padding(20).frame(maxWidth: 580).frame(maxWidth: .infinity)
        }.background(DocumentStyle.canvas)
        .onAppear {
            guard scenePhase == .active else { return }
            startGuide()
            model.speak("\(field.label) 안내를 시작해요. 같은 문서를 비추고 한 손의 검지를 펴 주세요.", force: true)
        }
        .onReceive(cueTimer) { now in
            guard scenePhase == .active else { return }
            let feedback = camera.feedback
            let reached = feedback.arrived && !atTarget
            atTarget = feedback.arrived
            if reached {
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                sound.play(.entered, quietly: model.guidanceSpeaking)
            } else if let interval = feedback.beepInterval, now.timeIntervalSince(lastBeep) >= interval {
                sound.play(.aligned, quietly: model.guidanceSpeaking)
                lastBeep = now
            }
            let elapsed = now.timeIntervalSince(lastSpeech)
            let repeatDue = feedback.repeatSpeechAfter.map { elapsed >= $0 } ?? false
            let changed = feedback.message != spoken && elapsed >= 2
            if reached || (!model.guidanceSpeaking && (changed || repeatDue)) {
                spoken = feedback.message
                lastSpeech = now
                model.speak(spoken)
            }
        }
        .onDisappear { stopGuide() }
    }

    private func startGuide() {
        camera.start()
        try? sound.start()
    }

    private func stopGuide() {
        camera.stop()
        sound.stop()
        model.stopSpeech()
        atTarget = false
        spoken = ""
        lastBeep = .distantPast
        lastSpeech = .distantPast
    }
}

private final class WritingPreviewSurface: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var preview: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

private struct WritingPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> WritingPreviewSurface {
        let view = WritingPreviewSurface()
        view.preview.videoGravity = .resizeAspect
        view.preview.session = session
        return view
    }
    func updateUIView(_ view: WritingPreviewSurface, context: Context) {
        if let connection = view.preview.connection, connection.isVideoRotationAngleSupported(90),
           connection.videoRotationAngle != 90 { connection.videoRotationAngle = 90 }
    }
}
