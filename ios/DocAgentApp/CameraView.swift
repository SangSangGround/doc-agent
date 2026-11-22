import SwiftUI
import AVFoundation
import AVKit
import CoreImage
import Vision
import DocAgentCore
import UIKit

final class CameraController: NSObject, ObservableObject,
    AVCaptureVideoDataOutputSampleBufferDelegate, AVCapturePhotoCaptureDelegate {
    let session = AVCaptureSession()
    @Published private(set) var step = CaptureStep.searching
    @Published private(set) var framingHint: FramingHint? = .findPage
    @Published private(set) var fitLevel: Int?
    @Published private(set) var movement: FramingTrend?
    @Published private(set) var outline: [CGPoint] = []
    @Published private(set) var running = false
    @Published private(set) var errorMessage: String?
    var onPhoto: ((Data) -> Void)?
    var onBeep: ((CameraCue) -> Void)?
    var onDirection: ((String) -> Void)?
    var maySpeak: (() -> Bool)?

    private let queue = DispatchQueue(label: "docagent.local-camera", qos: .userInitiated)
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let photoOutput = AVCapturePhotoOutput()
    private let startedAt = ContinuousClock.now
    private var elapsed: Double {
        let time = startedAt.duration(to: .now).components
        return Double(time.seconds) + Double(time.attoseconds) / 1e18
    }
    private var device: AVCaptureDevice?
    private var active = false
    private var gate = ManualCapture()
    private var lastPreview = -Double.infinity
    private var feedback = FramingFeedback()
    private var paper: PreviewPage?
    private var previewImage: CGImage?
    // Vision coordinates on the upright portrait frame, origin at lower left.
    private var focusPoint: CGPoint?
    private var focusingOnPaperEdge = false
    private var pendingPhoto: Int64?

    @MainActor func playDuePulse() {
        guard running, step != .capturing, let onBeep, let cue = feedback.pulse(at: elapsed) else { return }
        onBeep(cue)
    }

    func start() {
        queue.async {
            guard !self.active else { return }
            self.active = true
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized: self.configureAndStart()
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .video) { allowed in
                    self.queue.async {
                        guard self.active else { return }
                        if allowed { self.configureAndStart() } else { self.permissionError() }
                    }
                }
            default: self.permissionError()
            }
        }
    }
    private func permissionError() {
        report("설정에서 카메라 접근을 허용하거나 사진 불러오기를 사용해 주세요.")
    }
    private func report(_ message: String) {
        DispatchQueue.main.async { self.errorMessage = message }
    }
    private func configureAndStart() {
        guard active else { return }
        do {
            if device == nil { try configure() }
            session.startRunning()
            if session.isRunning { DispatchQueue.main.async { self.running = true } }
            else { report("카메라를 시작하지 못했어요. 닫은 뒤 다시 열어 주세요.") }
        } catch {
            report("후면 카메라를 시작하지 못했어요. 사진 불러오기를 사용할 수 있어요.")
        }
    }
    private func configure() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        guard let back = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        else { throw CocoaError(.featureUnsupported) }
        let input = try AVCaptureDeviceInput(device: back)
        let video = AVCaptureVideoDataOutput()
        video.alwaysDiscardsLateVideoFrames = true
        video.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        session.sessionPreset = .photo
        guard session.canAddInput(input) else { throw CocoaError(.featureUnsupported) }
        session.addInput(input)
        guard session.canAddOutput(video) else { throw CocoaError(.featureUnsupported) }
        session.addOutput(video)
        guard session.canAddOutput(photoOutput) else { throw CocoaError(.featureUnsupported) }
        session.addOutput(photoOutput)
        // The preview and Vision buffers share the portrait orientation of the UI.
        for output in [video as AVCaptureOutput, photoOutput as AVCaptureOutput] {
            guard let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(90)
            else { throw CocoaError(.featureUnsupported) }
            connection.videoRotationAngle = 90
        }
        guard let largest = back.activeFormat.supportedMaxPhotoDimensions.max(by: {
            Int64($0.width) * Int64($0.height) < Int64($1.width) * Int64($1.height)
        }) else { throw CocoaError(.featureUnsupported) }
        photoOutput.maxPhotoDimensions = largest
        photoOutput.maxPhotoQualityPrioritization = .quality
        video.setSampleBufferDelegate(self, queue: queue)
        device = back
    }

    func stop() {
        queue.async {
            self.active = false
            self.pendingPhoto = nil
            if self.session.isRunning { self.session.stopRunning() }
            self.paper = nil
            self.previewImage = nil
            self.focusPoint = nil
            self.focusingOnPaperEdge = false
            self.gate = ManualCapture()
            self.publish(.capturing)
            DispatchQueue.main.async {
                self.running = false
                self.feedback = FramingFeedback()
                self.fitLevel = nil
                self.movement = nil
                self.outline = []
                self.framingHint = .findPage
            }
        }
    }
    private func publish(_ previous: CaptureStep) {
        let next = gate.step
        if previous != next {
            DispatchQueue.main.async {
                self.errorMessage = nil
                if !next.needsFraming { self.framingHint = nil }
                self.step = next
            }
        }
    }
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let now = elapsed
        guard active, gate.step != .capturing, now - lastPreview >= 0.15,
              let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastPreview = now
        let original = CIImage(cvPixelBuffer: buffer)
        let scale = min(1, 800 / max(original.extent.width, original.extent.height))
        let small = original.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        previewImage = context.createCGImage(small, from: small.extent)
        let frame = previewImage.map { measureFrame($0, buffer: buffer) }
        paper = frame?.page
        let corners = paper?.corners ?? []
        let clear = focusPoint.map { hasSharpEdges(buffer, near: $0) } ?? false
        let previous = gate.step
        gate.observe(corners: corners, clear: clear, adjusting: device?.isAdjustingFocus ?? true, at: now,
                     partial: frame?.partial)
        if gate.step.needsFraming || gate.step == .framed {
            focusPoint = nil
            focusingOnPaperEdge = false
        }
        publish(previous)
        let next = gate.step, correction = gate.correction
        DispatchQueue.main.async {
            guard self.onBeep != nil, self.running, self.step == next else { return }
            if self.outline.count != corners.count || zip(self.outline, corners).contains(where: {
                hypot($0.0.x - $0.1.x, $0.0.y - $0.1.y) >= 0.009
            }) { self.outline = corners }
            let output = self.feedback.update(step: next, correction: correction, at: now,
                                              speechAvailable: self.maySpeak?() == true)
            let level = self.feedback.fit.map { Int(($0 * 8).rounded()) }
            let settled = self.feedback.direction
            let direction = next == .clipped && (settled == nil || settled == .findPage)
                ? correction?.direction : settled
            let hint: FramingHint? = next.needsFraming && next != .confirming ? direction ?? .findPage : nil
            if self.framingHint != hint { self.framingHint = hint }
            if self.movement != self.feedback.trend { self.movement = self.feedback.trend }
            if self.fitLevel != level { self.fitLevel = level }
            if let message = output.message { self.onDirection?(message) }
            if let cue = output.cue { self.onBeep?(cue) }
        }
    }

    private func measureFrame(_ image: CGImage, buffer: CVPixelBuffer) -> (page: PreviewPage?, partial: FramingCorrection?) {
        guard CVPixelBufferGetPlaneCount(buffer) == 2,
              CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else { return (nil, nil) }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let yPlane = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
              let cbcr = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) else { return (nil, nil) }
        let luma = yPlane.assumingMemoryBound(to: UInt8.self), colors = cbcr.assumingMemoryBound(to: UInt8.self)
        let width = CVPixelBufferGetWidthOfPlane(buffer, 0), height = CVPixelBufferGetHeightOfPlane(buffer, 0)
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0), uvStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        return PreviewDocumentDetector.observe(in: image) { point in
            let x = Int(point.x * CGFloat(width)), y = Int((1 - point.y) * CGFloat(height))
            guard (0..<width).contains(x), (0..<height).contains(y) else { return nil }
            let uv = (y / 2) * uvStride + (x / 2) * 2
            let chroma = max(abs(Int(colors[uv]) - 128), abs(Int(colors[uv + 1]) - 128))
            return (Int(luma[y * yStride + x]), chroma)
        }
    }

    // Sample the luminance plane near the requested text, before preview downscaling.
    private func hasSharpEdges(_ buffer: CVPixelBuffer, near point: CGPoint) -> Bool {
        guard CVPixelBufferGetPlaneCount(buffer) > 0,
              CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else { return false }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return false }
        let pixels = base.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let width = CVPixelBufferGetWidthOfPlane(buffer, 0), height = CVPixelBufferGetHeightOfPlane(buffer, 0)
        let x = Int(point.x * CGFloat(width)), y = Int((1 - point.y) * CGFloat(height))
        let radius = max(20, min(width, height) / 14)
        var energy = 0.0, count = 0, steepSamples = 0
        for row in Swift.stride(from: max(4, y - radius), to: min(height - 4, y + radius), by: 2) {
            for col in Swift.stride(from: max(4, x - radius), to: min(width - 4, x + radius), by: 2) {
                let i = row * stride + col
                let lap = 4 * Int(pixels[i]) - Int(pixels[i-1]) - Int(pixels[i+1]) - Int(pixels[i-stride]) - Int(pixels[i+stride])
                energy += Double(lap * lap)
                count += 1
                if focusingOnPaperEdge {
                    // A single sharp paper border occupies very little of an otherwise blank patch.
                    let short = max(abs(Int(pixels[i-1]) - Int(pixels[i+1])),
                                    abs(Int(pixels[i-stride]) - Int(pixels[i+stride])))
                    let wide = max(abs(Int(pixels[i-4]) - Int(pixels[i+4])),
                                   abs(Int(pixels[i-4*stride]) - Int(pixels[i+4*stride])))
                    if short >= 22 && short * 2 >= wide { steepSamples += 1 }
                }
            }
        }
        return count > 0 && (energy / Double(count) >= 25 || (focusingOnPaperEdge && steepSamples >= radius / 3))
    }

    private func nearestText(in rect: CGRect, image: CGImage) -> CGPoint? {
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let text = VNDetectTextRectanglesRequest()
        text.regionOfInterest = rect.insetBy(dx: rect.width * 0.05, dy: rect.height * 0.05)
        try? VNImageRequestHandler(cgImage: image, options: [:]).perform([text])
        return text.results?.map { CGPoint(x: $0.boundingBox.midX, y: $0.boundingBox.midY) }
            .filter { rect.contains($0) }
            .min { hypot($0.x-center.x, $0.y-center.y) < hypot($1.x-center.x, $1.y-center.y) }
    }
    func focus(devicePoint: CGPoint? = nil) {
        queue.async {
            guard self.active, self.gate.step != .capturing else { return }
            guard let device = self.device, let paper = self.paper, let image = self.previewImage,
                  !self.gate.step.needsFraming else {
                self.report(self.gate.framingHint?.instruction ?? self.gate.step.instruction)
                return
            }
            let previous = self.gate.step
            let bounds = paper.bounds
            // A blank side has no interior text. Its top paper edge still gives the lens contrast.
            let topLeft = paper.corners[0], topRight = paper.corners[1]
            let edge = CGPoint(x: (topLeft.x + topRight.x) / 2, y: (topLeft.y + topRight.y) / 2)
            let textOrTap = devicePoint.map { CGPoint(x: 1 - $0.y, y: 1 - $0.x) }
                ?? self.nearestText(in: bounds, image: image)
            let visionPoint = textOrTap ?? edge
            guard devicePoint == nil || bounds.contains(visionPoint) else {
                self.report("문서 위의 글자를 누르거나 초점 맞추기 버튼을 사용해 주세요.")
                return
            }
            guard self.gate.requestFocus(at: self.elapsed) else {
                self.publish(previous)
                return
            }
            self.focusPoint = visionPoint
            self.focusingOnPaperEdge = textOrTap == nil
            do {
                try device.lockForConfiguration()
                defer { device.unlockForConfiguration() }
                if device.isFocusPointOfInterestSupported {
                    device.focusPointOfInterest = CGPoint(x: 1 - visionPoint.y, y: 1 - visionPoint.x)
                }
                if device.isFocusModeSupported(.autoFocus) { device.focusMode = .autoFocus }
                else if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
                self.publish(previous)
            } catch {
                self.gate = ManualCapture()
                self.publish(previous)
                self.report("초점을 조정하지 못했어요. 다시 맞춰 주세요.")
            }
        }
    }
    func takePhoto() {
        queue.async {
            let previous = self.gate.step
            guard self.active, self.session.isRunning, previous != .capturing else { return }
            guard self.gate.requestShutter(at: self.elapsed, adjusting: self.device?.isAdjustingFocus ?? true)
            else {
                self.publish(previous)
                self.report(self.gate.framingHint?.instruction ?? self.gate.step.instruction)
                return
            }
            self.publish(previous)
            let settings = AVCapturePhotoSettings()
            settings.maxPhotoDimensions = self.photoOutput.maxPhotoDimensions
            settings.photoQualityPrioritization = .quality
            self.pendingPhoto = settings.uniqueID
            self.photoOutput.capturePhoto(with: settings, delegate: self)
        }
    }
    private func finishPhoto(_ id: Int64, data: Data?) {
        queue.async {
            guard self.active, self.pendingPhoto == id else { return }
            self.pendingPhoto = nil
            if let data {
                self.active = false
                self.session.stopRunning()
                DispatchQueue.main.async { self.running = false; self.onPhoto?(data) }
            } else {
                self.gate = ManualCapture()
                self.publish(.capturing)
                self.report("사진을 저장하지 못했어요. 초점을 다시 맞춰 주세요.")
            }
        }
    }
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        finishPhoto(photo.resolvedSettings.uniqueID, data: error == nil ? photo.fileDataRepresentation() : nil)
    }
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishCaptureFor settings: AVCaptureResolvedPhotoSettings, error: Error?) {
        if error != nil { finishPhoto(settings.uniqueID, data: nil) }
    }
}

private final class PreviewSurface: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var preview: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    var onTap: ((CGPoint) -> Void)?
    var corners = [CGPoint]()
    private let outer = CAShapeLayer()
    private let inner = CAShapeLayer()
    override init(frame: CGRect) {
        super.init(frame: frame)
        for border in [outer, inner] {
            border.fillColor = UIColor.clear.cgColor
            border.lineJoin = .round
            layer.addSublayer(border)
        }
        outer.lineWidth = 5
        outer.strokeColor = UIColor.black.withAlphaComponent(0.6).cgColor
        inner.lineWidth = 2.5
        inner.strokeColor = UIColor(red: 1, green: 0.81, blue: 0.24, alpha: 1).cgColor
    }
    required init?(coder: NSCoder) { nil }
    @objc func tapped(_ tap: UITapGestureRecognizer) {
        onTap?(preview.captureDevicePointConverted(fromLayerPoint: tap.location(in: self)))
    }
    override func layoutSubviews() {
        super.layoutSubviews()
        if let connection = preview.connection, connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
        let path = UIBezierPath()
        if corners.count == 4 {
            let points = corners.map { preview.layerPointConverted(fromCaptureDevicePoint: CGPoint(x: 1 - $0.y, y: 1 - $0.x)) }
            for (index, point) in points.enumerated() {
                for adjacent in [points[(index + 3) % 4], points[(index + 1) % 4]] {
                    let dx = adjacent.x - point.x
                    let dy = adjacent.y - point.y
                    let length = hypot(dx, dy)
                    guard length > 0 else { continue }
                    let scale = min(0.22, 32 / length)
                    path.move(to: point)
                    path.addLine(to: CGPoint(x: point.x + scale * dx, y: point.y + scale * dy))
                }
            }
        }
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        outer.path = path.cgPath
        inner.path = path.cgPath
        CATransaction.commit()
    }
}
private struct CameraPreview: UIViewRepresentable {
    @ObservedObject var camera: CameraController
    func makeUIView(context: Context) -> PreviewSurface {
        let view = PreviewSurface()
        view.preview.session = camera.session
        view.preview.videoGravity = .resizeAspect
        view.onTap = { camera.focus(devicePoint: $0) }
        view.addGestureRecognizer(UITapGestureRecognizer(target: view, action: #selector(PreviewSurface.tapped(_:))))
        if #available(iOS 17.2, *) {
            let buttons = AVCaptureEventInteraction(primary: { [weak camera] event in
                if event.phase == .ended { camera?.takePhoto() }
            }, secondary: { [weak camera] event in
                if event.phase == .ended { camera?.focus() }
            })
            buttons.isEnabled = false
            view.addInteraction(buttons)
        }
        return view
    }
    func updateUIView(_ view: PreviewSurface, context: Context) {
        if #available(iOS 17.2, *),
           let buttons = view.interactions.compactMap({ $0 as? AVCaptureEventInteraction }).first {
            buttons.isEnabled = camera.running && camera.step != .capturing
        }
        view.corners = camera.outline
        view.setNeedsLayout()
    }
}

struct CameraSheet: View {
    @ObservedObject var model: HomeModel
    @StateObject private var camera = CameraController()
    @StateObject private var sound = CaptureSound()
    @AccessibilityFocusState private var buttonFocus: CaptureStep?
    @State private var audioUnavailable = false
    @State private var announcedReady = false
    @Environment(\.dynamicTypeSize) private var typeSize
    @Environment(\.scenePhase) private var scenePhase

    private static let fitColors = [DocumentStyle.gold, DocumentStyle.gold, DocumentStyle.gold,
                                    DocumentStyle.orange, DocumentStyle.orange, DocumentStyle.orange,
                                    DocumentStyle.accent, DocumentStyle.accent]

    private var instruction: String {
        if let error = camera.errorMessage { return error }
        if camera.step == .confirming { return camera.step.instruction }
        if camera.step.needsFraming { return (camera.framingHint ?? .findPage).instruction }
        if #available(iOS 17.2, *) {
            if camera.step == .framed { return "위치를 유지하고 볼륨 위 버튼으로 초점을 맞춰 주세요." }
            if camera.step == .ready { return "준비됐어요. 볼륨 아래 버튼을 눌러 촬영하세요." }
        }
        return camera.step.instruction
    }
    private var status: String {
        if camera.errorMessage != nil { return "카메라 안내" }
        switch camera.step {
        case .searching: return "종이를 찾고 있어요"
        case .clipped:
            switch camera.movement {
            case .improving: return "구도가 좋아지고 있어요"
            case .worsening: return "종이가 다시 벗어나고 있어요"
            case nil: return "종이 일부가 잘렸어요"
            }
        case .confirming: return "네 모서리를 확인하고 있어요"
        case .framed: return "문서가 들어왔어요"
        case .focusing: return "초점 확인 중"
        case .ready: return "촬영할 준비가 됐어요"
        case .capturing: return "사진 촬영 중"
        }
    }
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Button {
                    model.stopSpeech()
                    model.cameraPresented = false
                } label: {
                    Image(systemName: "xmark").font(.headline).frame(width: 46, height: 46)
                        .background(.white.opacity(0.15), in: Circle())
                }.accessibilityLabel("촬영 닫기")
                Spacer()
                Text("문서 촬영").font(.headline).accessibilityAddTraits(.isHeader)
                Spacer()
                Color.clear.frame(width: 46, height: 1).accessibilityHidden(true)
            }
            .padding(.horizontal, 16).padding(.vertical, 5)
            .foregroundStyle(.white)
            .background(DocumentStyle.darkInk.ignoresSafeArea(edges: .top))
            GeometryReader { space in
                ScrollView {
                    VStack(spacing: 12) {
                        CameraPreview(camera: camera)
                            .frame(maxWidth: .infinity).frame(height: min(435, max(160, space.size.height * (typeSize.isAccessibilitySize ? 0.35 : 0.62))))
                            .background(DocumentStyle.darkInk)
                            .clipShape(RoundedRectangle(cornerRadius: 20))
                            .overlay { RoundedRectangle(cornerRadius: 20).strokeBorder(DocumentStyle.border).allowsHitTesting(false) }
                            .overlay(alignment: .bottomLeading) {
                                if !typeSize.isAccessibilitySize {
                                    Text(status).font(.caption.bold())
                                        .foregroundStyle(camera.step == .clipped ? .white : DocumentStyle.darkInk)
                                        .padding(.horizontal, 12).padding(.vertical, 7)
                                        .background(camera.step == .clipped ? DocumentStyle.color(0xA62C20) : DocumentStyle.gold, in: Capsule()).padding(10)
                                        .allowsHitTesting(false)
                                }
                            }
                            .accessibilityHidden(true)
                        VStack(alignment: .leading, spacing: 10) {
                            HStack(alignment: .center) {
                                Text(status).font(.title3.bold()).accessibilityAddTraits(.isHeader)
                                Spacer(minLength: 4)
                                Button { model.speak(instruction, reading: true) } label: {
                                    Image(systemName: "speaker.wave.2.fill").frame(width: 44, height: 44)
                                        .background(DocumentStyle.accentFill, in: RoundedRectangle(cornerRadius: 12))
                                }.accessibilityLabel("현재 촬영 안내 다시 듣기")
                            }
                            Text(instruction).font(.body).fixedSize(horizontal: false, vertical: true)
                            if camera.running {
                                let level = camera.step.needsFraming ? min(7, camera.fitLevel ?? 0) : 8
                                HStack(spacing: 5) {
                                    ForEach(0..<8, id: \.self) { index in
                                        Capsule().fill(index < level ? Self.fitColors[index] : DocumentStyle.border).frame(height: 6)
                                    }
                                }.accessibilityHidden(true)
                            }
                            if audioUnavailable {
                                Text("비프음을 켤 수 없어요. 음성 안내와 준비 진동을 사용해 주세요.").font(.subheadline)
                            }
                        }.padding(16).frame(maxWidth: .infinity, alignment: .leading).documentCard()
                    }.padding(.horizontal, 16).padding(.vertical, 12)
                }
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            Group {
                if typeSize.isAccessibilitySize { VStack(spacing: 8) { captureButtons } }
                else { HStack(spacing: 10) { captureButtons } }
            }
            .padding(.horizontal, 16).padding(.vertical, 10)
            .background(DocumentStyle.card.ignoresSafeArea(edges: .bottom))
        }
        .foregroundStyle(DocumentStyle.ink).background(DocumentStyle.canvas).tint(DocumentStyle.accent)
        .task { @MainActor in
            do {
                while !Task.isCancelled {
                    try await Task.sleep(for: .milliseconds(40))
                    camera.playDuePulse()
                }
            } catch { /* SwiftUI cancels the sound clock when the camera closes. */ }
        }
        .onAppear {
            guard scenePhase == .active, model.cameraPresented else { return }
            camera.onPhoto = { if model.cameraPresented { model.capture($0) } }
            camera.onBeep = present
            camera.maySpeak = { model.automaticSpeech && !model.guidanceSpeaking }
            camera.onDirection = { model.speak($0) }
            do { try sound.start() } catch { audioUnavailable = true }
            camera.start()
            model.speak("충전 단자를 몸 쪽으로 두세요. 구도가 좋아질수록 삐 소리가 빨라져요.")
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { stopCamera() }
        }
        .onDisappear { stopCamera() }
        .onChange(of: camera.errorMessage) { _, error in if let error { model.speak(error) } }
        .onChange(of: camera.step) { previous, step in
            if step.needsFraming || (step == .focusing && previous == .framed) { announcedReady = false }
            if step == .framed && (previous == .ready || previous == .focusing) {
                buttonFocus = .framed
                model.speak("초점을 다시 맞춰 주세요.")
            }
        }
    }

    private func stopCamera() {
        guard camera.onPhoto != nil else { return }
        camera.onBeep = nil
        camera.onDirection = nil
        camera.maySpeak = nil
        camera.onPhoto = nil
        camera.stop()
        sound.stop()
        if !model.busy { model.stopSpeech() }
    }

    @ViewBuilder private var captureButtons: some View {
        Button { camera.focus() } label: { Label("초점 맞추기", systemImage: "scope") }
            .buttonStyle(DocumentButtonStyle(prominent: camera.step != .ready))
            .disabled(!camera.running || camera.step.needsFraming || camera.step == .capturing)
            .accessibilityFocused($buttonFocus, equals: .framed)
        Button { camera.takePhoto() } label: { Label("촬영", systemImage: "camera.fill") }
            .buttonStyle(DocumentButtonStyle()).disabled(camera.step != .ready)
            .accessibilityFocused($buttonFocus, equals: .ready)
    }

    private func present(_ cue: CameraCue) {
        sound.play(cue, quietly: model.guidanceSpeaking)
        if cue == .entered {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            if camera.step == .framed {
                buttonFocus = .framed
                model.speak("문서가 들어왔어요. " + instruction)
            }
        } else if cue == .ready && !announcedReady {
            announcedReady = true
            buttonFocus = .ready
            UINotificationFeedbackGenerator().notificationOccurred(.success)
            model.speak(instruction)
        }
    }
}
