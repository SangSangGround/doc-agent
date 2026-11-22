import AVFoundation
import Combine
import CoreMedia
import DocAgentCore
import Vision

/// A new camera session is opened only for the writing sheet; the original
/// capture state machine and its success tones never run on these frames.
final class WritingCameraController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    @Published private(set) var feedback = WritingGuidance.seeking
    @Published private(set) var frameAspect: CGFloat = 9.0 / 16.0

    private let reference: CGImage
    private let target: WritingTarget
    private let fields: [WritingTarget]
    private let queue = DispatchQueue(label: "docagent.writing-camera", qos: .userInitiated)
    private let hand = VNDetectHumanHandPoseRequest()
    private var registration: WritingPageRegistration?
    private var guidance = WritingGuidance()
    private var active = false
    private var lastFrameTime = -Double.infinity
    private var missingFrames: DispatchWorkItem?

    init(reference: CGImage, target: WritingTarget, fields: [WritingTarget]) {
        self.reference = reference
        self.target = target
        self.fields = fields
        super.init()
        hand.maximumHandCount = 2
        NotificationCenter.default.addObserver(self, selector: #selector(interrupted(_:)),
                                               name: AVCaptureSession.wasInterruptedNotification, object: session)
        NotificationCenter.default.addObserver(self, selector: #selector(interrupted(_:)),
                                               name: AVCaptureSession.runtimeErrorNotification, object: session)
    }

    @objc private func interrupted(_ notification: Notification) {
        queue.async {
            guard self.active else { return }
            self.resetTracking()
            self.publish(.init(message: "카메라 연결이 멈췄어요. 문서와 검지가 보이게 다시 열어 주세요."))
        }
    }

    func start() {
        queue.async {
            guard !self.active else { return }
            self.active = true
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized: self.configureAndRun()
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .video) { permitted in
                    self.queue.async {
                        guard self.active else { return }
                        if permitted { self.configureAndRun() } else { self.showPermissionError() }
                    }
                }
            default: self.showPermissionError()
            }
        }
    }

    func stop() {
        queue.async {
            self.active = false
            if self.session.isRunning { self.session.stopRunning() }
            self.resetTracking()
            self.publish(WritingGuidance.seeking)
        }
    }

    private func resetTracking() {
        missingFrames?.cancel()
        missingFrames = nil
        guidance = WritingGuidance()
        lastFrameTime = -.infinity
    }

    private func showPermissionError() {
        publish(.init(message: "설정에서 문서 도우미의 카메라 접근을 허용해 주세요."))
    }

    private func configureAndRun() {
        if registration == nil { registration = WritingPageRegistration(reference: reference) }
        guard registration != nil else {
            publish(.init(message: "문서 글자의 특징을 충분히 찾지 못했어요. 문서를 밝은 곳에서 다시 찍어 주세요."))
            return
        }
        do {
            if session.inputs.isEmpty { try configure() }
            session.startRunning()
            if !session.isRunning { throw CocoaError(.featureUnsupported) }
        } catch {
            publish(.init(message: "후면 카메라를 열지 못했어요. 작성 안내를 닫고 다시 열어 주세요."))
        }
    }

    private func configure() throws {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        else { throw CocoaError(.featureUnsupported) }
        let input = try AVCaptureDeviceInput(device: device)
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        if session.canSetSessionPreset(.hd1280x720) { session.sessionPreset = .hd1280x720 }
        guard session.canAddInput(input) else { throw CocoaError(.featureUnsupported) }
        session.addInput(input)
        guard session.canAddOutput(output) else {
            session.removeInput(input)
            throw CocoaError(.featureUnsupported)
        }
        session.addOutput(output)
        guard let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(90) else {
            session.removeOutput(output)
            session.removeInput(input)
            throw CocoaError(.featureUnsupported)
        }
        connection.videoRotationAngle = 90
        output.setSampleBufferDelegate(self, queue: queue)
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard active, let registration, let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        guard time.isFinite, time - lastFrameTime >= 0.10 else { return }
        lastFrameTime = time
        let pose = registration.register(buffer)
        var tip: CGPoint?
        if pose != nil {
            let request = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up)
            if (try? request.perform([hand])) != nil, let hands = hand.results, hands.count == 1,
               let point = try? hands[0].recognizedPoint(.indexTip), point.confidence >= 0.55 {
                tip = CGPoint(x: point.location.x, y: 1 - point.location.y)
            }
        }
        let value = guidance.update(pose: pose, fingertip: tip, target: target, fields: fields, time: time,
                                    referenceAspect: CGFloat(reference.width) / CGFloat(reference.height))
        publish(value, aspect: CGFloat(CVPixelBufferGetWidth(buffer)) / CGFloat(CVPixelBufferGetHeight(buffer)))
        // A frozen camera must not leave a latched arrival message or approach tone active.
        missingFrames?.cancel()
        let timeout = DispatchWorkItem { [weak self] in
            guard let self, self.active else { return }
            self.guidance = WritingGuidance()
            self.publish(WritingGuidance.seeking)
        }
        missingFrames = timeout
        queue.asyncAfter(deadline: .now() + 0.85, execute: timeout)
    }

    private func publish(_ value: WritingFeedback, aspect: CGFloat? = nil) {
        DispatchQueue.main.async {
            if let aspect, self.frameAspect != aspect { self.frameAspect = aspect }
            if self.feedback != value { self.feedback = value }
        }
    }
}
