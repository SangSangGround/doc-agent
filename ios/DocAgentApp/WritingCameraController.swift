import AVFoundation
import Combine
import CoreMedia
import DocAgentCore
import MediaPipeTasksVision

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
    private var hand: HandLandmarker?
    private var registration: WritingPageRegistration?
    private var guidance = WritingGuidance()
    private var active = false
    private var lastTimestampMs = -1
    private var missingFrames: DispatchWorkItem?

    init(reference: CGImage, target: WritingTarget, fields: [WritingTarget]) {
        self.reference = reference
        self.target = target
        self.fields = fields
        super.init()
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
            self.hand = nil
            if self.session.isRunning { self.session.stopRunning() }
            self.resetTracking()
            self.publish(WritingGuidance.seeking)
        }
    }

    private func resetTracking() {
        missingFrames?.cancel()
        missingFrames = nil
        guidance = WritingGuidance()
        lastTimestampMs = -1
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
            guard let model = Bundle.main.path(forResource: "hand_landmarker", ofType: "task") else {
                throw CocoaError(.fileNoSuchFile)
            }
            let options = HandLandmarkerOptions()
            options.baseOptions.modelAssetPath = model
            options.runningMode = .video
            options.numHands = 1
            options.minHandDetectionConfidence = 0.6
            options.minHandPresenceConfidence = 0.6
            options.minTrackingConfidence = 0.6
            hand = try HandLandmarker(options: options)
        } catch {
            publish(.init(message: "손 추적 모델을 열지 못했어요. 작성 안내를 다시 열어 주세요."))
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
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
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
        guard time.isFinite, time >= 0 else { return }
        let timestampMs = Int(time * 1000)
        guard timestampMs > lastTimestampMs else { return }
        lastTimestampMs = timestampMs
        let pose = registration.register(buffer)
        var tip: CGPoint?
        if pose != nil, let hand,
           let image = try? MPImage(pixelBuffer: buffer),
           let result = try? hand.detect(videoFrame: image, timestampInMilliseconds: timestampMs),
           let landmarks = result.landmarks.first, landmarks.count > 8 {
            let indexTip = landmarks[8]
            tip = CGPoint(x: CGFloat(indexTip.x), y: CGFloat(indexTip.y))
        }
        let value = guidance.update(pose: pose, fingertip: tip, target: target, fields: fields, time: time,
                                    paperCoordinates: registration.paperCoordinates,
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
