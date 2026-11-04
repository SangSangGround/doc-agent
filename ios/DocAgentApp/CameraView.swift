import SwiftUI
import AVFoundation
import UIKit
import CoreImage

final class CameraController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    @Published var errorMessage: String?
    @Published var latestJPEG: Data?
    var onFrame: ((Data) -> Void)?
    private let queue = DispatchQueue(label: "docagent.camera")
    private let context = CIContext()
    private let lock = NSLock()
    private var wanted = false
    private var configured = false
    private var lastFrame = Date.distantPast

    func start() {
        lock.lock(); wanted = true; lock.unlock()
        Task {
            let permission = AVCaptureDevice.authorizationStatus(for: .video)
            let granted: Bool
            if permission == .authorized { granted = true }
            else if permission == .notDetermined { granted = await AVCaptureDevice.requestAccess(for: .video) }
            else { granted = false }
            guard granted else {
                DispatchQueue.main.async { self.errorMessage = "카메라 접근을 허용해야 촬영할 수 있습니다. 설정에서 권한을 확인하거나 사진 불러오기를 사용하세요." }
                return
            }
            queue.async { [self] in
                lock.lock(); let shouldStart = wanted; lock.unlock()
                guard shouldStart else { return }
                do {
                    if !configured {
                        session.beginConfiguration()
                        defer { session.commitConfiguration() }
                        session.sessionPreset = .hd1920x1080
                        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                            throw NSError(domain: "Camera", code: 1)
                        }
                        let input = try AVCaptureDeviceInput(device: device)
                        let output = AVCaptureVideoDataOutput()
                        output.alwaysDiscardsLateVideoFrames = true
                        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
                        guard session.canAddInput(input), session.canAddOutput(output) else { throw NSError(domain: "Camera", code: 2) }
                        session.addInput(input); session.addOutput(output)
                        output.setSampleBufferDelegate(self, queue: queue)
                        if let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(90) {
                            connection.videoRotationAngle = 90
                        }
                        configured = true
                    }
                    if !session.isRunning { session.startRunning() }
                } catch {
                    DispatchQueue.main.async { self.errorMessage = "카메라를 시작하지 못했습니다. 실제 아이폰에서 실행하거나 사진 불러오기를 사용하세요." }
                }
            }
        }
    }
    func stop() {
        lock.lock(); wanted = false; lock.unlock()
        queue.async { if self.session.isRunning { self.session.stopRunning() } }
    }
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard Date().timeIntervalSince(lastFrame) >= 1.2, let pixel = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastFrame = Date()
        let ciImage = CIImage(cvPixelBuffer: pixel)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent),
              let data = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.9) else { return }
        DispatchQueue.main.async {
            self.lock.lock(); let active = self.wanted; self.lock.unlock()
            guard active else { return }
            self.latestJPEG = data; self.onFrame?(data)
        }
    }
}

private final class PreviewSurface: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var preview: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    override func layoutSubviews() {
        super.layoutSubviews()
        if let connection = preview.connection, connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
    }
}
private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> PreviewSurface {
        let view = PreviewSurface(); view.preview.session = session
        view.preview.videoGravity = .resizeAspect
        return view
    }
    func updateUIView(_ view: PreviewSurface, context: Context) { view.setNeedsLayout() }
}

@MainActor
struct CameraSheet: View {
    @ObservedObject var model: HomeModel
    @StateObject private var camera = CameraController()
    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                CameraPreview(session: camera.session).accessibilityHidden(true)
                Text(camera.errorMessage ?? model.cameraNotice).font(.title3).padding(.horizontal)
                if model.readyToCapture { Label("촬영 준비됨", systemImage: "checkmark.circle") }
                Button("현재 화면 촬영") {
                    guard let data = camera.latestJPEG else { return }
                    camera.stop(); model.capture(data, scan: false)
                }.buttonStyle(.borderedProminent).controlSize(.large)
                    .disabled(camera.latestJPEG == nil || model.busy)
                Text("종이 전체가 보이도록 맞춰 주세요. 프레임을 PC로 보내 촬영 상태를 확인합니다.").font(.footnote).padding()
            }
            .navigationTitle("문서 촬영")
            .toolbar { ToolbarItem(placement: .cancellationAction) {
                Button("취소") { camera.stop(); model.cameraPresented = false; model.stopSpeech() }
            } }
            .onAppear { camera.onFrame = { data in Task { @MainActor in model.frame(data) } }; camera.start() }
            .onDisappear { camera.stop(); camera.onFrame = nil }
        }
    }
}
