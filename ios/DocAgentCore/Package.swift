// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DocAgentCore",
    platforms: [.iOS(.v17), .macOS(.v13)],
    products: [.library(name: "DocAgentCore", targets: ["DocAgentCore"]),
               .executable(name: "docagent-connection-check", targets: ["ConnectionCheck"])],
    dependencies: [
        .package(url: "https://github.com/yeatse/opencv-spm.git", exact: "4.10.0")
    ],
    targets: [
        .target(name: "WritingVision", dependencies: [.product(name: "OpenCV", package: "opencv-spm")]),
        .target(name: "DocAgentCore", dependencies: ["WritingVision"],
                exclude: ["ReviewModel", "DocumentLanguageModel.swift", "VisualReview.swift", "QuestionWhisper.swift",
                          "ConsentConversation.swift", "ConsentDocument.swift", "PageQuestion.swift", "AttachmentSection.swift"],
                resources: [.copy("consent-library.json")]),
        .executableTarget(name: "ConnectionCheck", dependencies: ["DocAgentCore"]),
        .testTarget(name: "DocAgentCoreTests", dependencies: ["DocAgentCore"],
                    exclude: ["QuestionWhisperTests.swift", "ConsentConversationTests.swift", "PageQuestionTests.swift"],
                    resources: [.copy("Fixtures")])
    ]
)
