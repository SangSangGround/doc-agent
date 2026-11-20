// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DocAgentCore",
    platforms: [.iOS(.v17), .macOS(.v13)],
    products: [.library(name: "DocAgentCore", targets: ["DocAgentCore"]),
               .executable(name: "docagent-connection-check", targets: ["ConnectionCheck"])],
    targets: [
        .binaryTarget(name: "llama",
                      url: "https://github.com/ggml-org/llama.cpp/releases/download/b12559/llama-b12559-xcframework.zip",
                      checksum: "b5adcc29276e2a41550260520768b5b7453a688680b8eca62d940760843e0a1a"),
        .target(name: "DocAgentCore", dependencies: ["llama"],
                resources: [.copy("ReviewModel"), .copy("consent-library.json")]),
        .executableTarget(name: "ConnectionCheck", dependencies: ["DocAgentCore"]),
        .testTarget(name: "DocAgentCoreTests", dependencies: ["DocAgentCore"], resources: [.copy("Fixtures")])
    ]
)
