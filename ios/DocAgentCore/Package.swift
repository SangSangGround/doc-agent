// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DocAgentCore",
    platforms: [.iOS(.v17), .macOS(.v13)],
    products: [.library(name: "DocAgentCore", targets: ["DocAgentCore"]),
               .executable(name: "docagent-connection-check", targets: ["ConnectionCheck"])],
    targets: [
        .target(name: "DocAgentCore"),
        .executableTarget(name: "ConnectionCheck", dependencies: ["DocAgentCore"]),
        .testTarget(name: "DocAgentCoreTests", dependencies: ["DocAgentCore"], resources: [.copy("Fixtures")])
    ]
)
