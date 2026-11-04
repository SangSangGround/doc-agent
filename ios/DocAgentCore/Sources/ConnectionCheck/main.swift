import Foundation
import DocAgentCore

@main
struct ConnectionCheck {
    static func main() async {
        do {
            if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--check-workflow" {
                try await checkWorkflow(pairing: CommandLine.arguments[2], directory: CommandLine.arguments[3]); return
            }
            if CommandLine.arguments.count == 3 && CommandLine.arguments[1] == "--check-contracts" {
                try checkContracts(fixture: CommandLine.arguments[2]); return
            }
            guard CommandLine.arguments.count == 2 else { throw ConnectionError.invalidPairing }
            let text = try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
            let client = APIClient(settings: try ConnectionSettings.parse(text))
            defer { client.close() }
            let state: HomeState = try await client.request()
            print("HTTPS 연결·인증서 확인·JSON 해석 성공. 모드: \(state.purpose), 항목 수: \(state.session.groups.count)")
        } catch {
            fputs("연결 검증 실패: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}

private func checkContracts(fixture: String) throws {
    var cases = 0
    func require(_ condition: Bool) throws {
        guard condition else { throw ConnectionError.server("Swift 통신 계약 검사 실패") }
        cases += 1
    }
    let base: [String: Any] = ["schema_version": 1, "server_url": "https://192.168.0.2:8766",
                                "token": String(repeating: "a", count: 43), "certificate_sha256": String(repeating: "b", count: 64)]
    func parse(_ body: [String: Any]) throws -> ConnectionSettings {
        try ConnectionSettings.parse(String(data: JSONSerialization.data(withJSONObject: body), encoding: .utf8)!)
    }
    try require(try parse(base).serverUrl.host == "192.168.0.2")
    for url in ["http://192.168.0.2:8766", "https://8.8.8.8:8766", "https://example.com:8766",
                "https://user:pass@192.168.0.2:8766", "https://192.168.0.2:8766/?token=a", "https://192.168.0.2:8766/api"] {
        var body = base; body["server_url"] = url
        try require((try? parse(body)) == nil)
    }
    for id in ["iPhone14,2", "iPhone14,5", "iPhone15,4", "iPhone17,3"] { try require(SupportedDevice.allows(id)) }
    for id in ["iPhone13,2", "iPhone14,6", "iPad13,1"] { try require(!SupportedDevice.allows(id)) }
    let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
    let state = try decoder.decode(HomeState.self, from: Data(contentsOf: URL(fileURLWithPath: fixture)))
    try require(state.session.documentId == "synthetic-home-consent")
    try require(state.session.groups.count == 2 && !state.session.complete)
    print("Swift 연결 설정·지원 기종·Python 응답 계약 \(cases)건 확인")
}

private func checkWorkflow(pairing: String, directory: String) async throws {
    let client = APIClient(settings: try ConnectionSettings.parse(String(contentsOfFile: pairing, encoding: .utf8)))
    defer { client.close() }
    let initial: HomeState = try await client.request()
    guard initial.purpose == "demo", initial.session.documentId == "synthetic-home-consent" else {
        throw ConnectionError.server("합성 홈 데모에서만 실행할 수 있습니다.")
    }
    func require(_ value: Bool) throws {
        guard value else { throw ConnectionError.server("합성 HTTPS 흐름 검증 실패") }
    }
    func capture(_ name: String) async throws -> HomeState {
        let image = try Data(contentsOf: URL(fileURLWithPath: directory).appendingPathComponent(name + ".png"))
        return try await client.request(.capture, body: ["image": image.base64EncodedString(), "mode": "scan"])
    }
    let _: HomeState = try await client.request(.resetCapture)
    let blank = try await capture("reference")
    try require(blank.documentReady)
    let read: HomeState = try await client.request(.read, body: ["scope": "page"])
    try require(!(read.instruction ?? "").isEmpty)
    for group in blank.session.groups {
        let _: HomeState = try await client.request(.select, body: ["group_id": group.id, "option_ids": ["disagree"]])
        let confirmed: HomeState = try await client.request(.confirm, body: ["group_id": group.id])
        try require(!confirmed.canVerify)
    }
    let _ = try await capture("correct")
    let correct: HomeState = try await client.request(.verify)
    try require(correct.session.complete)
    let _ = try await capture("duplicate")
    let duplicate: HomeState = try await client.request(.verify)
    try require(!duplicate.session.complete && duplicate.session.groups.contains { $0.verification.status == "multiple" })
    let blurred = try await capture("blurred")
    try require(!blurred.documentReady && !blurred.session.complete)
    let cleared: HomeState = try await client.request(.resetCapture)
    try require(!cleared.hasCapture && !cleared.canVerify)
    print("Swift ↔ Python HTTPS 합성 흐름 통과: 원문 읽기·선택 확정·재촬영·일치·중복·흐림·초기화")
}
