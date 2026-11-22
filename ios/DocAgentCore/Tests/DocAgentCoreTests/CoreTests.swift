import XCTest
@testable import DocAgentCore

final class CoreTests: XCTestCase {
    private var payload: [String: Any] {
        ["schema_version": 1, "server_url": "https://192.168.0.2:8766",
         "token": String(repeating: "a", count: 43), "certificate_sha256": String(repeating: "b", count: 64)]
    }
    private func parse(_ value: [String: Any]) throws -> ConnectionSettings {
        try ConnectionSettings.parse(String(data: JSONSerialization.data(withJSONObject: value), encoding: .utf8)!)
    }
    func testValidPairing() throws {
        let settings = try parse(payload)
        XCTAssertEqual(settings.serverUrl.host, "192.168.0.2")
        XCTAssertEqual(settings.certificateSha256.count, 64)
    }
    func testUnsafeOrMisleadingEndpointsRejected() {
        for url in ["http://192.168.0.2:8766", "https://example.com:8766", "https://8.8.8.8:8766",
                    "https://192.168.0.2:8766/?token=a", "https://user:pass@192.168.0.2:8766",
                    "https://192.168.0.2:8766/api", "https://0.0.0.0:8766", "https://192.168.0.2:0"] {
            var value = payload; value["server_url"] = url
            XCTAssertThrowsError(try parse(value), url)
        }
    }
    func testTokenPinAndVersionAreRequired() {
        for (key, value) in [("schema_version", 2 as Any), ("schema_version", true as Any),
                              ("token", "short" as Any), ("token", String(repeating: "가", count: 43) as Any),
                              ("certificate_sha256", String(repeating: "z", count: 64) as Any)] {
            var data = payload; data[key] = value
            XCTAssertThrowsError(try parse(data), key)
        }
        XCTAssertThrowsError(try ConnectionSettings.parse("[]"))
    }
    func testSupportedPhones() {
        for id in ["iPhone14,2", "iPhone14,3", "iPhone14,4", "iPhone14,5", "iPhone14,7", "iPhone15,4", "iPhone17,3"] {
            XCTAssertTrue(SupportedDevice.allows(id), id)
        }
        for id in ["iPhone13,2", "iPhone14,6", "iPad13,1", "arm64", "iPhonefoo,3", "iPhone15,0"] {
            XCTAssertFalse(SupportedDevice.allows(id), id)
        }
    }
    func testPythonStateContractDecodes() throws {
        let url = Bundle.module.url(forResource: "state", withExtension: "json", subdirectory: "Fixtures")!
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let state = try decoder.decode(HomeState.self, from: Data(contentsOf: url))
        XCTAssertEqual(state.session.documentId, "synthetic-home-consent")
        XCTAssertEqual(state.session.groups.count, 2)
        XCTAssertFalse(state.session.complete)
        XCTAssertNil(state.session.currentGroup.pendingOptionIds)
        XCTAssertEqual(state.session.preferences.speechRate, 1)
    }
}
