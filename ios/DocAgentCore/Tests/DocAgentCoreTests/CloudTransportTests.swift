import Foundation
import XCTest
@testable import DocAgentCore

final class CloudTransportTests: XCTestCase {
    @MainActor func testCloudSessionProhibitsCellularAndDoesNotQueueRequests() async {
        let config = CloudDocumentAPI.configuration()
        XCTAssertFalse(config.allowsCellularAccess)
        XCTAssertFalse(config.waitsForConnectivity)
        XCTAssertNil(config.urlCache)
        XCTAssertNil(config.httpCookieStorage)
    }

}
