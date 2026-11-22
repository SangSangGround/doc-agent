import Combine
import Foundation
import Network
import Security

/// A connected Wi-Fi route is required even when the Mac is reachable by another interface.
@MainActor
public final class CloudWiFiState: ObservableObject {
    public static let shared = CloudWiFiState()
    @Published public private(set) var available = false
    private let monitor = NWPathMonitor()

    private init() {
        monitor.pathUpdateHandler = { [weak self] path in
            let connected = path.status == .satisfied && path.usesInterfaceType(.wifi)
            DispatchQueue.main.async { self?.available = connected }
        }
        monitor.start(queue: DispatchQueue(label: "DocAgent.WiFi"))
    }
}

enum CloudRequestError: LocalizedError {
    case needsWiFi, needsPairing, keychain, network, authentication, response, busy, upstream(String)

    var errorDescription: String? {
        switch self {
        case .needsWiFi: return "아이폰을 맥과 같은 Wi-Fi에 연결해 주세요."
        case .needsPairing: return "설정의 AI 연결에서 맥이 만든 연결 정보를 등록해 주세요."
        case .keychain: return "연결 정보를 저장하지 못했어요. 기기 잠금을 풀고 다시 시도해 주세요."
        case .network: return "맥의 중계 프로그램과 Wi-Fi 연결을 확인해 주세요."
        case .authentication: return "맥에서 새로 만든 연결 정보를 설정의 AI 연결에 다시 등록해 주세요."
        case .response: return "중계 응답을 확인할 수 없어요. 다시 시도해 주세요."
        case .busy: return "지금 요청이 많아요. 잠시 후 다시 시도해 주세요."
        case .upstream(let message): return message
        }
    }
}

private enum RelayKeychain {
    private static var query: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: "org.sangsangground.DocAgent.cloud-relay",
         kSecAttrAccount as String: "pairing"]
    }
    static func read() -> String? {
        var search = query
        search[kSecReturnData as String] = true
        var item: CFTypeRef?
        guard SecItemCopyMatching(search as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }
    static func save(_ pairing: String) throws {
        let values: [String: Any] = [kSecValueData as String: Data(pairing.utf8),
                                     kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly]
        let status = SecItemUpdate(query as CFDictionary, values as CFDictionary)
        if status == errSecItemNotFound {
            let item = query.merging(values) { _, new in new }
            guard SecItemAdd(item as CFDictionary, nil) == errSecSuccess else { throw CloudRequestError.keychain }
        } else if status != errSecSuccess { throw CloudRequestError.keychain }
    }
    static func clear() throws {
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw CloudRequestError.keychain }
    }
}

public struct RelayStatus: Decodable {
    public let answersReady: Bool
    public let fieldsReady: Bool?
}

/// A relay restart issues a new bearer token. Its certificate and LAN origin are paired together.
@MainActor
public final class CloudDocumentAPI: ObservableObject {
    public static let shared = CloudDocumentAPI()
    @Published public private(set) var pairedAddress: String?
    public var isConfigured: Bool { pairedAddress != nil }

    private let wifi: CloudWiFiState
    private var pairing: ConnectionSettings?
    private var session: URLSession?
    private var networkChanges: AnyCancellable?

    private init() {
        wifi = .shared
        if let saved = RelayKeychain.read(), let settings = try? ConnectionSettings.parse(saved) {
            pairing = settings
            pairedAddress = settings.serverUrl.absoluteString
        }
        networkChanges = wifi.$available.removeDuplicates().sink { [weak self] available in
            if !available { self?.closeSession() }
        }
    }

    public func pair(_ text: String) throws {
        let settings = try ConnectionSettings.parse(text)
        try RelayKeychain.save(text)
        closeSession()
        pairing = settings
        pairedAddress = settings.serverUrl.absoluteString
    }

    public func forget() throws {
        try RelayKeychain.clear()
        closeSession()
        pairing = nil
        pairedAddress = nil
    }

    private func closeSession() {
        session?.invalidateAndCancel()
        session = nil
    }

    static func configuration() -> URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.allowsCellularAccess = false
        config.waitsForConnectivity = false
        config.httpShouldSetCookies = false
        config.httpCookieStorage = nil
        config.urlCache = nil
        config.urlCredentialStorage = nil
        config.timeoutIntervalForRequest = 55
        config.timeoutIntervalForResource = 170
        return config
    }

    public func status() async throws -> RelayStatus {
        try await perform("status")
    }

    public func review(_ page: CloudOCRReviewRequest) async throws -> CloudOCRReviewResponse {
        try await perform("review", body: JSONEncoder().encode(page))
    }

    public func answer(_ question: CloudQuestionRequest) async throws -> CloudQuestionResponse {
        try await perform("answer", body: JSONEncoder().encode(question))
    }

    /// Call only after the person opts to send this still from the writing screen.
    public func detectFieldsAfterConsent(_ reference: WritingReference) async throws -> [WritingTarget] {
        guard try await status().fieldsReady == true else {
            throw CloudRequestError.upstream("맥에서 최신 중계 프로그램을 실행하고 Gemini API 키를 설정해 주세요.")
        }
        struct Payload: Encodable {
            let imageJPEG: Data
            let imageSHA256: String
            let allowImageProcessing = true
        }
        let payload = Payload(imageJPEG: reference.jpeg, imageSHA256: reference.digest)
        let response: DetectedWritingFields = try await perform("fields", body: JSONEncoder().encode(payload), timeout: 160)
        return try response.resolve(for: reference.digest)
    }

    private func perform<T: Decodable>(_ route: String, body: Data? = nil, timeout: TimeInterval = 55) async throws -> T {
        try Task.checkCancellation()
        guard wifi.available else { throw CloudRequestError.needsWiFi }
        guard let pairing else { throw CloudRequestError.needsPairing }
        let client = session ?? URLSession(configuration: Self.configuration(),
                                            delegate: PinnedConnectionDelegate(settings: pairing), delegateQueue: nil)
        session = client
        var request = URLRequest(url: pairing.serverUrl.appendingPathComponent("v1").appendingPathComponent(route))
        request.timeoutInterval = timeout
        request.httpMethod = body == nil ? "GET" : "POST"
        request.httpBody = body
        if body != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        request.setValue("Bearer " + pairing.token, forHTTPHeaderField: "Authorization")
        request.allowsCellularAccess = false
        do {
            let (data, response) = try await client.data(for: request)
            try Task.checkCancellation()
            guard wifi.available else { throw CloudRequestError.needsWiFi }
            guard client === session else { throw CancellationError() }
            guard let http = response as? HTTPURLResponse, data.count <= 256 * 1024 else { throw CloudRequestError.response }
            switch http.statusCode {
            case 200:
                guard let parsed = try? JSONDecoder().decode(T.self, from: data) else { throw CloudRequestError.response }
                return parsed
            case 401, 403: throw CloudRequestError.authentication
            case 429: throw CloudRequestError.busy
            default:
                if let failure = try? JSONDecoder().decode(RelayFailure.self, from: data),
                   !failure.error.isEmpty, failure.error.count <= 256 {
                    throw CloudRequestError.upstream(failure.error)
                }
                throw CloudRequestError.response
            }
        } catch let error as URLError {
            try Task.checkCancellation()
            guard wifi.available else { throw CloudRequestError.needsWiFi }
            if error.code == .cancelled { throw CancellationError() }
            throw CloudRequestError.network
        }
    }
}

private struct RelayFailure: Decodable { let error: String }
