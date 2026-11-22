import Foundation
import CryptoKit
import Security

public enum ConnectionError: LocalizedError {
    case invalidPairing, invalidResponse, server(String)
    public var errorDescription: String? {
        switch self {
        case .invalidPairing: return "PC가 만든 연결 정보 전체를 붙여넣어 주세요. 같은 Wi-Fi의 HTTPS 주소가 필요합니다."
        case .invalidResponse: return "PC 응답을 읽지 못했습니다. 앱과 서버 버전을 확인해 주세요."
        case .server(let message): return message
        }
    }
}

public struct ConnectionSettings: Decodable {
    public let schemaVersion: Int
    public let serverUrl: URL
    public let token: String
    public let certificateSha256: String

    public static func parse(_ text: String) throws -> Self {
        guard let data = text.data(using: .utf8), data.count < 8192 else { throw ConnectionError.invalidPairing }
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let value = try? decoder.decode(Self.self, from: data), value.schemaVersion == 1,
              value.serverUrl.scheme == "https", let host = value.serverUrl.host,
              isPrivateIPv4(host), let port = value.serverUrl.port, (1...65535).contains(port),
              value.serverUrl.user == nil, value.serverUrl.password == nil,
              value.serverUrl.query == nil, value.serverUrl.fragment == nil,
              ["", "/"].contains(value.serverUrl.path),
              value.token.count >= 32, value.token.count <= 128,
              value.token.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "_" || $0 == "-") }),
              value.certificateSha256.count == 64,
              value.certificateSha256.allSatisfy({ "0123456789abcdef".contains($0) })
        else { throw ConnectionError.invalidPairing }
        return value
    }
    private static func isPrivateIPv4(_ host: String) -> Bool {
        let parts = host.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return false }
        let numbers = parts.compactMap { Int($0) }
        guard numbers.count == 4, numbers.allSatisfy({ (0...255).contains($0) }),
              zip(parts, numbers).allSatisfy({ String($0.1) == String($0.0) }) else { return false }
        return numbers[0] == 10 || numbers[0] == 127 ||
            (numbers[0] == 172 && (16...31).contains(numbers[1])) ||
            (numbers[0] == 192 && numbers[1] == 168)
    }
}

final class PinnedConnectionDelegate: NSObject, URLSessionDelegate, URLSessionTaskDelegate {
    private let settings: ConnectionSettings
    init(settings: ConnectionSettings) { self.settings = settings }
    func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              challenge.protectionSpace.host == settings.serverUrl.host,
              let trust = challenge.protectionSpace.serverTrust,
              let chain = SecTrustCopyCertificateChain(trust) as? [SecCertificate],
              let cert = chain.first else {
            completionHandler(.cancelAuthenticationChallenge, nil); return
        }
        let data = SecCertificateCopyData(cert) as Data
        let fingerprint = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard fingerprint == settings.certificateSha256 else {
            completionHandler(.cancelAuthenticationChallenge, nil); return
        }
        // 사용자가 지정한 이 개발 인증서만 신뢰한다. 호스트·유효기간 검사는 유지한다.
        SecTrustSetAnchorCertificates(trust, [cert] as CFArray)
        SecTrustSetAnchorCertificatesOnly(trust, true)
        guard SecTrustEvaluateWithError(trust, nil) else {
            completionHandler(.cancelAuthenticationChallenge, nil); return
        }
        completionHandler(.useCredential, URLCredential(trust: trust))
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil) // 인증 토큰·사진을 다른 주소로 전달하지 않는다.
    }
}

public enum HomeAction: String {
    case frame, capture, privacy, read, move, select, confirm, cancel, verify, preferences
    case resetCapture = "reset_capture"
}

private struct Failure: Decodable { let error: String }

public final class APIClient {
    public let settings: ConnectionSettings
    private let session: URLSession
    public init(settings: ConnectionSettings) {
        self.settings = settings
        let config = URLSessionConfiguration.ephemeral
        config.urlCache = nil; config.httpCookieStorage = nil
        config.timeoutIntervalForRequest = 45; config.timeoutIntervalForResource = 60
        config.waitsForConnectivity = false
        session = URLSession(configuration: config, delegate: PinnedConnectionDelegate(settings: settings), delegateQueue: nil)
    }
    public func close() { session.invalidateAndCancel() }
    public func request<T: Decodable>(_ action: HomeAction? = nil, body: [String: Any] = [:]) async throws -> T {
        let route = action?.rawValue ?? "state"
        var request = URLRequest(url: settings.serverUrl.appendingPathComponent("api/\(route)"))
        request.setValue(settings.token, forHTTPHeaderField: "X-Session-Token")
        request.cachePolicy = .reloadIgnoringLocalCacheData
        if action != nil {
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, data.count <= 24 * 1024 * 1024 else { throw ConnectionError.invalidResponse }
        guard http.statusCode == 200 else {
            let message = (try? JSONDecoder().decode(Failure.self, from: data))?.error ?? "PC 연결을 확인하고 다시 시도해 주세요."
            throw ConnectionError.server(String(message.prefix(300)))
        }
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        do { return try decoder.decode(T.self, from: data) }
        catch { throw ConnectionError.invalidResponse }
    }
}
