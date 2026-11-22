import Foundation

public struct HomeState: Decodable {
    public let session: DocumentSession
    public let documentReady: Bool
    public let hasCapture: Bool
    public let canVerify: Bool
    public let purpose: String
    public let title: String
    public let instruction: String?
}
public struct DocumentSession: Decodable {
    public let documentId: String
    public let documentVersion: String
    public let currentIndex: Int
    public let currentGroup: DocumentGroup
    public let groups: [DocumentGroup]
    public let complete: Bool
    public let preferences: ReadingPreferences
}
public struct ReadingPreferences: Decodable {
    public let speechRate: Double
}
public struct DocumentGroup: Decodable, Identifiable {
    public let id: String
    public let sourceText: String
    public let minSelected: Int
    public let maxSelected: Int
    public let options: [DocumentOption]
    public let selectedOptionIds: [String]?
    public let pendingOptionIds: [String]?
    public let confirmed: Bool
    public let verification: Verification
}
public struct DocumentOption: Decodable, Identifiable {
    public let id: String
    public let label: String
}
public struct Verification: Decodable {
    public let status: String
    public let message: String
    public let markedOptionIds: [String]
}
public struct PrivacyResult: Decodable {
    public let instruction: String
    public let preview: String?
}
public struct FrameResult: Decodable {
    public let instruction: String
    public let shouldAnnounce: Bool
    public let readyToCapture: Bool
}

public enum SupportedDevice {
    /// iPhone 13(mini/Pro/Max 포함) 이후. 같은 내부 세대인 SE 3는 제외한다.
    public static func allows(_ identifier: String) -> Bool {
        guard identifier.hasPrefix("iPhone") else { return false }
        let parts = identifier.dropFirst(6).split(separator: ",")
        guard parts.count == 2, let major = Int(parts[0]), let minor = Int(parts[1]), minor > 0 else { return false }
        return major > 14 || (major == 14 && [2, 3, 4, 5, 7, 8].contains(minor))
    }
}
