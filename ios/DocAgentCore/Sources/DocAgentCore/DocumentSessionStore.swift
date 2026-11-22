import CryptoKit
import Foundation

/// The digest identifies the upright JPEG whose coordinates the field detector used.
public struct WritingCheckpoint: Codable, Sendable {
    public let referenceDigest: String
    public let fields: [WritingTarget]

    public init(referenceDigest: String, fields: [WritingTarget]) {
        self.referenceDigest = referenceDigest
        self.fields = fields
    }
}

public struct DocumentCheckpoint: Codable, Sendable {
    public enum Screen: String, Codable, Sendable { case result, reading, questions, writing }

    public let id: UUID
    public var lines: [PrintedLine] = []
    public var turns: [ConsentTurn] = []
    public var demoForm: DemoConsentForm.Match?
    public var notice = ""
    public var recognitionPending = false
    public var readingIndex = 0
    public var questionDraft = ""
    public var questionToConfirm: String?
    public var screen: Screen = .result
    public var writing: WritingCheckpoint?
    public var selectedTargetID: String?
    public var ocrReviewStatus: String?
    public var canRetryOCRReview = false

    public init(id: UUID = UUID()) {
        self.id = id
    }
}

/// The manifest commits a complete photo + checkpoint pair. Cursor and conversation updates
/// replace only the small checkpoint. Interrupted replacements leave the last committed pair valid.
public struct DocumentSessionStore: Sendable {
    private struct Manifest: Codable {
        let version: Int
        let id: UUID
        let photoDigest: Data
    }

    private static let lock = NSLock()
    private let directory: URL
    private var manifestURL: URL { directory.appendingPathComponent("current.json") }

    public init() {
        self.init(directory: FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CurrentDocument", isDirectory: true))
    }

    public init(directory: URL) { self.directory = directory }

    public func replace(with checkpoint: DocumentCheckpoint, photo: Data) throws {
        try Self.lock.withLock {
            guard !photo.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
            try prepareDirectory()
            let digest = Data(SHA256.hash(data: photo))
            let new = Manifest(version: 1, id: checkpoint.id, photoDigest: digest)
            let old = try readManifest()
            if old?.id == checkpoint.id {
                guard old?.photoDigest == digest else { throw CocoaError(.fileReadCorruptFile) }
                try writeCheckpoint(checkpoint)
                return
            }
            do {
                try photo.write(to: file(checkpoint.id, "photo"), options: Self.writeOptions)
                try writeCheckpoint(checkpoint)
                try JSONEncoder().encode(new).write(to: manifestURL, options: Self.writeOptions)
            } catch {
                try? FileManager.default.removeItem(at: file(checkpoint.id, "photo"))
                try? FileManager.default.removeItem(at: file(checkpoint.id, "json"))
                throw error
            }
            try? removeOrphans(keeping: new.id)
        }
    }

    public func save(_ checkpoint: DocumentCheckpoint) throws {
        try Self.lock.withLock {
            guard try readManifest()?.id == checkpoint.id else { throw CocoaError(.fileNoSuchFile) }
            try writeCheckpoint(checkpoint)
        }
    }

    public func load() throws -> (checkpoint: DocumentCheckpoint, photo: Data)? {
        try Self.lock.withLock {
            guard let manifest = try readManifest() else {
                try? removeOrphans(keeping: nil)
                return nil
            }
            let checkpoint = try JSONDecoder().decode(DocumentCheckpoint.self,
                from: Data(contentsOf: file(manifest.id, "json")))
            let photo = try Data(contentsOf: file(manifest.id, "photo"))
            guard manifest.id == checkpoint.id, Data(SHA256.hash(data: photo)) == manifest.photoDigest else {
                throw CocoaError(.fileReadCorruptFile)
            }
            try? removeOrphans(keeping: manifest.id)
            return (checkpoint, photo)
        }
    }

    public func clear() throws {
        try Self.lock.withLock {
            do { try FileManager.default.removeItem(at: manifestURL) }
            catch let error as CocoaError where error.code == .fileNoSuchFile {}
            try removeOrphans(keeping: nil)
        }
    }

    private static var writeOptions: Data.WritingOptions {
        #if os(iOS)
        return [.atomic, .completeFileProtection]
        #else
        return .atomic
        #endif
    }

    private func prepareDirectory() throws {
        var url = directory
        var attributes: [FileAttributeKey: Any] = [.posixPermissions: 0o700]
        #if os(iOS)
        attributes[.protectionKey] = FileProtectionType.complete
        #endif
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true, attributes: attributes)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try url.setResourceValues(values)
    }

    private func file(_ id: UUID, _ ext: String) -> URL {
        directory.appendingPathComponent(id.uuidString).appendingPathExtension(ext)
    }

    private func readManifest() throws -> Manifest? {
        let data: Data
        do { data = try Data(contentsOf: manifestURL) }
        catch let error as CocoaError where error.code == .fileReadNoSuchFile { return nil }
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        guard manifest.version == 1 else { throw CocoaError(.fileReadUnknown) }
        return manifest
    }

    private func writeCheckpoint(_ checkpoint: DocumentCheckpoint) throws {
        try JSONEncoder().encode(checkpoint).write(to: file(checkpoint.id, "json"), options: Self.writeOptions)
    }

    private func removeOrphans(keeping id: UUID?) throws {
        let entries: [URL]
        do { entries = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) }
        catch let error as CocoaError where error.code == .fileReadNoSuchFile { return }
        for entry in entries where entry.pathExtension == "photo" || entry.pathExtension == "json" {
            if let fileID = UUID(uuidString: entry.deletingPathExtension().lastPathComponent), fileID != id {
                try FileManager.default.removeItem(at: entry)
            }
        }
    }
}
