import Foundation

/// Owns automatic listening from the end-of-speech cue until its one recognition result.
/// A recording explicitly started by the user has no automatic turn.
@MainActor
public final class ConversationMicrophoneTurn {
    private enum State: Equatable {
        case idle
        case scheduled(UUID)
        case recording(UUID)
    }

    private var state = State.idle
    private var delay: Task<Void, Never>?

    public init() {}

    public func schedule(after interval: Duration, then start: @escaping @MainActor (UUID) -> Void) {
        let id = schedule()
        delay = Task { [weak self] in
            do { try await Task.sleep(for: interval) } catch { return }
            guard let self, self.beginScheduled(id) else { return }
            start(id)
        }
    }

    func schedule() -> UUID {
        cancel()
        let id = UUID()
        state = .scheduled(id)
        return id
    }

    func beginScheduled(_ id: UUID) -> Bool {
        guard state == .scheduled(id) else { return false }
        delay = nil
        state = .recording(id)
        return true
    }

    public func finishScheduled(_ id: UUID) -> Bool {
        guard state == .recording(id) else { return false }
        state = .idle
        return true
    }

    /// Returns true when the caller also needs to stop an automatically opened microphone.
    @discardableResult
    public func cancel() -> Bool {
        delay?.cancel()
        delay = nil
        let previous = state
        state = .idle
        if case .recording = previous { return true }
        return false
    }
}
