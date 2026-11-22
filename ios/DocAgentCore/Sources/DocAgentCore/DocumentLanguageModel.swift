import Foundation
import llama

/// Shared, CPU-only text model for both document review and local question answering.
final class DocumentLanguageModel {
    private static let backend: Void = {
        // Native debug output may contain input fragments. Do not log document data.
        llama_log_set({ _, _, _ in }, nil)
        mtmd_log_set({ _, _, _ in }, nil)
        llama_backend_init()
    }()
    static let weights = Bundle.module.url(forResource: "Qwen3.5-0.8B-Q4_K_M", withExtension: "gguf",
                                           subdirectory: "ReviewModel")
    static let threads = Int32(min(4, ProcessInfo.processInfo.activeProcessorCount))
    let model: OpaquePointer
    let context: OpaquePointer
    // llama retains a non-owning callback pointer until the context is freed.
    let cancellation: ReviewCancellation

    init(cancellation: ReviewCancellation) throws {
        guard let weights = Self.weights else { throw VisualReviewError.unavailable }
        _ = Self.backend
        self.cancellation = cancellation
        let flag = Unmanaged.passUnretained(cancellation).toOpaque()
        var params = llama_model_default_params()
        params.n_gpu_layers = 0
        params.load_mode = LLAMA_LOAD_MODE_MMAP
        params.progress_callback_user_data = flag
        params.progress_callback = { _, flag in
            !Unmanaged<ReviewCancellation>.fromOpaque(flag!).takeUnretainedValue().isCancelled
        }
        guard let model = llama_model_load_from_file(weights.path, params) else { throw VisualReviewError.unavailable }
        let threads = Self.threads
        var settings = llama_context_default_params()
        settings.n_ctx = 1536
        settings.n_batch = 192
        settings.n_ubatch = 192
        settings.n_outputs_max = 1
        settings.n_threads = threads
        settings.n_threads_batch = threads
        settings.offload_kqv = false
        settings.op_offload = false
        settings.abort_callback_data = flag
        settings.abort_callback = { flag in
            Unmanaged<ReviewCancellation>.fromOpaque(flag!).takeUnretainedValue().isCancelled
        }
        guard let context = llama_init_from_model(model, settings) else {
            llama_model_free(model)
            throw VisualReviewError.unavailable
        }
        self.model = model
        self.context = context
    }

    deinit {
        llama_free(context)
        llama_model_free(model)
    }

    func tokens(forLabels labels: [String]) throws -> [llama_token] {
        let vocabulary = llama_model_get_vocab(model)
        return try labels.map { label in
            var token: llama_token = 0
            guard llama_tokenize(vocabulary, label, Int32(label.utf8.count), &token, 1, false, false) == 1
                else { throw VisualReviewError.inference }
            return token
        }
    }

    func selectedLabel(_ labels: [llama_token], minimumLead: Float = 1) throws -> Int? {
        guard labels.count >= 2, let logits = llama_get_logits_ith(context, -1) else { throw VisualReviewError.inference }
        let scores = labels.map { logits[Int($0)] }
        let ranked = scores.indices.sorted { scores[$0] > scores[$1] }
        let winner = ranked[0]
        let count = Int(llama_vocab_n_tokens(llama_model_get_vocab(model)))
        let maximum = UnsafeBufferPointer(start: logits, count: count).max() ?? .infinity
        guard scores[winner] >= maximum - 0.5, scores[winner] - scores[ranked[1]] >= minimumLead else { return nil }
        return winner
    }

    /// Returns false when the actual tokenized prompt exceeds the local context.
    func evaluate(_ prompt: String, reserving outputTokens: Int = 0) throws -> Bool {
        try Task.checkCancellation()
        let vocab = llama_model_get_vocab(model)
        var tokens = [llama_token](repeating: 0, count: Int(llama_n_ctx(context)))
        let count = tokens.withUnsafeMutableBufferPointer { buffer in
            llama_tokenize(vocab, prompt, Int32(prompt.utf8.count), buffer.baseAddress, Int32(buffer.count), false, true)
        }
        guard count > 0, Int(count) + outputTokens < tokens.count else { return false }
        guard let batch = llama_batch_ext_init(context) else { throw VisualReviewError.inference }
        defer { llama_batch_ext_free(batch) }
        llama_memory_clear(llama_get_memory(context), true)
        let size = Int(llama_n_batch(context))
        for start in stride(from: 0, to: Int(count), by: size) {
            try Task.checkCancellation()
            llama_batch_ext_clear(batch)
            for i in start..<min(start + size, Int(count)) {
                let index = llama_batch_ext_add_token(batch, 0, tokens[i])
                var position = llama_pos(i)
                guard index >= 0, llama_batch_ext_set_pos(batch, index, &position),
                      llama_batch_ext_set_output_logits(batch, index, i == Int(count) - 1) else { throw VisualReviewError.inference }
            }
            let status = llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch)
            try Task.checkCancellation()
            guard status == 0 else { throw VisualReviewError.inference }
        }
        return true
    }

    /// Greedy, bounded, single-line completion. Token pieces can split a Korean UTF-8 character.
    func complete(_ prompt: String, limit: Int) throws -> String? {
        guard try evaluate(prompt, reserving: limit) else { return nil }
        guard let sampler = llama_sampler_init_greedy() else { throw VisualReviewError.inference }
        defer { llama_sampler_free(sampler) }
        guard let batch = llama_batch_ext_init(context) else { throw VisualReviewError.inference }
        defer { llama_batch_ext_free(batch) }
        let vocab = llama_model_get_vocab(model)
        var bytes: [UInt8] = []
        var position = llama_memory_seq_pos_max(llama_get_memory(context), 0) + 1
        for _ in 0..<limit {
            try Task.checkCancellation()
            let token = llama_sampler_sample(sampler, context, -1)
            if llama_vocab_is_eog(vocab, token) { break }
            var piece = [CChar](repeating: 0, count: 256)
            let count = llama_token_to_piece(vocab, token, &piece, Int32(piece.count), 0, false)
            guard count >= 0 else { return nil }
            bytes += piece.prefix(Int(count)).map { UInt8(bitPattern: $0) }
            if bytes.last == 10, bytes.contains(where: { $0 > 32 }) { break }
            llama_batch_ext_clear(batch)
            let index = llama_batch_ext_add_token(batch, 0, token)
            guard index >= 0, llama_batch_ext_set_pos(batch, index, &position),
                  llama_batch_ext_set_output_logits(batch, index, true),
                  llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch) == 0 else {
                try Task.checkCancellation()
                throw VisualReviewError.inference
            }
            position += 1
        }
        return String(bytes: bytes, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
