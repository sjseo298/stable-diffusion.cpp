#ifndef __SD_DGL_OPS_CHAT_H__
#define __SD_DGL_OPS_CHAT_H__

#include <functional>
#include <string>
#include <vector>

namespace sd::dgl {
    struct Context;
}

namespace sd::dgl::ops {
    struct ChatMessage {
        std::string role;  // "system", "user" or "assistant"
        std::string content;
    };

    struct ChatRequest {
        std::vector<ChatMessage> messages;
        int max_tokens         = 256;
        int max_context_tokens = 4096;

        // Text encoder to reuse. When text_encoder_path is set, the encoder is resolved
        // through the same resource cache as image generation (loaded on demand if absent).
        // When empty, an LLM text encoder already cached in the context is reused.
        std::string text_encoder_path;
        std::string text_encoder_family = "auto";
        std::string tokenizer_path;

        // Sampling: greedy when temperature <= 0; nucleus (top-p) sampling otherwise.
        float temperature = 0.f;
        float top_p       = 1.f;
        int seed          = -1;

        // Incremental K/V decode (default). False selects the full-recompute reference
        // path, used to verify greedy equivalence.
        bool use_kv_cache = true;

        // Optional per-token callback; receives newly decoded text (used by SSE in SO-3).
        std::function<void(const std::string&)> on_token;

        // Optional cancellation poll; when it returns true the decode loop stops early
        // (e.g. SSE client disconnect). finish_reason stays "stop" with partial text.
        std::function<bool()> should_cancel;
    };

    struct ChatResult {
        std::string text;
        int prompt_tokens     = 0;
        int completion_tokens = 0;
        std::string finish_reason;  // "stop" (EOS) or "length" (token/context cap)
        std::string error;          // empty on success
    };

    // Full-recompute greedy chat over the LLM text encoder held by the DGL context.
    // Weights are shared with image conditioning; no second copy is loaded.
    ChatResult generate_chat(Context& ctx, const ChatRequest& request);
}  // namespace sd::dgl::ops

#endif  // __SD_DGL_OPS_CHAT_H__
