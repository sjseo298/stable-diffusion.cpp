#include "dgl/ops/chat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>

#include "conditioning/conditioner.hpp"
#include "core/backend_fit.h"
#include "core/file_access.h"
#include "core/ggml_extend_backend.h"
#include "core/tensor.hpp"
#include "core/util.h"
#include "dgl/context.h"
#include "dgl/models/text_encoder.h"
#include "dgl/registry.h"
#include "model/te/llm.hpp"

namespace sd::dgl::ops {

    namespace {
        // Qwen ChatML uses <|im_end|> (151645) as the turn terminator; the tokenizer's
        // EOS/PAD id (<|endoftext|>, 151643) also ends a generation.
        constexpr int kImEndId = 151645;

        struct ChatRunGuard {
            Context& ctx;
            std::unique_lock<std::mutex> lock;

            explicit ChatRunGuard(Context& c) : ctx(c), lock(c.mutex) {
                if (ctx.active_run || ctx.generating)
                    throw std::runtime_error("DGL context is busy");
                ctx.generating = true;
                lock.unlock();
            }
            ~ChatRunGuard() {
                std::lock_guard<std::mutex> guard(ctx.mutex);
                ctx.generating = false;
            }
        };

        std::shared_ptr<models::TextEncoder> resolve_text_encoder(Context& ctx, const ChatRequest& request) {
            if (!request.text_encoder_path.empty()) {
                const auto& registry = operations();
                const auto it        = registry.find("load_text_encoder");
                if (it == registry.end())
                    throw std::runtime_error("load_text_encoder operation is unavailable");
                Values args;
                for (const auto& argument : it->second.arguments)
                    if (argument.default_value)
                        args.emplace(argument.name, *argument.default_value);
                args["path"]   = Value::literal(request.text_encoder_path);
                args["family"] = Value::literal(request.text_encoder_family);
                if (!request.tokenizer_path.empty())
                    args["tokenizer"] = Value::literal(request.tokenizer_path);
                // Match the generation resource key: RunPlan::load_resource resolves an empty
                // backend to the auto-fit default before caching, so chat must do the same to
                // reuse (not duplicate) the encoder loaded by image generation.
                auto backend = args.find("backend");
                if (backend != args.end() && trim(backend->second.scalar.get<std::string>()).empty()) {
                    SDBackendManager backends(ctx.backends);
                    const std::string resolved =
                        ctx.auto_fit() ? sd::backend_fit::select_default_backend(ctx.models->max_vram(), [&](auto device, auto free, auto total) {
                            return ctx.models->available_vram_for_fit(backends.existing_backend(device), free, total);
                        })
                                       : sd_backend_resolve_name("");
                    backend->second = Value::literal(resolved);
                }
                Value value = ctx.load(*it, args);
                return value.as<models::TextEncoder>();
            }
            std::lock_guard<std::mutex> guard(ctx.mutex);
            for (auto& [key, resource] : ctx.resources) {
                auto cached = resource.value.lock();
                if (!cached || cached->type().kind != Kind::TextEncoder)
                    continue;
                auto encoder = cached->as<models::TextEncoder>();
                if (std::dynamic_pointer_cast<LLMEmbedder>(encoder->cond_stage_model))
                    return encoder;
            }
            throw std::runtime_error("no LLM text encoder is loaded; set ChatRequest::text_encoder_path");
        }

        std::string format_chatml(const std::vector<ChatMessage>& messages) {
            std::string prompt;
            for (const auto& message : messages) {
                if (message.role != "system" && message.role != "user" && message.role != "assistant")
                    throw std::runtime_error("unsupported chat role '" + message.role + "'");
                prompt += "<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>\n";
            }
            prompt += "<|im_start|>assistant\n";
            return prompt;
        }

        // Inverse of the GPT-2 byte<->unicode table used by byte-level BPE tokenizers.
        const std::map<char32_t, unsigned char>& byte_inverse() {
            static const std::map<char32_t, unsigned char> inverse = [] {
                std::set<int> direct;
                for (int b = '!'; b <= '~'; ++b)
                    direct.insert(b);
                for (int b = 161; b <= 172; ++b)
                    direct.insert(b);
                for (int b = 174; b <= 255; ++b)
                    direct.insert(b);
                std::map<char32_t, unsigned char> table;
                for (int b : direct)
                    table[static_cast<char32_t>(b)] = static_cast<unsigned char>(b);
                int n = 0;
                for (int b = 0; b < 256; ++b)
                    if (!direct.count(b))
                        table[static_cast<char32_t>(256 + n++)] = static_cast<unsigned char>(b);
                return table;
            }();
            return inverse;
        }

        // Tokenizer::decode yields the byte-mapped BPE representation; map it back to raw bytes.
        std::string decode_generated(const Tokenizer& tokenizer, const std::vector<int>& tokens) {
            std::string mapped;
            tokenizer.decode(tokens, mapped);
            const auto& inverse = byte_inverse();
            std::string bytes;
            for (char32_t cp : utf8_to_utf32(mapped)) {
                const auto found = inverse.find(cp);
                if (found != inverse.end())
                    bytes.push_back(static_cast<char>(found->second));
            }
            return bytes;
        }

        // Length of the longest prefix that ends on a complete UTF-8 character boundary.
        size_t utf8_complete_len(const std::string& text) {
            size_t n = text.size();
            if (n == 0)
                return 0;
            size_t start = n;
            size_t back  = 0;
            while (start > 0 && back < 4 && (static_cast<unsigned char>(text[start - 1]) & 0xC0) == 0x80) {
                --start;
                ++back;
            }
            if (start == 0)
                return 0;
            const unsigned char lead = static_cast<unsigned char>(text[start - 1]);
            size_t width;
            if (lead < 0x80)
                width = 1;
            else if ((lead & 0xE0) == 0xC0)
                width = 2;
            else if ((lead & 0xF0) == 0xE0)
                width = 3;
            else if ((lead & 0xF8) == 0xF0)
                width = 4;
            else
                return n;
            return (start - 1 + width <= n) ? n : start - 1;
        }

        int argmax(const sd::Tensor<float>& logits) {
            const int64_t vocab = logits.shape()[0];
            const float* data   = logits.data();
            int64_t best        = 0;
            for (int64_t i = 1; i < vocab; ++i)
                if (data[i] > data[best])
                    best = i;
            return static_cast<int>(best);
        }

        struct PinnedParams {
            std::shared_ptr<ModelManager::ResourceHandle> handle;
            std::vector<ggml_tensor*> params;
            bool active = false;
            ~PinnedParams() {
                if (active)
                    handle->release_compute_backend_params(params);
            }
        };
        // Rectangular causal mask [L_k, n_q]: key k is visible to query q iff k <= n_past + q.
        sd::Tensor<float> causal_mask(int64_t L_k, int64_t n_q, int64_t n_past) {
            std::vector<float> v(static_cast<size_t>(L_k * n_q));
            for (int64_t q = 0; q < n_q; q++)
                for (int64_t k = 0; k < L_k; k++)
                    v[static_cast<size_t>(q * L_k + k)] = (k > n_past + q) ? -INFINITY : 0.f;
            return sd::Tensor<float>({L_k, n_q}, std::move(v));
        }

        // Greedy at temperature <= 0; otherwise temperature + top-p nucleus sampling.
        int sample_logits(const sd::Tensor<float>& logits, float temperature, float top_p, std::mt19937& rng) {
            const auto& vals = logits.values();
            const int64_t n  = static_cast<int64_t>(vals.size());
            if (temperature <= 0.f || n == 0)
                return static_cast<int>(std::max_element(vals.begin(), vals.end()) - vals.begin());
            std::vector<std::pair<float, int>> scored;
            scored.reserve(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; i++)
                scored.emplace_back(vals[i] / temperature, static_cast<int>(i));
            std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            const float max_logit = scored[0].first;
            double sum            = 0;
            for (const auto& s : scored)
                sum += std::exp(static_cast<double>(s.first) - max_logit);
            std::vector<double> probs;
            probs.reserve(scored.size());
            double acc  = 0;
            size_t keep = scored.size();
            for (size_t i = 0; i < scored.size(); i++) {
                const double p = std::exp(static_cast<double>(scored[i].first) - max_logit) / sum;
                probs.push_back(p);
                acc += p;
                if (top_p < 1.f && acc >= top_p) {
                    keep = i + 1;
                    break;
                }
            }
            double total = 0;
            for (size_t i = 0; i < keep; i++)
                total += probs[i];
            std::uniform_real_distribution<double> dist(0.0, total);
            const double r = dist(rng);
            double cum     = 0;
            for (size_t i = 0; i < keep; i++) {
                cum += probs[i];
                if (r <= cum)
                    return scored[i].second;
            }
            return scored[keep - 1].second;
        }
    }  // namespace

    ChatResult generate_chat(Context& ctx, const ChatRequest& request) {
        ChatResult result;
        try {
            if (request.messages.empty())
                throw std::runtime_error("chat requires at least one message");
            if (request.max_tokens <= 0)
                throw std::runtime_error("max_tokens must be positive");
            if (request.max_context_tokens <= 0)
                throw std::runtime_error("max_context_tokens must be positive");

            ChatRunGuard guard(ctx);
            sd::FileAccessScope file_access(ctx.file_access);

            auto encoder  = resolve_text_encoder(ctx, request);
            auto embedder = std::dynamic_pointer_cast<LLMEmbedder>(encoder->cond_stage_model);
            if (!embedder || !embedder->llm || !embedder->tokenizer)
                throw std::runtime_error("chat requires an LLM text encoder");
            auto& llm       = *embedder->llm;
            auto& tokenizer = *embedder->tokenizer;

            models::Model::ExecutionScope scope(*encoder);
            scope.require_ready();

            // Protect the full LLM parameter set (including the final blocks / norm / lm_head
            // that image conditioning may not tap) for the whole generation.
            std::map<std::string, ggml_tensor*> param_map;
            llm.get_param_tensors(param_map, "text_encoders.llm");
            PinnedParams pinned;
            pinned.handle = encoder->model_weights;
            pinned.params.reserve(param_map.size());
            for (auto& [name, tensor] : param_map)
                pinned.params.push_back(tensor);
            if (!encoder->model_weights->prepare_params(pinned.params))
                throw std::runtime_error("cannot prepare LLM weights for chat");
            pinned.active = true;

            const std::string prompt = format_chatml(request.messages);
            std::vector<int> prompt_tokens;
            if (!tokenizer.encode(prompt, prompt_tokens))
                throw std::runtime_error("chat prompt tokenization failed");
            result.prompt_tokens = static_cast<int>(prompt_tokens.size());
            if (result.prompt_tokens >= request.max_context_tokens)
                throw std::runtime_error("chat prompt exceeds the context limit");

            const int eos_id = tokenizer.EOS_TOKEN_ID;
            const int pad_id = tokenizer.PAD_TOKEN_ID;
            std::vector<int> generated;
            std::string decoded;
            size_t committed     = 0;
            result.finish_reason = "length";

            std::mt19937 rng(request.seed >= 0 ? static_cast<unsigned>(request.seed) : std::random_device{}());
            const bool use_kv = request.use_kv_cache;
            LLM::LLMKVCache kv;
            sd::Tensor<float> logits;
            int64_t n_past = 0;
            if (use_kv) {
                // Prefill: run the whole prompt once, cache per-layer K/V, keep last-token logits.
                const auto pos  = llm.decode_position_ids(0, prompt_tokens.size());
                const auto mask = causal_mask(prompt_tokens.size(), prompt_tokens.size(), 0);
                std::vector<int32_t> ids(prompt_tokens.begin(), prompt_tokens.end());
                logits = llm.compute_logits(encoder->n_threads,
                                            sd::Tensor<int32_t>::from_vector(ids),
                                            mask,
                                            sd::Tensor<int32_t>::from_vector(pos),
                                            true,
                                            false,
                                            &kv);
                if (logits.empty())
                    throw std::runtime_error("LLM logits computation failed");
                n_past = prompt_tokens.size();
            }

            while (static_cast<int>(generated.size()) < request.max_tokens) {
                if (request.should_cancel && request.should_cancel()) {
                    result.finish_reason = "stop";
                    break;
                }
                if (result.prompt_tokens + static_cast<int>(generated.size()) + 1 > request.max_context_tokens) {
                    result.finish_reason = "length";
                    break;
                }
                if (!use_kv) {
                    // Reference path: recompute the full sequence every step.
                    std::vector<int32_t> sequence(prompt_tokens.begin(), prompt_tokens.end());
                    sequence.insert(sequence.end(), generated.begin(), generated.end());
                    logits = llm.compute_logits(encoder->n_threads, sd::Tensor<int32_t>::from_vector(sequence));
                    if (logits.empty())
                        throw std::runtime_error("LLM logits computation failed");
                }
                const int next = sample_logits(logits, request.temperature, request.top_p, rng);
                if (next == eos_id || next == pad_id || next == kImEndId) {
                    result.finish_reason = "stop";
                    break;
                }
                generated.push_back(next);

                if (use_kv) {
                    // Decode the new token against the cached history (grow-by-concat).
                    const auto pos  = llm.decode_position_ids(n_past, 1);
                    const auto mask = causal_mask(n_past + 1, 1, n_past);
                    std::vector<int32_t> ids{next};
                    kv.n_past = n_past;
                    logits    = llm.compute_logits(encoder->n_threads,
                                                sd::Tensor<int32_t>::from_vector(ids),
                                                mask,
                                                sd::Tensor<int32_t>::from_vector(pos),
                                                true,
                                                false,
                                                &kv);
                    if (logits.empty())
                        throw std::runtime_error("LLM logits computation failed");
                    n_past += 1;
                }

                decoded               = decode_generated(tokenizer, generated);
                const size_t complete = utf8_complete_len(decoded);
                if (complete > committed) {
                    if (request.on_token)
                        request.on_token(decoded.substr(committed, complete - committed));
                    committed = complete;
                }
            }
            if (use_kv) {
                llm.clear_kv_cache();
                llm.runner_end();
            }

            result.text              = decoded.substr(0, utf8_complete_len(decoded));
            result.completion_tokens = static_cast<int>(generated.size());
        } catch (const std::exception& error) {
            result.error = error.what();
            result.finish_reason.clear();
        }
        return result;
    }

}  // namespace sd::dgl::ops
