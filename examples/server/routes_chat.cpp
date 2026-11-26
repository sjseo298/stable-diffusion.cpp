#include "routes.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include "async_jobs.h"
#include "dgl/context.h"
#include "dgl/models/component_sources.h"
#include "dgl/ops/chat.h"
#include "model_definition.h"

namespace {
    // Shared state between the SSE content provider (httplib thread) and the chat
    // worker thread. The provider joins the worker before the state is destroyed.
    struct ChatStreamState {
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<std::string> chunks;
        bool done = false;
        std::atomic<bool> cancel{false};
        std::thread worker;

        ~ChatStreamState() {
            cancel.store(true);
            cv.notify_all();
            if (worker.joinable())
                worker.join();
        }

        void push(std::string chunk) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                chunks.push_back(std::move(chunk));
            }
            cv.notify_one();
        }
    };

    json make_chat_chunk(const std::string& id, const std::string& model,
                         const json& delta, const json& finish_reason) {
        json choice;
        choice["index"]         = 0;
        choice["delta"]         = delta;
        choice["finish_reason"] = finish_reason;
        json r;
        r["id"]      = id;
        r["object"]  = "chat.completion.chunk";
        r["created"] = static_cast<long long>(std::time(nullptr));
        r["model"]   = model;
        r["choices"] = json::array({choice});
        return r;
    }

    std::string sse_data(const json& j) {
        return "data: " + j.dump() + "\n\n";
    }

    bool parse_chat_request(const httplib::Request& req, ServerRuntime& runtime,
                            sd::dgl::ops::ChatRequest& chat, std::string& model,
                            bool& stream, std::string& error) {
        if (req.body.empty()) {
            error = "empty body";
            return false;
        }
        json j;
        try {
            j = json::parse(req.body);
        } catch (const std::exception&) {
            error = "invalid JSON";
            return false;
        }
        if (!j.contains("messages") || !j["messages"].is_array() || j["messages"].empty()) {
            error = "messages must be a non-empty array";
            return false;
        }
        for (const auto& m : j["messages"]) {
            const std::string role = m.value("role", "");
            if (role != "system" && role != "user" && role != "assistant") {
                error = "unsupported message role: " + role;
                return false;
            }
            if (!m.contains("content") || !m["content"].is_string()) {
                error = "message content must be a string";
                return false;
            }
            chat.messages.push_back({role, m["content"].get<std::string>()});
        }
        chat.max_tokens  = std::max(1, j.value("max_tokens", 256));
        chat.temperature  = j.value("temperature", 0.f);
        chat.top_p        = j.value("top_p", 1.f);
        chat.use_kv_cache = j.value("use_kv_cache", true);
        stream          = j.value("stream", false);
        model           = j.value("model", "sd-cpp-local");
        // Reuse the text encoder configured at launch (--llm); when absent, the chat
        // operation falls back to an LLM encoder already cached in the context.
        if (runtime.ctx_params && !runtime.ctx_params->llm_path.empty())
            chat.text_encoder_path = runtime.ctx_params->llm_path;
        // The component loader needs the model family (definition id); resolve it from
        // the configured diffusion model exactly like image generation does.
        if (runtime.ctx_params) {
            const std::string& dit = !runtime.ctx_params->diffusion_model_path.empty()
                                         ? runtime.ctx_params->diffusion_model_path
                                         : runtime.ctx_params->model_path;
            if (!dit.empty()) {
                try {
                    const auto detected = sd::dgl::models::resolve_family(VERSION_COUNT, dit);
                    if (const auto* definition = find_model_definition(detected))
                        chat.text_encoder_family = definition->id;
                } catch (const std::exception&) {
                    // Keep "auto"; the loader reports a clear error if it cannot identify.
                }
            }
        }
        return true;
    }
}  // namespace

void register_chat_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;
    RuntimeRoutes routes(svr, rt);

    routes.post("/v1/chat/completions", [runtime](const httplib::Request& req, httplib::Response& res) {
        sd::dgl::ops::ChatRequest chat;
        std::string model, error;
        bool stream = false;
        if (!parse_chat_request(req, *runtime, chat, model, stream, error)) {
            res.status = 400;
            res.set_content(json({{"error", error}}).dump(), "application/json");
            return;
        }
        auto* ctx = sd_ctx_get_dgl_ctx(runtime->sd_ctx);
        if (!ctx) {
            res.status = 503;
            res.set_content(json({{"error", "no execution context"}}).dump(), "application/json");
            return;
        }
        const std::string id = "chatcmpl-" + std::to_string(unix_timestamp_now_ms());

        if (!stream) {
            SynchronousContextJob job(*runtime);
            std::lock_guard<std::mutex> lock(*runtime->sd_ctx_mutex);
            const auto result = sd::dgl::ops::generate_chat(*ctx, chat);
            if (!result.error.empty()) {
                res.status = 500;
                res.set_content(json({{"error", result.error}}).dump(), "application/json");
                return;
            }
            json choice;
            choice["index"]         = 0;
            choice["message"]       = {{"role", "assistant"}, {"content", result.text}};
            choice["finish_reason"] = result.finish_reason;
            json out;
            out["id"]      = id;
            out["object"]  = "chat.completion";
            out["created"] = static_cast<long long>(std::time(nullptr));
            out["model"]   = model;
            out["choices"] = json::array({choice});
            out["usage"]   = {{"prompt_tokens", result.prompt_tokens},
                              {"completion_tokens", result.completion_tokens},
                              {"total_tokens", result.prompt_tokens + result.completion_tokens}};
            res.set_content(out.dump(), "application/json");
            return;
        }

        // Streaming SSE: a worker thread runs the generation while the httplib
        // content provider drains decoded text chunks to the client.
        auto state = std::make_shared<ChatStreamState>();
        chat.on_token = [state, id, model](const std::string& text) {
            if (text.empty())
                return;
            state->push(sse_data(make_chat_chunk(id, model, json{{"content", text}}, json(nullptr))));
        };
        chat.should_cancel = [state]() { return state->cancel.load(); };
        state->push(sse_data(make_chat_chunk(id, model, json{{"role", "assistant"}}, json(nullptr))));

        ChatStreamState* raw = state.get();
        state->worker = std::thread([runtime, ctx, chat, raw, id, model]() {
            SynchronousContextJob job(*runtime);
            std::lock_guard<std::mutex> lock(*runtime->sd_ctx_mutex);
            const auto result = sd::dgl::ops::generate_chat(*ctx, chat);
            const std::string finish = result.error.empty() ? result.finish_reason : "stop";
            raw->push(sse_data(make_chat_chunk(id, model, json::object(), json(finish))));
            if (!result.error.empty())
                raw->push(sse_data(json({{"error", result.error}})));
            raw->push("data: [DONE]\n\n");
            {
                std::lock_guard<std::mutex> lock(raw->mutex);
                raw->done = true;
            }
            raw->cv.notify_all();
        });

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream",
                                         [state](size_t, httplib::DataSink& sink) -> bool {
                                             std::unique_lock<std::mutex> lock(state->mutex);
                                             for (;;) {
                                                 if (!state->chunks.empty()) {
                                                     const std::string chunk = std::move(state->chunks.front());
                                                     state->chunks.pop_front();
                                                     lock.unlock();
                                                     const bool ok = sink.write(chunk.data(), chunk.size());
                                                     if (!ok) {
                                                         state->cancel.store(true);
                                                         state->cv.notify_all();
                                                         if (state->worker.joinable())
                                                             state->worker.join();
                                                         return false;
                                                     }
                                                     lock.lock();
                                                     continue;
                                                 }
                                                 if (state->done) {
                                                     lock.unlock();
                                                     if (state->worker.joinable())
                                                         state->worker.join();
                                                     return false;
                                                 }
                                                 state->cv.wait_for(lock, std::chrono::milliseconds(200));
                                                 if (!sink.is_writable()) {
                                                     state->cancel.store(true);
                                                     state->cv.notify_all();
                                                     lock.unlock();
                                                     if (state->worker.joinable())
                                                         state->worker.join();
                                                     return false;
                                                 }
                                             }
                                         });
    });
}
