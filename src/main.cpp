#include "cloud_circuit.hpp"
#include "mlx_model.hpp"

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

using Json = nlohmann::json;

namespace {
struct Reply {
    long status;
    std::string body;
};

struct ResponseHeaders {
    CURL* curl;
    bool stopped_on_503 = false;
};

std::string environment(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "";
}

std::size_t finish_on_unavailable(char* data, std::size_t size, std::size_t count, void* target) {
    const auto bytes = size * count;
    const std::string_view line(data, bytes);
    if (line == "\r\n" || line == "\n") {
        auto& headers = *static_cast<ResponseHeaders*>(target);
        long status = 0;
        curl_easy_getinfo(headers.curl, CURLINFO_RESPONSE_CODE, &status);
        if (status == 503) {
            headers.stopped_on_503 = true;
            return 0;
        }
    }
    return bytes;
}

std::size_t append_response(char* data, std::size_t size, std::size_t count, void* target) {
    auto& response = *static_cast<std::string*>(target);
    const auto bytes = size * count;
    if (bytes > 4 * 1024 * 1024 - response.size()) {
        return 0;
    }
    try {
        response.append(data, bytes);
        return bytes;
    } catch (...) {
        return 0;
    }
}

Reply call_cloud(std::string base, const std::string& model, const std::string& prompt,
                 int max_tokens, long timeout_ms) {
    if (!(base.starts_with("https://") || base.starts_with("http://")) || model.empty()) {
        throw std::runtime_error("Set CLOUD_BASE_URL to an HTTP(S) API base and CLOUD_MODEL to its model ID.");
    }
    while (base.ends_with('/')) {
        base.pop_back();
    }
    const auto url = base + "/chat/completions";
    const auto payload = Json{
        {"model", model},
        {"messages", Json::array({{{"role", "user"}, {"content", prompt}}})},
        {"max_tokens", max_tokens},
        {"stream", false}
    }.dump();

    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), curl_easy_cleanup);
    if (!curl) {
        throw std::runtime_error("Failed to initialize libcurl.");
    }
    std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(nullptr, curl_slist_free_all);
    const auto add_header = [&](const std::string& header) {
        auto* next = curl_slist_append(headers.get(), header.c_str());
        if (!next) {
            throw std::runtime_error("Failed to allocate an HTTP header.");
        }
        headers.release();
        headers.reset(next);
    };
    add_header("Content-Type: application/json");
    if (const auto key = environment("CLOUD_API_KEY"); !key.empty()) {
        add_header("Authorization: Bearer " + key);
    }

    Reply reply{};
    ResponseHeaders response_headers{curl.get()};
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers.get());
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(payload.size()));
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS, std::min(10000L, timeout_ms));
    curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS, timeout_ms);
    curl_easy_setopt(curl.get(), CURLOPT_HEADERFUNCTION, finish_on_unavailable);
    curl_easy_setopt(curl.get(), CURLOPT_HEADERDATA, &response_headers);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, append_response);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &reply.body);
    const auto result = curl_easy_perform(curl.get());
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &reply.status);
    if (result == CURLE_WRITE_ERROR && response_headers.stopped_on_503) return reply;
    if (result != CURLE_OK) {
        throw std::runtime_error(std::string("Cloud request failed: ") + curl_easy_strerror(result));
    }
    if (reply.status == 503) {
        throw std::runtime_error("Cloud request failed: incomplete HTTP 503 headers.");
    }
    return reply;
}

std::string completion(const Reply& reply) {
    if (reply.status < 200 || reply.status >= 300) {
        throw std::runtime_error("Cloud HTTP " + std::to_string(reply.status));
    }
    return Json::parse(reply.body).at("choices").at(0).at("message").at("content").get<std::string>();
}

struct Arguments {
    std::string prompt;
    std::filesystem::path model_directory;
    int max_tokens = 64;
    int prefix_cache_tokens = 0;
    long cloud_timeout_ms = 90000;
    std::chrono::milliseconds cloud_cooldown{0};
    bool demo = false;
    bool offline = false;
    bool trace_token_ids = false;
    bool batch = false;
};

int positive_integer(std::string_view option, const std::string& value, int limit) {
    int result = 0;
    const auto end = value.data() + value.size();
    const auto [parsed, error] = std::from_chars(value.data(), end, result);
    if (error != std::errc{} || parsed != end || result < 1 || result > limit) {
        throw std::runtime_error(std::string(option) + " must be an integer from 1 to " + std::to_string(limit) + ".");
    }
    return result;
}

Arguments arguments(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--demo") {
            args.demo = true;
        } else if (arg == "--offline") {
            args.offline = true;
        } else if (arg == "--batch") {
            args.batch = true;
        } else if (arg == "--trace-token-ids") {
            args.trace_token_ids = true;
        } else if (arg == "--prompt" || arg == "--model" || arg == "--max-tokens"
                   || arg == "--cloud-timeout-ms" || arg == "--cloud-cooldown-ms" || arg == "--prefix-cache-tokens") {
            if (++i >= argc) {
                throw std::runtime_error("Missing value for " + std::string(arg));
            }
            const std::string value = argv[i];
            if (arg == "--prompt") {
                args.prompt = value;
            } else if (arg == "--model") {
                args.model_directory = value;
            } else if (arg == "--max-tokens") {
                args.max_tokens = positive_integer(arg, value, 1024);
            } else if (arg == "--cloud-timeout-ms") {
                args.cloud_timeout_ms = positive_integer(arg, value, 600000);
            } else if (arg == "--prefix-cache-tokens") {
                args.prefix_cache_tokens = positive_integer(arg, value, max_prefix_cache_tokens);
            } else {
                args.cloud_cooldown = std::chrono::milliseconds(positive_integer(arg, value, 600000));
            }
        } else {
            throw std::runtime_error("Unknown argument: " + std::string(arg));
        }
    }
    if (args.batch && !args.prompt.empty()) {
        throw std::runtime_error("Use either --batch or --prompt.");
    }
    if (!args.batch && args.demo && args.prompt.empty()) {
        args.prompt = "Explain why local inference helps during a cloud outage.";
    }
    if (!args.batch && args.prompt.empty()) {
        throw std::runtime_error("Supply --prompt TEXT, --batch, or --demo.");
    }
    if ((args.offline && !args.demo) && args.model_directory.empty()) {
        throw std::runtime_error("--offline requires --model DIR.");
    }
    if (args.cloud_cooldown.count() > 0
        && (!args.batch || args.demo || args.offline || args.model_directory.empty())) {
        throw std::runtime_error("--cloud-cooldown-ms requires cloud-first --batch with --model DIR.");
    }
    if (args.prefix_cache_tokens > 0 && (!args.batch || args.model_directory.empty())) {
        throw std::runtime_error("--prefix-cache-tokens requires --batch and --model DIR.");
    }
    return args;
}
class Session {
public:
    explicit Session(const Arguments& args) : args_(args), circuit_(args.cloud_cooldown) {}

    Json generate(const std::string& prompt, int max_tokens) {
        if (prompt.empty()) throw std::runtime_error("prompt must be a nonempty string");
        std::optional<long> cloud_status;
        std::string_view fallback = args_.offline ? "offline" : "demo";
        if (args_.demo && !args_.offline) cloud_status = 503;
        if (!args_.demo && !args_.offline) {
            if (circuit_.bypass(CloudCircuit::Clock::now())) {
                fallback = "circuit_open";
            } else {
                const auto reply = call_cloud(environment("CLOUD_BASE_URL"), environment("CLOUD_MODEL"),
                                              prompt, max_tokens, args_.cloud_timeout_ms);
                circuit_.observe(reply.status, CloudCircuit::Clock::now());
                if (reply.status != 503) {
                    return {{"source", "cloud"}, {"text", completion(reply)}};
                }
                cloud_status = reply.status;
                fallback = "cloud_503";
            }
        }
        if (args_.demo && args_.model_directory.empty()) {
            return {
                {"source", "fixture"},
                {"cloud_status", 503},
                {"text", "Local inference keeps serving requests when the cloud is unavailable."}
            };
        }
        if (args_.model_directory.empty()) {
            throw std::runtime_error("Cloud returned HTTP 503. Pass --model DIR for local inference.");
        }
#ifdef EDGE_HAS_MLX
        const bool reused = static_cast<bool>(local_);
        double load_ms = 0.0;
        if (!local_) {
            const auto start = std::chrono::steady_clock::now();
            local_ = std::make_unique<LocalModel>(args_.model_directory, args_.prefix_cache_tokens);
            load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        auto result = local_->generate(prompt, max_tokens);
        const auto metric = [](std::optional<double> value) {
            return value ? Json(*value) : Json(nullptr);
        };
        Json output{
            {"source", "mlx"},
            {"cloud_status", cloud_status ? Json(*cloud_status) : Json(nullptr)},
            {"fallback_reason", fallback},
            {"text", std::move(result.text)},
            {"local_metrics", {
                {"model_reused", reused}, {"load_ms", load_ms}, {"prefill_ms", result.prefill_ms},
                {"generation_ms", result.generation_ms}, {"prompt_tokens", result.prompt_tokens},
                {"generated_tokens", result.token_ids.size()},
                {"ttft_ms", metric(result.ttft_ms)},
                {"decode_ms", metric(result.decode_ms)},
                {"decode_tokens_per_second", metric(result.decode_tokens_per_second)},
                {"stop_decision_ms", metric(result.stop_decision_ms)},
                {"stop_reason", result.stopped_on_token ? "stop_token" : "token_limit"},
                {"reused_prefix_tokens", result.reused_prefix_tokens},
                {"retained_prefix_tokens", result.retained_prefix_tokens},
                {"retained_prefix_bytes", result.retained_prefix_bytes}
            }}
        };
        if (args_.trace_token_ids) output["generated_token_ids"] = std::move(result.token_ids);
        return output;
#else
        throw std::runtime_error("Build with EDGE_ENABLE_MLX=ON to use a local model.");
#endif
    }

private:
    const Arguments& args_;
    CloudCircuit circuit_;
#ifdef EDGE_HAS_MLX
    std::unique_ptr<LocalModel> local_;
#endif
};

struct BatchLine {
    std::string value;
    bool too_large = false;
};

std::optional<BatchLine> read_batch_line(std::istream& input) {
    constexpr std::size_t max_line = 64 * 1024;
    BatchLine line;
    for (char value; input.get(value);) {
        if (value == '\n') return line;
        if (line.value.size() < max_line) line.value.push_back(value);
        else line.too_large = true;
    }
    if (input.bad()) throw std::runtime_error("Failed to read batch input.");
    if (line.value.empty()) return std::nullopt;
    return line;
}

Json process_batch_line(const BatchLine& line, Session& session, int default_tokens) {
    if (line.too_large) throw std::runtime_error("Request line exceeds 64 KiB.");
    const auto request = Json::parse(line.value, nullptr, false);
    if (!request.is_object() || !request.contains("prompt") || !request["prompt"].is_string()) {
        throw std::runtime_error("Each line must be a JSON object with a string prompt.");
    }
    int max_tokens = default_tokens;
    if (request.contains("max_tokens")) {
        const auto& tokens = request["max_tokens"];
        if (!tokens.is_number_integer() || tokens.get<std::int64_t>() < 1 || tokens.get<std::int64_t>() > 1024) {
            throw std::runtime_error("max_tokens must be an integer from 1 to 1024.");
        }
        max_tokens = tokens.get<int>();
    }
    return session.generate(request["prompt"].get_ref<const std::string&>(), max_tokens);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
            std::cout << "edge_fallback [--demo | --offline] [--prompt TEXT | --batch] [--model DIR] [--max-tokens N] "
                         "[--cloud-timeout-ms N] [--cloud-cooldown-ms N] "
                         "[--prefix-cache-tokens N] [--trace-token-ids]\n";
            return 0;
        }
        const auto args = arguments(argc, argv);
        Session session(args);
        if (args.batch) {
            std::size_t line_number = 0;
            while (auto line = read_batch_line(std::cin)) {
                ++line_number;
                try {
                    auto output = process_batch_line(*line, session, args.max_tokens);
                    output["line"] = line_number;
                    output["ok"] = true;
                    std::cout << output.dump() << std::endl;
                } catch (const std::exception& error) {
                    std::cout << Json{{"line", line_number}, {"ok", false}, {"error", error.what()}}.dump()
                              << std::endl;
                }
            }
            return 0;
        }
        std::cout << session.generate(args.prompt, args.max_tokens).dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
