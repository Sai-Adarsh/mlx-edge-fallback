#include "mlx_model.hpp"

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

using Json = nlohmann::json;

namespace {
struct Reply {
    long status;
    std::string body;
};

std::string environment(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "";
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

Reply call_cloud(std::string base, const std::string& model, const std::string& prompt, int max_tokens) {
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
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers.get());
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(payload.size()));
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 90L);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, append_response);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &reply.body);
    const auto result = curl_easy_perform(curl.get());
    if (result != CURLE_OK) {
        throw std::runtime_error(std::string("Cloud request failed: ") + curl_easy_strerror(result));
    }
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &reply.status);
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
    bool demo = false;
    bool offline = false;
};

Arguments arguments(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--demo") {
            args.demo = true;
        } else if (arg == "--offline") {
            args.offline = true;
        } else if (arg == "--prompt" || arg == "--model" || arg == "--max-tokens") {
            if (++i >= argc) {
                throw std::runtime_error("Missing value for " + std::string(arg));
            }
            const std::string value = argv[i];
            if (arg == "--prompt") {
                args.prompt = value;
            } else if (arg == "--model") {
                args.model_directory = value;
            } else {
                std::size_t consumed = 0;
                args.max_tokens = std::stoi(value, &consumed);
                if (consumed != value.size() || args.max_tokens < 1 || args.max_tokens > 1024) {
                    throw std::runtime_error("--max-tokens must be an integer from 1 to 1024.");
                }
            }
        } else {
            throw std::runtime_error("Unknown argument: " + std::string(arg));
        }
    }
    if (args.demo && args.prompt.empty()) {
        args.prompt = "Explain why local inference helps during a cloud outage.";
    }
    if (args.prompt.empty()) {
        throw std::runtime_error("Supply --prompt TEXT or --demo.");
    }
    if ((args.offline && !args.demo) && args.model_directory.empty()) {
        throw std::runtime_error("--offline requires --model DIR.");
    }
    return args;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
            std::cout << "edge_fallback [--demo | --offline] --prompt TEXT --model DIR [--max-tokens N]\n";
            return 0;
        }
        const auto args = arguments(argc, argv);
        const auto reply = args.demo || args.offline
            ? Reply{503, {}}
            : call_cloud(environment("CLOUD_BASE_URL"), environment("CLOUD_MODEL"), args.prompt, args.max_tokens);
        if (reply.status != 503) {
            std::cout << Json{{"source", "cloud"}, {"text", completion(reply)}}.dump() << '\n';
            return 0;
        }
        if (args.demo && args.model_directory.empty()) {
            std::cout << Json{
                {"source", "fixture"},
                {"cloud_status", 503},
                {"text", "Local inference keeps serving requests when the cloud is unavailable."}
            }.dump() << '\n';
            return 0;
        }
        if (args.model_directory.empty()) {
            throw std::runtime_error("Cloud returned HTTP 503. Pass --model DIR for local inference.");
        }
#ifdef EDGE_HAS_MLX
        std::cout << Json{
            {"source", "mlx"},
            {"cloud_status", args.offline ? Json(nullptr) : Json(503)},
            {"text", generate_local(args.model_directory, args.prompt, args.max_tokens)}
        }.dump() << '\n';
        return 0;
#else
        throw std::runtime_error("Build with EDGE_ENABLE_MLX=ON to use a local model.");
#endif
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
