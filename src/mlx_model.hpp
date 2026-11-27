#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

inline constexpr int max_prefix_cache_tokens = 4096;

struct LocalGeneration {
    std::string text;
    std::size_t prompt_tokens;
    std::vector<std::int32_t> token_ids;
    std::size_t reused_prefix_tokens;
    std::size_t retained_prefix_tokens;
    std::size_t retained_prefix_bytes;
    double prefill_ms;
    double generation_ms;
    std::optional<double> ttft_ms;
    std::optional<double> decode_ms;
    std::optional<double> decode_tokens_per_second;
    std::optional<double> stop_decision_ms;
    bool stopped_on_token;
};

// Sequential runner. Each generation owns its KV cache.
class LocalModel {
public:
    explicit LocalModel(const std::filesystem::path& model_directory, std::size_t prefix_tokens = 0);
    ~LocalModel();

    LocalGeneration generate(const std::string& prompt, int max_tokens);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
