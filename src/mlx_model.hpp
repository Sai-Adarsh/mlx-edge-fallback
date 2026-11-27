#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

struct LocalGeneration {
    std::string text;
    std::size_t prompt_tokens;
    std::size_t generated_tokens;
    double prefill_ms;
    double generation_ms;
};

// Sequential runner. Each generation owns its KV cache.
class LocalModel {
public:
    explicit LocalModel(const std::filesystem::path& model_directory);
    ~LocalModel();

    LocalGeneration generate(const std::string& prompt, int max_tokens);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
