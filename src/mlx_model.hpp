#pragma once

#include <filesystem>
#include <string>

std::string generate_local(
    const std::filesystem::path& model_directory,
    const std::string& prompt,
    int max_tokens);
