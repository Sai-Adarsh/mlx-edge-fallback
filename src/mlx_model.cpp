#include "mlx_model.hpp"

#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <tokenizers_cpp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mx = mlx::core;
using Json = nlohmann::json;

namespace {
std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

using Weights = std::unordered_map<std::string, mx::array>;

Weights load_weights(const std::filesystem::path& directory) {
    Weights weights;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        if (!entry.is_regular_file() || !name.starts_with("model") || entry.path().extension() != ".safetensors") {
            continue;
        }
        auto [shard, metadata] = mx::load_safetensors(entry.path().string());
        weights.merge(shard);
        if (!shard.empty()) {
            throw std::runtime_error("Duplicate tensors across safetensors shards.");
        }
    }
    if (weights.empty()) {
        throw std::runtime_error("No model*.safetensors weights found in " + directory.string());
    }
    return weights;
}

struct Config {
    int hidden;
    int heads;
    int kv_heads;
    int layers;
    int vocabulary;
    int max_context;
    float epsilon;
    float rope_base;
    bool tied_embedding;
    std::optional<int> group_size;
    std::optional<int> bits;

    explicit Config(const Json& config)
        : hidden(config.at("hidden_size")),
          heads(config.at("num_attention_heads")),
          kv_heads(config.at("num_key_value_heads")),
          layers(config.at("num_hidden_layers")),
          vocabulary(config.at("vocab_size")),
          max_context(config.at("max_position_embeddings")),
          epsilon(config.at("rms_norm_eps")),
          rope_base(config.at("rope_theta")),
          tied_embedding(config.value("tie_word_embeddings", false)) {
        if ((config.at("model_type") != "qwen2" && config.at("model_type") != "llama")
            || config.value("use_sliding_window", false)
            || hidden <= 0 || heads <= 0 || kv_heads <= 0 || layers <= 0 || vocabulary <= 0 || max_context <= 0
            || hidden % heads != 0 || heads % kv_heads != 0 || (hidden / heads) % 2 != 0
            || config.value("head_dim", hidden / heads) != hidden / heads) {
            throw std::runtime_error("Expected a supported Qwen2 or Llama checkpoint with full attention.");
        }
        if (config.contains("quantization") && !config["quantization"].is_null()) {
            const auto& quantization = config.at("quantization");
            group_size = quantization.at("group_size").get<int>();
            bits = quantization.at("bits").get<int>();
            if (quantization.value("mode", std::string("affine")) != "affine" || *bits != 4) {
                throw std::runtime_error("This backend supports dense weights and MLX affine 4-bit weights.");
            }
        }
    }
};

struct Cache {
    mx::array keys;
    mx::array values;
};

std::optional<mx::array> rope_periods(const Json& config, int head_size, float base) {
    if (!config.contains("rope_scaling") || config["rope_scaling"].is_null()) {
        return std::nullopt;
    }
    const auto& scaling = config.at("rope_scaling");
    if (scaling.at("rope_type") != "llama3") {
        throw std::runtime_error("Unsupported RoPE scaling configuration.");
    }
    const float factor = scaling.at("factor");
    const float high = scaling.at("high_freq_factor");
    const float low = scaling.at("low_freq_factor");
    const float context = scaling.at("original_max_position_embeddings");
    if (!(factor >= 1 && high > low && low > 0 && context > 0)) {
        throw std::runtime_error("Invalid Llama RoPE scaling configuration.");
    }
    std::vector<float> periods;
    for (int i = 0; i < head_size; i += 2) {
        const float period = std::pow(base, static_cast<float>(i) / head_size);
        const float wavelength = 2 * std::numbers::pi_v<float> * period;
        if (wavelength > context / low) {
            periods.push_back(period * factor);
        } else if (wavelength >= context / high) {
            const float smooth = (context / wavelength - low) / (high - low);
            periods.push_back(period / ((1 - smooth) / factor + smooth));
        } else {
            periods.push_back(period);
        }
    }
    return mx::array(periods.begin(), {head_size / 2});
}

class Decoder {
    const Config& config_;
    const Weights& weights_;
    std::vector<std::optional<Cache>> cache_;
    const std::optional<mx::array>& rope_periods_;
    int offset_ = 0;

    const mx::array& weight(const std::string& name) const {
        const auto item = weights_.find(name);
        if (item == weights_.end()) {
            throw std::runtime_error("Missing model tensor " + name);
        }
        return item->second;
    }

    mx::array linear(const mx::array& input, const std::string& name) const {
        const auto& matrix = weight(name + ".weight");
        const auto scales = weights_.find(name + ".scales");
        auto output = scales == weights_.end()
            ? mx::matmul(input, mx::transpose(matrix))
            : mx::quantized_matmul(input, matrix, scales->second, weight(name + ".biases"),
                                   true, config_.group_size, config_.bits);
        if (const auto bias = weights_.find(name + ".bias"); bias != weights_.end()) {
            output = output + bias->second;
        }
        return output;
    }

    mx::array embeddings(const std::vector<std::int32_t>& tokens) const {
        const mx::array ids(tokens.begin(), {static_cast<int>(tokens.size())});
        const std::string name = "model.embed_tokens";
        auto rows = mx::take(weight(name + ".weight"), ids, 0);
        if (weights_.contains(name + ".scales")) {
            rows = mx::dequantize(rows, mx::take(weight(name + ".scales"), ids, 0),
                                  mx::take(weight(name + ".biases"), ids, 0),
                                  config_.group_size, config_.bits);
        }
        return rows;
    }

    mx::array attention(const mx::array& input, const std::string& base, int layer, int steps) {
        const auto head_size = config_.hidden / config_.heads;
        const auto split_heads = [&](const std::string& name, int heads) {
            return mx::transpose(mx::reshape(linear(input, base + name), {1, steps, heads, head_size}),
                                 {0, 2, 1, 3});
        };
        const auto rope = [&](const mx::array& x) {
            const auto base = rope_periods_ ? std::nullopt : std::optional<float>(config_.rope_base);
            return mx::fast::rope(x, head_size, false, base, 1.0f, offset_, rope_periods_);
        };
        auto queries = rope(split_heads(".q_proj", config_.heads));
        auto keys = rope(split_heads(".k_proj", config_.kv_heads));
        auto values = split_heads(".v_proj", config_.kv_heads);
        if (cache_[layer]) {
            keys = mx::concatenate({cache_[layer]->keys, keys}, 2);
            values = mx::concatenate({cache_[layer]->values, values}, 2);
        }
        cache_[layer] = Cache{keys, values};
        auto attention = mx::fast::scaled_dot_product_attention(
            queries, keys, values, 1.0f / std::sqrt(static_cast<float>(head_size)), -1,
            steps > 1 ? "causal" : "");
        return linear(mx::reshape(mx::transpose(attention, {0, 2, 1, 3}),
                                  {steps, config_.hidden}), base + ".o_proj");
    }

public:
    Decoder(const Config& config, const Weights& weights, const std::optional<mx::array>& periods)
        : config_(config), weights_(weights), cache_(config.layers), rope_periods_(periods) {}

    int next_token(const std::vector<std::int32_t>& tokens) {
        const auto steps = static_cast<int>(tokens.size());
        for (auto token : tokens) {
            if (token < 0 || token >= config_.vocabulary) {
                throw std::runtime_error("Tokenizer and model vocabulary differ.");
            }
        }
        auto hidden = embeddings(tokens);
        for (int layer = 0; layer < config_.layers; ++layer) {
            const auto name = "model.layers." + std::to_string(layer);
            const auto norm = mx::fast::rms_norm(hidden, weight(name + ".input_layernorm.weight"), config_.epsilon);
            hidden = hidden + attention(norm, name + ".self_attn", layer, steps);
            const auto mlp_input = mx::fast::rms_norm(hidden, weight(name + ".post_attention_layernorm.weight"),
                                                     config_.epsilon);
            const auto gate = linear(mlp_input, name + ".mlp.gate_proj");
            const auto activated = gate * mx::sigmoid(gate);
            hidden = hidden + linear(activated * linear(mlp_input, name + ".mlp.up_proj"), name + ".mlp.down_proj");
        }
        offset_ += steps;
        const auto last = mx::take(hidden, steps - 1, 0);
        const auto final = mx::reshape(mx::fast::rms_norm(last, weight("model.norm.weight"), config_.epsilon),
                                       {1, config_.hidden});
        const auto logits = linear(final, config_.tied_embedding ? "model.embed_tokens" : "lm_head");
        return static_cast<int>(mx::argmax(logits).item<std::uint32_t>());
    }
};

std::string chat_prompt(const std::string& prompt, const Json& config) {
    if (prompt.find("<|") != std::string::npos) {
        throw std::runtime_error("Prompt contains a reserved chat token prefix.");
    }
    if (config.at("model_type") == "llama") {
        return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + prompt
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    }
    return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
}
} // namespace

struct LocalModel::Impl {
    Json config_json;
    Config config;
    Weights weights;
    std::optional<mx::array> periods;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    std::vector<int> stop_ids;

    explicit Impl(const std::filesystem::path& directory)
        : config_json(Json::parse(read_file(directory / "config.json"))),
          config(config_json), weights(load_weights(directory)),
          periods(rope_periods(config_json, config.hidden / config.heads, config.rope_base)),
          tokenizer(tokenizers::Tokenizer::FromBlobJSON(read_file(directory / "tokenizer.json"))) {
        if (!tokenizer) {
            throw std::runtime_error("Cannot load the model tokenizer.");
        }
        for (const auto& token : config_json.at("eos_token_id")) {
            stop_ids.push_back(token.get<int>());
        }
        for (const auto* special : {"<|endoftext|>", "<|im_start|>", "<|start_header_id|>"}) {
            const int id = tokenizer->TokenToId(special);
            if (id >= 0) stop_ids.push_back(id);
        }
        std::vector<mx::array> arrays;
        arrays.reserve(weights.size());
        for (const auto& [name, weight] : weights) arrays.push_back(weight);
        mx::eval(arrays);
    }
};

LocalModel::LocalModel(const std::filesystem::path& directory) {
    if (!mx::is_available(mx::Device::gpu)) {
        throw std::runtime_error("MLX Metal GPU is unavailable.");
    }
    mx::set_default_device(mx::Device::gpu);
    impl_ = std::make_unique<Impl>(directory);
}

LocalModel::~LocalModel() = default;

LocalGeneration LocalModel::generate(const std::string& prompt, int max_tokens) {
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    if (max_tokens < 1) throw std::runtime_error("max_tokens must be positive.");
    auto& model = *impl_;
    const auto prompt_tokens = model.tokenizer->Encode(chat_prompt(prompt, model.config_json));
    if (prompt_tokens.empty() || prompt_tokens.size() + static_cast<std::size_t>(max_tokens) >
            static_cast<std::size_t>(model.config.max_context)) {
        throw std::runtime_error("Prompt and generation budget exceed the model context.");
    }
    Decoder decoder(model.config, model.weights, model.periods);
    std::vector<std::int32_t> generated;
    auto next = decoder.next_token(prompt_tokens);
    const auto first = Clock::now();
    while (std::find(model.stop_ids.begin(), model.stop_ids.end(), next) == model.stop_ids.end()) {
        generated.push_back(next);
        if (generated.size() >= static_cast<std::size_t>(max_tokens)) {
            break;
        }
        next = decoder.next_token({next});
    }
    auto text = model.tokenizer->Decode(generated);
    const auto milliseconds = [](auto duration) {
        return std::chrono::duration<double, std::milli>(duration).count();
    };
    return {std::move(text), prompt_tokens.size(), generated.size(),
            milliseconds(first - started), milliseconds(Clock::now() - started)};
}
