#include "mlx_model.hpp"
#include "decode_kv.hpp"

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
#include <span>
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

std::size_t validate_prefix_capacity(const Config& config, std::size_t capacity) {
    constexpr std::size_t max_bytes = 256 * 1024 * 1024;
    // Reserve for the largest supported K/V element size, regardless of model weight packing.
    const auto token_budget = max_bytes / 2 / sizeof(float) / config.layers
        / config.kv_heads / (config.hidden / config.heads);
    if (capacity > max_prefix_cache_tokens || capacity > token_budget
        || capacity >= static_cast<std::size_t>(config.max_context)) {
        throw std::runtime_error("Prefix cache exceeds the 256 MiB or model context limit.");
    }
    return capacity;
}

struct Cache {
    mx::array keys;
    mx::array values;
};

using LayerCaches = std::vector<std::optional<Cache>>;

struct Prefix {
    std::vector<std::int32_t> tokens;
    LayerCaches layers;
    std::size_t bytes;
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
    LayerCaches cache_;
    const std::optional<mx::array>& rope_periods_;
    int offset_ = 0;
    int decode_capacity_ = 0;

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

    mx::array embeddings(std::span<const std::int32_t> tokens) const {
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
        if (decode_capacity_ > 0) {
            auto& layer_cache = *cache_[layer];
            keys = edge::append_decode_kv(layer_cache.keys, keys, offset_, decode_capacity_);
            values = edge::append_decode_kv(layer_cache.values, values, offset_, decode_capacity_);
        } else {
            if (cache_[layer]) {
                keys = mx::concatenate({cache_[layer]->keys, keys}, 2);
                values = mx::concatenate({cache_[layer]->values, values}, 2);
            }
            cache_[layer] = Cache{keys, values};
        }
        // MLX aligns the last query with the last key for cached multi-token suffixes.
        auto attention = mx::fast::scaled_dot_product_attention(
            queries, keys, values, 1.0f / std::sqrt(static_cast<float>(head_size)), -1,
            steps > 1 ? "causal" : "");
        return linear(mx::reshape(mx::transpose(attention, {0, 2, 1, 3}),
                                  {steps, config_.hidden}), base + ".o_proj");
    }

    mx::array forward(std::span<const std::int32_t> tokens) {
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
        return hidden;
    }

public:
    Decoder(const Config& config, const Weights& weights, const std::optional<mx::array>& periods)
        : config_(config), weights_(weights), cache_(config.layers), rope_periods_(periods) {}

    Prefix prefill_prefix(std::span<const std::int32_t> tokens) {
        (void)forward(tokens);
        std::size_t bytes = 0;
        for (const auto& layer : cache_) {
            bytes += layer->keys.nbytes() + layer->values.nbytes();
        }
        return {std::vector<std::int32_t>(tokens.begin(), tokens.end()), cache_, bytes};
    }

    void restore_prefix(const Prefix& prefix) {
        cache_ = prefix.layers;
        offset_ = static_cast<int>(prefix.tokens.size());
    }

    void reserve_decode(int capacity) { decode_capacity_ = capacity; }

    std::int32_t next_token(std::span<const std::int32_t> tokens) {
        const auto hidden = forward(tokens);
        const auto last = mx::take(hidden, static_cast<int>(tokens.size()) - 1, 0);
        const auto final = mx::reshape(mx::fast::rms_norm(last, weight("model.norm.weight"), config_.epsilon),
                                       {1, config_.hidden});
        const auto logits = linear(final, config_.tied_embedding ? "model.embed_tokens" : "lm_head");
        return static_cast<std::int32_t>(mx::argmax(logits).item<std::uint32_t>());
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
    const std::size_t prefix_capacity;
    Weights weights;
    std::optional<mx::array> periods;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    std::vector<int> stop_ids;
    std::optional<Prefix> prefix;

    Impl(const std::filesystem::path& directory, std::size_t prefix_tokens)
        : config_json(Json::parse(read_file(directory / "config.json"))),
          config(config_json), prefix_capacity(validate_prefix_capacity(config, prefix_tokens)),
          weights(load_weights(directory)),
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

LocalModel::LocalModel(const std::filesystem::path& directory, std::size_t prefix_tokens) {
    if (!mx::is_available(mx::Device::gpu)) {
        throw std::runtime_error("MLX Metal GPU is unavailable.");
    }
    mx::set_default_device(mx::Device::gpu);
    impl_ = std::make_unique<Impl>(directory, prefix_tokens);
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
    std::span<const std::int32_t> suffix(prompt_tokens);
    std::optional<Prefix> replacement;
    std::size_t reused_tokens = 0;
    if (model.prefix_capacity > 0 && suffix.size() > model.prefix_capacity) {
        const auto leading = suffix.first(model.prefix_capacity);
        if (model.prefix && std::ranges::equal(model.prefix->tokens, leading)) {
            decoder.restore_prefix(*model.prefix);
            reused_tokens = leading.size();
        } else {
            // Drop the prior anchor before allocating a different one.
            model.prefix.reset();
            replacement = decoder.prefill_prefix(leading);
        }
        suffix = suffix.subspan(leading.size());
    }

    std::vector<std::int32_t> generated;
    generated.reserve(max_tokens);
    auto next = decoder.next_token(suffix);
    const auto first_decision = Clock::now();
    decoder.reserve_decode(static_cast<int>(prompt_tokens.size()) + max_tokens - 1);
    auto decision = first_decision;
    auto last_output = first_decision;
    while (std::find(model.stop_ids.begin(), model.stop_ids.end(), next) == model.stop_ids.end()) {
        generated.push_back(next);
        last_output = decision;
        if (generated.size() >= static_cast<std::size_t>(max_tokens)) {
            break;
        }
        next = decoder.next_token(std::span(&next, 1));
        decision = Clock::now();
    }
    const bool stopped_on_token = generated.size() < static_cast<std::size_t>(max_tokens);
    auto text = model.tokenizer->Decode(generated);
    // next_token synchronized the suffix and the prefix K/V it consumed. Cache
    // only the prefix, whose handles were saved before any suffix was appended.
    if (replacement) model.prefix = std::move(*replacement);
    const auto milliseconds = [](auto duration) {
        return std::chrono::duration<double, std::milli>(duration).count();
    };
    const double prefill_ms = milliseconds(first_decision - started);
    const auto emitted = generated.size();
    std::optional<double> ttft_ms;
    std::optional<double> decode_ms;
    std::optional<double> output_rate;
    std::optional<double> stop_ms;
    if (emitted > 0) {
        ttft_ms = prefill_ms;
        if (stopped_on_token) stop_ms = milliseconds(decision - last_output);
    }
    if (emitted > 1) {
        decode_ms = milliseconds(last_output - first_decision);
        if (*decode_ms > 0) output_rate = (emitted - 1) * 1000.0 / *decode_ms;
    }
    return {std::move(text), prompt_tokens.size(), std::move(generated), reused_tokens,
            model.prefix ? model.prefix->tokens.size() : 0,
            model.prefix ? model.prefix->bytes : 0,
            prefill_ms, milliseconds(Clock::now() - started),
            ttft_ms, decode_ms, output_rate, stop_ms, stopped_on_token};
}
