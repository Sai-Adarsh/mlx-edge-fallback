#include <mlx/mlx.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace fs = std::filesystem;
namespace mx = mlx::core;
using Json = nlohmann::json;

void write_mlx_fixture(const fs::path& directory, bool include_head) {
    fs::create_directory(directory);
    // Tiny dense Qwen2-compatible checkpoint. Zero attention and MLP residuals
    // leave unit embeddings unchanged, so the biased head always selects "yes".
    std::ofstream(directory / "config.json") << Json{
        {"model_type", "qwen2"}, {"hidden_size", 8}, {"num_attention_heads", 2},
        {"num_key_value_heads", 1}, {"num_hidden_layers", 1}, {"vocab_size", 4},
        {"max_position_embeddings", 16}, {"rms_norm_eps", 1e-5}, {"rope_theta", 10000.0},
        {"tie_word_embeddings", false}, {"eos_token_id", 3}
    }.dump();
    std::ofstream(directory / "tokenizer.json") << R"({
        "version":"1.0",
        "truncation":null,
        "padding":null,
        "added_tokens":[],
        "normalizer":null,
        "pre_tokenizer":{"type":"WhitespaceSplit"},
        "post_processor":null,
        "decoder":null,
        "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"yes":2,"[EOS]":3},"unk_token":"[UNK]"}
    })";
    const std::string layer = "model.layers.0.";
    std::unordered_map<std::string, mx::array> tensors{
        {"model.embed_tokens.weight", mx::ones({4, 8})},
        {"model.norm.weight", mx::ones({8})},
        {layer + "input_layernorm.weight", mx::ones({8})},
        {layer + "post_attention_layernorm.weight", mx::ones({8})},
        {layer + "self_attn.q_proj.weight", mx::zeros({8, 8})},
        {layer + "self_attn.q_proj.bias", mx::zeros({8})},
        {layer + "self_attn.k_proj.weight", mx::zeros({4, 8})},
        {layer + "self_attn.v_proj.weight", mx::zeros({4, 8})},
        {layer + "self_attn.o_proj.weight", mx::zeros({8, 8})},
        {layer + "mlp.gate_proj.weight", mx::zeros({16, 8})},
        {layer + "mlp.up_proj.weight", mx::zeros({16, 8})},
        {layer + "mlp.down_proj.weight", mx::zeros({8, 16})},
        {"lm_head.weight", mx::zeros({4, 8})},
        {"lm_head.bias", mx::array({0.f, 0.f, 2.f, 0.f})},
    };
    if (!include_head) tensors.erase("lm_head.weight");
    mx::save_safetensors((directory / "model.safetensors").string(), tensors);
}
