# MLX edge fallback

This C++ client switches from a cloud chat endpoint to a 3B model on Apple Silicon when the endpoint returns HTTP 503. The MLX Metal backend runs attention and RMSNorm with a KV cache per layer. It loads Llama 3.2 Instruct and Qwen2.5 Instruct from safetensors weights in dense or MLX affine 4-bit format.

## Run with a fixture

The fixture injects HTTP 503 and returns labeled text when no model directory is supplied. It requires CMake 3.27+, C++20, and libcurl. CMake fetches the JSON dependency if needed.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/edge_fallback --demo
```

## Run a local 3B model

Install the MLX C++ library with Metal and safetensors support. The prebuilt MLX Metal package includes the library and CMake configuration. Clone tokenizers-cpp recursively and put Cargo on your PATH before building. [MLX installation](https://ml-explore.github.io/mlx/build/html/install.html) · [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)

The model directory needs `config.json`, `tokenizer.json`, and all `model*.safetensors` shards. [Llama-3.2-3B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit) · [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

```sh
cmake -S . -B build-mlx -DCMAKE_BUILD_TYPE=Release \
  -DEDGE_ENABLE_MLX=ON -DMLX_DIR=/path/to/mlx/share/cmake/MLX \
  -DTOKENIZERS_CPP_SOURCE=/path/to/tokenizers-cpp
cmake --build build-mlx
./build-mlx/edge_fallback --offline --model /path/to/Llama-3.2-3B-Instruct-4bit \
  --prompt 'Explain KV caching in one sentence.' --max-tokens 48
```

For cloud-first requests, set `CLOUD_BASE_URL` through `/v1`, set `CLOUD_MODEL` to the served model ID, and set `CLOUD_API_KEY` if required. Run the same command without `--offline`. Use `--demo --model DIR` to inject a 503 and run the local model without a cloud call. The JSON response identifies which source generated the text.

In a local run on Apple Silicon with MLX 0.32.5, the loopback server returned HTTP 503. The 4-bit Llama 3.2 3B checkpoint then generated `4` for a short arithmetic prompt on Metal.

References: [MLX C++ build guide](https://github.com/ml-explore/mlx/blob/main/docs/src/dev/mlx_in_cpp.rst) and [Qwen2.5 architecture configuration](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json).
