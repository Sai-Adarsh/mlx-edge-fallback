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

## Keep the local model warm

Pass `--batch` to read independent JSON requests from stdin and write one JSON reply per line. An optional `max_tokens` overrides the CLI default. A malformed line returns a per-line error and the next request can still run. The process limits input records to 64 KiB and handles them sequentially.

```sh
printf '%s\n' '{"prompt":"What is 2 + 2? Answer with a number only.","max_tokens":5}' \
  '{"prompt":"What is 3 + 3? Answer with a number only.","max_tokens":5}' |
  ./build-mlx/edge_fallback --offline --batch --model /path/to/Llama-3.2-3B-Instruct-4bit
```

The first local request loads the tokenizer and materializes the weights. Each request has its own KV cache, so one prompt never becomes part of the next request's context. The same batch interface can call the cloud first by omitting `--offline`. Local replies report whether the weights were reused, initial load time, prefill time, total local generation time, and token counts. Prefill time includes tokenization and the first decoding decision. Total generation time excludes weight loading and the cloud request.

In a loopback HTTP 503 run on Apple Silicon, the first request returned `4` after loading the model in about 1,085 ms. An invalid input line produced an error. The next request returned `6` using the resident weights and reported zero load time.

References: [MLX C++ build guide](https://github.com/ml-explore/mlx/blob/main/docs/src/dev/mlx_in_cpp.rst) and [Qwen2.5 architecture configuration](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json).

## Back off during a cloud outage

Run cloud-first batch mode with a local model and a cooldown.

```sh
./build-mlx/edge_fallback --batch --model /path/to/checkpoint \
  --cloud-cooldown-ms 10000 --cloud-timeout-ms 1500
```

After an HTTP 503, the batch routes requests straight to MLX for ten seconds
in this example. The interval starts when the 503 arrives. After it expires,
the next request tries the cloud again. Another 503 starts a new interval. The final 503 status is sufficient even if its body is truncated.
Local responses record `fallback_reason: "cloud_503"` for the initial
response and `fallback_reason: "circuit_open"` with a null `cloud_status`
for requests that skipped the cloud.

The circuit starts closed and is disabled unless a cooldown is supplied.
HTTP 429, other error statuses, and ambiguous transport failures remain
visible as errors. The default cloud request deadline is 90 seconds.

## Reuse the start of a prompt

Use `--prefix-cache-tokens N` with `--batch --model DIR` when requests
share the same opening context. The batch retains one prefix of exactly N
token IDs after chat formatting. A hit restores its per-layer K/V arrays and
runs prefill on the remaining suffix at the original positions. A different
leading sequence evicts the old entry. Prompts with no suffix use ordinary
prefill and leave the current entry intact. Each generation gets its own
suffix and decode state.

The model checks the option against its context limit and a 256 MiB logical
K/V budget calculated conservatively with four bytes per element. The limit
covers the retained prefix. Layer activations, the request's full K/V cache,
weights, and allocator overhead also use memory. Local replies report
`reused_prefix_tokens`, `retained_prefix_tokens`, and `retained_prefix_bytes`.
Pass `--trace-token-ids` to include the generated IDs for parity checks.

```sh
sh bench/prefix_ab.sh ./build-mlx/edge_fallback /path/to/checkpoint 1024 6 > paired.csv
```

The benchmark requires `jq`. It runs the same release binary with prefix
reuse disabled and enabled, alternates process order, and checks all
generated IDs and cache outcomes before emitting CSV.

### Measured on Apple Metal

The release build ran Llama 3.2 3B Instruct 4-bit on an Apple M3 Pro with
MLX 0.32.5. Six alternating process pairs each sent eight requests. The
handbook requests had 1,355 prompt tokens and the short request had 24.
All 48 pairs returned the same greedy token IDs and each response generated
one token. The retained 1,024-token prefix used 112 MiB of K/V payload.

| Case | Pairs | Median prefill without reuse | Median prefill with cache enabled |
|---|---:|---:|---:|
| Exact prefix hit | 24 | 1,826.3 ms | 449.2 ms |
| Different prefix replaces the entry | 12 | 1,756.5 ms | 1,631.6 ms |
| Short prompt uses ordinary prefill | 6 | 72.4 ms | 58.5 ms |

The median of the paired hit speed ratios was 3.85×. Hit p90 was 2,075.2 ms
without reuse and 526.7 ms with reuse, using nearest-rank percentiles. Prefill
time includes tokenization and the first greedy decision. Model loading is
separate. The six cold warmup rows are outside the table. The short case
runs the same code in both configurations and shows run-to-run variation.
These results describe prefill and one-token responses for this workload.
[Raw timing pairs](evidence/llama-3b-prefix-ab.csv) · [Build and checkpoint fingerprints](evidence/llama-3b-prefix-ab-environment.txt)

## Measure first-token and decode timing

Each local result includes these fields in `local_metrics`.

| Field | Measurement |
|---|---|
| `ttft_ms` | Local generation start through the first synchronized non-stop token ID. Includes tokenization and prefill. Null when the first decision is a stop ID. |
| `decode_ms` | First through last non-stop token decision. Null for fewer than two generated IDs. |
| `decode_tokens_per_second` | Generated IDs after the first, divided by decode time. Null when there is no positive decode interval. |
| `stop_decision_ms` | Time from the last non-stop token through the extra stop-token decision, when both exist. |
| `stop_reason` | `stop_token` for a model stop ID or `token_limit` for the requested output cap. |

The batch CLI sends a complete JSON line after generation. The timings are
local to the model and exclude cloud calls, initial model load and response
delivery. `generation_ms` still covers detokenization and final local
bookkeeping. `prefill_ms` also includes the first decision when it is a stop ID.

```sh
sh bench/long_decode_ab.sh ./build-mlx/edge_fallback /path/to/checkpoint 1024 4 > long-decode.csv
```

Each process first seeds the 1,024-token prefix with a short decode. It then
generates 48 tokens for each of two longer prompts. The runner alternates
process and prompt-length order independently, compares all output IDs between cache-off
and cache-on runs, and checks the decode arithmetic before emitting CSV.

### Long prompts on Apple Metal

Four release process pairs on an Apple M3 Pro ran 4-bit Llama 3.2 3B
with MLX 0.32.5. Each process seeded the model and then ran both long
shapes. All eight measured pairs produced identical text and generated
token IDs between cache-off and cache-on, with 48 tokens per response.
The table excludes the four seed rows. Both long shapes were first use
within each model-resident process.

| Prompt tokens | TTFT off → on, median | Decode off, median (range) | Decode on, median (range) |
|---:|---:|---:|---:|
| 2,700 | 3,737 → 2,265 ms | 14.70 tok/s (3.74–29.97) | 11.09 tok/s (4.53–28.45) |
| 4,620 | 6,476 → 4,822 ms | 9.93 tok/s (4.10–16.33) | 10.43 tok/s (1.32–17.81) |

Each decode rate uses 47 post-first-token intervals. Cache hits were
faster to the first token in every measured pair. Decode rates varied
widely across processes, and the medians moved in different directions.
[Raw timing pairs](evidence/llama-3b-long-decode.csv) · [Environment and checksums](evidence/llama-3b-long-decode-environment.txt)

## Tests

The default build tests the CLI against a local HTTP server. It covers cloud
success, fallback selection, malformed and oversized replies, and JSONL
recovery after an invalid request.

```sh
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

An MLX build also tests a deterministic, generated Qwen2-style checkpoint on
Metal. The test checks dense attention, output token limits, EOS and context
handling, and invalid checkpoint files. For the cached 4-bit Llama 3.2 3B
integration test, configure `build-mlx` with
`-DEDGE_TEST_MODEL=/path/to/Llama-3.2-3B-Instruct-4bit`.
Run `ctest --test-dir build-mlx --output-on-failure` to include the real
checkpoint. A local server supplies the HTTP 503 responses. The prefix test
checks exact token parity against ordinary prefill, including multi-token
decode, prefix replacement, and an intervening short request. The dense
fixture covers immediate EOS, EOS after multiple tokens, and token limits.
Set `-DEDGE_TEST_LONG_DECODE=ON` with the Llama checkpoint and `jq` to run
the long-prompt parity check in CTest.

To measure the application and MLX code, configure each build separately
with Clang and `-DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping"`.
On macOS, run these commands from the source directory with a fresh instrumented build. On Linux, use the matching LLVM tools directly.

```sh
build_dir=build  # Use build-mlx for the Metal configuration.
LLVM_PROFILE_FILE="$PWD/$build_dir/%p-%m.profraw" \
  ctest --test-dir "$build_dir" --output-on-failure
xcrun llvm-profdata merge -sparse "$build_dir"/*.profraw -o "$build_dir/run.profdata"
xcrun llvm-cov report "$build_dir/edge_fallback" \
  -instr-profile "$build_dir/run.profdata" "$PWD/src"
```
