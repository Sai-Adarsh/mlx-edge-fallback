# MLX edge fallback

A C++ client that runs a local 3B model on Apple Silicon when the cloud returns HTTP 503.

## Run with a fixture

Requires CMake 3.27+, C++20, and libcurl. The demo simulates a 503.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/edge_fallback --demo
```

- [Set up MLX and a local model](docs/reference.md#run-a-local-3b-model)
- [Cloud outage behavior](docs/reference.md#back-off-during-a-cloud-outage)
- [Prefix reuse benchmark](docs/reference.md#reuse-the-start-of-a-prompt)
- [First-token and decode benchmarks](docs/reference.md#measure-first-token-and-decode-timing)
- [Full reference](docs/reference.md)
