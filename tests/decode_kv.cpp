#include "../src/decode_kv.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace mx = mlx::core;

void check_isolation() {
    const auto prefix = mx::reshape(mx::arange(24, mx::float32), {1, 2, 3, 4});
    auto storage = prefix;
    auto expected = prefix;
    for (int offset = 3; offset < 6; ++offset) {
        const auto token = mx::full({1, 2, 1, 4}, static_cast<float>(offset), mx::float32);
        expected = mx::concatenate({expected, token}, 2);
        const auto live = edge::append_decode_kv(storage, token, offset, 6);
        if (storage.shape(2) != 6 || live.shape(2) != offset + 1
            || !mx::array_equal(live, expected).item<bool>()
            || !mx::array_equal(prefix, mx::reshape(mx::arange(24, mx::float32), {1, 2, 3, 4})).item<bool>()) {
            throw std::runtime_error("decode append changed live values or the saved prefix");
        }
    }
}

void check_growth() {
    auto storage = mx::ones({1, 2, 3, 4});
    const auto fill = mx::full({1, 2, 63, 4}, 2, mx::float32);
    const auto first = edge::append_decode_kv(storage, fill, 3, 70);
    if (storage.shape(2) != 67 || first.shape(2) != 66) {
        throw std::runtime_error("decode block did not reserve the expected capacity");
    }
    const auto last = mx::full({1, 2, 2, 4}, 3, mx::float32);
    const auto live = edge::append_decode_kv(storage, last, 66, 70);
    const auto expected = mx::concatenate({mx::ones({1, 2, 3, 4}), fill, last}, 2);
    if (storage.shape(2) != 70 || live.shape(2) != 68 || !mx::array_equal(live, expected).item<bool>()) {
        throw std::runtime_error("growing the decode block lost values or exceeded its limit");
    }
}

void probe(bool reserved) {
    using Clock = std::chrono::steady_clock;
    constexpr int positions = 2700;
    constexpr int steps = 12;
    auto storage = mx::ones({1, 8, positions, 128}, mx::bfloat16);
    const auto token = mx::full({1, 8, 1, 128}, 2, mx::bfloat16);
    mx::eval(storage, token);
    mx::clear_cache();
    int reused = 0;
    const auto start = Clock::now();
    for (int offset = positions; offset < positions + steps; ++offset) {
        const auto previous = reinterpret_cast<std::uintptr_t>(storage.buffer().ptr());
        if (reserved) {
            const auto live = edge::append_decode_kv(storage, token, offset, positions + steps);
            mx::eval(live);
        } else {
            storage = mx::concatenate({storage, token}, 2);
            mx::eval(storage);
        }
        reused += previous == reinterpret_cast<std::uintptr_t>(storage.buffer().ptr());
    }
    const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::cout << (reserved ? "reserved" : "concatenate") << ',' << ms << ',' << reused << '/' << steps
              << ',' << mx::get_cache_memory() << '\n';
}

int main(int argc, char** argv) try {
    const std::string_view mode = argc == 2 ? argv[1] : "";
    mx::set_default_device(mode == "--cpu" ? mx::Device::cpu : mx::Device::gpu);
    check_isolation();
    check_growth();
    if (mode == "--probe") {
        // Warm both paths, then alternate order. One BF16 tensor, no model load.
        probe(false);
        probe(true);
        std::cout << "samples_ms_and_retained_allocator_bytes\n";
        for (int i = 0; i < 6; ++i) {
            probe(i % 2 == 0);
            probe(i % 2 != 0);
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
