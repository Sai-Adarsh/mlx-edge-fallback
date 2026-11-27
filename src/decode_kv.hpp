#pragma once

#include <mlx/mlx.h>

#include <algorithm>

namespace edge {

// Cache tensors are [batch, KV heads, positions, head size]. Only the
// incremental decoder calls this function; prefix snapshots stay compact.
inline mlx::core::array append_decode_kv(
    mlx::core::array& storage, const mlx::core::array& update, int offset, int limit) {
    namespace mx = mlx::core;
    const mx::Shape origin{0, 0, 0, 0};
    auto end = storage.shape();
    const int live_positions = offset + update.shape(2);
    const int capacity = storage.shape(2);
    if (capacity < live_positions) {
        // Limit the additional allocation if generation stops before its budget.
        constexpr int chunk_positions = 64;
        auto tail = end;
        tail[2] = std::min(limit - capacity, std::max(chunk_positions, live_positions - capacity));
        storage = mx::concatenate({storage, mx::zeros(tail, storage.dtype())}, 2);
    }
    end[2] = live_positions;
    // A stable allocation size allows reuse by the allocator even when the
    // backend needs to copy instead of donating the input buffer.
    storage = mx::slice_update(storage, update, {0, 0, offset, 0}, end);
    return mx::slice(storage, origin, end);
}

} // namespace edge
