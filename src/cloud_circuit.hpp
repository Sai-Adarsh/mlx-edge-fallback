#pragma once

#include <chrono>
#include <optional>

class CloudCircuit {
public:
    using Clock = std::chrono::steady_clock;

    explicit CloudCircuit(std::chrono::milliseconds cooldown) : cooldown_(cooldown) {}

    bool bypass(Clock::time_point now) const {
        return retry_at_ && now < *retry_at_;
    }

    // A sequential session lets the first request after the deadline probe the cloud.
    void observe(long status, Clock::time_point now) {
        if (status == 503 && cooldown_.count() > 0) retry_at_ = now + cooldown_;
        else retry_at_.reset();
    }

private:
    std::chrono::milliseconds cooldown_;
    std::optional<Clock::time_point> retry_at_;
};
