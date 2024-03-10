//
// Created by finne on 08.04.2023.
//

#pragma once
#include <chrono>
#include <cstdint>

class Timer {
    public:
    // starts the timer
    void start();

    // stops the timer
    void stop();

    // returns the elapsed time in milliseconds
    [[nodiscard]] uint64_t elapsed() const;

    private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
};
