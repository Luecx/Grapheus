//
// Created by Luecx on 18.03.2023.
//

#pragma once
#include <cstdint>

namespace data {

enum Device : uint8_t {
    NONE,    // 00
    CPU,     // 01
    GPU,     // 10
    BOTH     // 11
};

constexpr inline bool is_cpu(Device dev) {
    return dev & CPU;
}

constexpr inline bool is_gpu(Device dev) {
    return dev & GPU;
}
}    // namespace data
