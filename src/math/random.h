
#pragma once

#include "../data/matrix_dense.h"

#include <random>

namespace math {

extern std::mt19937 twister;

inline void         seed(uint32_t seed_value) {
    twister.seed(seed_value);
}

template<typename TYPE>
inline void fill(data::DenseMatrix<TYPE>& matrix, TYPE value) {
    for (size_t i = 0; i < matrix.m; i++)
        for (size_t j = 0; j < matrix.n; j++)
            matrix.get(i, j) = value;
}

template<typename TYPE>
inline void kaiming(data::DenseMatrix<TYPE>& matrix, size_t expected_inputs) {
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < matrix.m; i++) {
        for (size_t j = 0; j < matrix.n; j++) {
            auto r1 = dis(twister), r2 = dis(twister);
            auto r           = std::sqrt(-2.0 * std::log(r1)) * std::cos(6.28318530718 * r2);
            matrix.get(i, j) = r * std::sqrt(2.0 / expected_inputs);
        }
    }
}

template<typename TYPE>
inline void normal(data::DenseMatrix<TYPE>& matrix, TYPE mean, TYPE dev) {
    std::normal_distribution<TYPE> distribution(mean, dev);
    for (int j = 0; j < matrix.n; j++) {
        for (int i = 0; i < matrix.m; i++) {
            matrix.get(i, j) = distribution(twister);
        }
    }
}

template<typename TYPE>
inline void uniform(data::DenseMatrix<TYPE>& matrix, TYPE lower, TYPE upper) {
    if constexpr (std::is_integral_v<TYPE>) {
        std::uniform_int_distribution<TYPE> distribution(lower, upper);
        for (int i = 0; i < matrix.m; i++) {
            for (int j = 0; j < matrix.n; j++) {
                matrix.get(i, j) = distribution(twister);
            }
        }
    } else if constexpr (std::is_floating_point_v<TYPE>) {
        std::uniform_real_distribution<TYPE> distribution(lower, upper);
        for (int i = 0; i < matrix.m; i++) {
            for (int j = 0; j < matrix.n; j++) {
                matrix.get(i, j) = distribution(twister);
            }
        }
    }
}

inline void uniform(data::DenseMatrix<bool>& matrix, bool lower, bool upper) {
    std::uniform_int_distribution<int> distribution(lower, upper);
    for (int i = 0; i < matrix.m; i++) {
        for (int j = 0; j < matrix.n; j++) {
            matrix.get(i, j) = (bool) distribution(twister);
        }
    }
}

}    // namespace math
