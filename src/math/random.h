
#pragma once

#include "../data/matrix_dense.h"
#include <random>

namespace math{

extern std::mt19937 twister;

inline void seed(uint32_t seed_value){
    twister.seed(seed_value);
}

template<typename TYPE>
inline void normal(data::DenseMatrix<TYPE>& matrix, TYPE mean, TYPE dev){
    // TODO: remove the generator here and use the twister
    // this is only for reproducibility compared to CudAD
    std::default_random_engine     generator{};
    std::normal_distribution<TYPE> distribution(mean, dev);
    for (int j = 0; j < matrix.n; j++) {
        for (int i = 0; i < matrix.m; i++) {
            matrix.get(i, j) = distribution(generator);
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
    } else if constexpr(std::is_floating_point_v<TYPE>) {
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


}

