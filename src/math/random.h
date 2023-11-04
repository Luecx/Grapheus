
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
    if (!matrix.template is_allocated<data::CPU>()) {
        matrix.template malloc<data::CPU>();
    }
    matrix = value;
}

template<typename TYPE>
inline void kaiming(data::DenseMatrix<TYPE>& matrix, size_t expected_inputs) {
    if (!matrix.template is_allocated<data::CPU>()) {
        matrix.template malloc<data::CPU>();
    }
    std::uniform_real_distribution<> dis(0.0, 1.0);
    matrix.for_each([&](size_t, size_t, TYPE& element) {
        auto r1 = dis(twister), r2 = dis(twister);
        auto r  = std::sqrt(-2.0 * std::log(r1)) * std::cos(6.28318530718 * r2);
        element = r * std::sqrt(2.0 / expected_inputs);
    });
}

template<typename TYPE>
inline void normal(data::DenseMatrix<TYPE>& matrix, TYPE mean, TYPE dev) {
    if (!matrix.template is_allocated<data::CPU>()) {
        matrix.template malloc<data::CPU>();
    }
    std::normal_distribution<TYPE> distribution(mean, dev);
    matrix.for_each([&](size_t, size_t, TYPE& element) { element = distribution(twister); });
}

template<typename TYPE>
inline void uniform(data::DenseMatrix<TYPE>& matrix, TYPE lower, TYPE upper) {
    if (!matrix.template is_allocated<data::CPU>()) {
        matrix.template malloc<data::CPU>();
    }
    if constexpr (std::is_integral_v<TYPE>) {
        std::uniform_int_distribution<TYPE> distribution(lower, upper);
        matrix.for_each([&](size_t, size_t, TYPE& element) { element = distribution(twister); });
    } else if constexpr (std::is_floating_point_v<TYPE>) {
        std::uniform_real_distribution<TYPE> distribution(lower, upper);
        matrix.for_each([&](size_t, size_t, TYPE& element) { element = distribution(twister); });
    }
}

inline void uniform(data::DenseMatrix<bool>& matrix, bool lower, bool upper) {
    if (!matrix.template is_allocated<data::CPU>()) {
        matrix.template malloc<data::CPU>();
    }
    std::uniform_int_distribution<int> distribution(lower ? 1 : 0, upper ? 1 : 0);
    matrix.for_each([&](size_t, size_t, bool& element) { element = distribution(twister) != 0; });
}

template<typename TYPE>
class Initialiser {
    public:
    virtual void operator()(data::DenseMatrix<TYPE>& matrix, size_t expected_inputs) const {};
};

template<typename TYPE>
class FillInitialiser : public Initialiser<TYPE> {
    TYPE value;

    public:
    FillInitialiser(TYPE v)
        : value(v) {}

    void operator()(data::DenseMatrix<TYPE>& matrix, size_t expected_inputs) const {
        fill(matrix, value);
    }
};

template<typename TYPE>
class KaimingInitializer : public Initialiser<TYPE> {
    void operator()(data::DenseMatrix<TYPE>& matrix, size_t expected_inputs) const {
        kaiming(matrix, expected_inputs);
    }
};

}    // namespace math
