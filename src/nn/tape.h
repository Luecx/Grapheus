#pragma once
#include "../data/matrix_dense.h"

namespace nn {
struct Tape {
    data::DenseMatrix<float> values;
    data::DenseMatrix<float> gradients;

    Tape(size_t m, size_t n)
        : values(data::DenseMatrix {m, n})
        , gradients(data::DenseMatrix {m, n}) {};

    Tape(const Tape& other, size_t m, size_t n, size_t offset_m = 0, size_t offset_n = 0)
        : values(other.values, m, n, offset_m, offset_n)
        , gradients(other.gradients, m, n, offset_m, offset_n) {};

    void malloc() {
        values.malloc<data::BOTH>();
        gradients.malloc<data::BOTH>();
    }
};
}    // namespace nn

