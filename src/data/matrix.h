//
// Created by Luecx on 18.03.2023.
//

#pragma once
#include "sarray.h"

#include <cstdint>

namespace data {
#define MATRIX_INDEX(ld, m, n) (ld * n + m)

class Matrix {

    public:
    size_t m;
    size_t n;

    Matrix(size_t m, size_t n)
        : m(m)
        , n(n) {}
    Matrix(const Matrix& other) {
        this->m = other.m;
        this->n = other.n;
    }
    Matrix(Matrix&& other) {
        this->m = other.m;
        this->n = other.n;
    }
    Matrix& operator=(const Matrix& other) {
        this->m = other.m;
        this->n = other.n;
        return *this;
    }
    Matrix& operator=(Matrix&& other) {
        this->m = other.m;
        this->n = other.n;
        return *this;
    }
};
}    // namespace data
