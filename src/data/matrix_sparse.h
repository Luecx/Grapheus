#pragma once
#include "device.h"
#include "matrix.h"

#include <iostream>

namespace data {
struct SparseMatrix : public data::Matrix {
    size_t               max_entries_per_column;
    data::SArray<size_t> values {0};

    SparseMatrix(size_t m, size_t n, size_t max_entries_per_column)
        : Matrix(m, n)
        , max_entries_per_column(max_entries_per_column) {
        values = data::SArray<size_t> {n * (1 + max_entries_per_column)};
    }

    void set(int input_idx, int index) {
        auto offset = (max_entries_per_column + 1) * input_idx;
        ASSERT(values.address<data::CPU>());
        ASSERT(values.address<data::CPU>()[offset] <= max_entries_per_column);
        values[offset]++;
        values[offset + values[offset]] = index;
    }

    void sort(int input_idx) {
        auto offset  = (max_entries_per_column + 1) * input_idx;
        auto entries = values[offset];
        auto first   = &(values[offset + 1]);
        std::sort(first, first + entries);
    }

    void malloc() {
        values.malloc<data::BOTH>();
    }

    size_t count(int input_idx) {
        auto offset = (max_entries_per_column + 1) * input_idx;
        return values[offset];
    }

    void clear() {
        for (int i = 0; i < n; i++) {
            values.address<data::CPU>()[i * (max_entries_per_column + 1)] = 0;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const SparseMatrix& data) {
        os << std::fixed << std::setprecision(0);
        for (int p_i = 0; p_i <= data.max_entries_per_column; p_i++) {
            for (int p_n = 0; p_n < data.n; p_n++) {
                int count = data.values(p_n * (data.max_entries_per_column + 1));
                if (p_i > count) {
                    os << std::setw(11) << ".";
                } else {
                    os << std::setw(11)
                       << (int) data.values(p_i + p_n * (data.max_entries_per_column + 1));
                }
            }
            os << "\n";
            if (p_i == 0) {
                for (int p_n = 0; p_n < data.n; p_n++) {
                    os << "-----------";
                }
                os << "\n";
            }
        }
        return os;
    }
};

}    // namespace data
