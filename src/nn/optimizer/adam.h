#pragma once

#include "../../operations/operations.h"
#include "optimizer.h"

namespace nn {

struct Adam : public Optimizer{

    std::vector<data::DenseMatrix<float>> first_moment{};
    std::vector<data::DenseMatrix<float>> second_moment{};

    float beta1;
    float beta2;
    float eps;

    Adam(const std::vector<OptimizerEntry>& entries, float beta1, float beta2, float eps)
        : Optimizer(entries)
        , beta1(beta1)
        , beta2(beta2)
        , eps(eps) {
        first_moment .reserve(1024);
        second_moment.reserve(1024);
    }

    protected:
    void add_field(OptimizerEntry& entry) override {
        first_moment.emplace_back(
            entry.m_reference->values.m, entry.m_reference->values.n);
        second_moment.emplace_back(
            entry.m_reference->values.m, entry.m_reference->values.n);

        first_moment .back().malloc<data::BOTH>();
        second_moment.back().malloc<data::BOTH>();
    }

    void step(OptimizerEntry& entry, int idx, float lr) override {
        float bc1 = 1.0f - std::pow(beta1, step_);
        float bc2 = 1.0f - std::pow(beta2, step_);
        float slr = lr * sqrtf(bc2) / bc1;

        operations::adam<data::GPU>(
            entry.m_reference->values,
            entry.m_reference->gradients,
            first_moment [idx],
            second_moment[idx],
            slr,
            beta1,
            beta2,
            eps,
            entry.m_min,
            entry.m_max,
            entry.m_lasso,
            entry.m_ridge);
    }
};

}