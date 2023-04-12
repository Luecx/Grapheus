#pragma once
#include <limits>

namespace nn {
struct OptimizerEntry {
    Tape*          m_reference;
    float          m_min       = std::numeric_limits<float>::lowest();
    float          m_max       = std::numeric_limits<float>::max();
    float          m_lasso     = 0.0f;
    float          m_ridge     = 0.0f;
    float          m_lr_scalar = 1.0f;



    [[nodiscard]] OptimizerEntry min(float min_val) const {
        OptimizerEntry res {*this};
        res.m_min = min_val;
        return res;
    }

    [[nodiscard]] OptimizerEntry max(float max_val) const {
        OptimizerEntry res {*this};
        res.m_max = max_val;
        return res;
    }

    [[nodiscard]] OptimizerEntry clamp(float min_val, float max_val) const {
        OptimizerEntry res {*this};
        res.m_min = min_val;
        res.m_max = max_val;
        return res;
    }

    [[nodiscard]] OptimizerEntry lasso(float lasso_val) const {
        OptimizerEntry res {*this};
        res.m_lasso = lasso_val;
        return res;
    }

    [[nodiscard]] OptimizerEntry ridge(float ridge_val) const {
        OptimizerEntry res {*this};
        res.m_ridge = ridge_val;
        return res;
    }

    [[nodiscard]] OptimizerEntry lr_scalar(float lr_scalar_val) const {
        OptimizerEntry res {*this};
        res.m_lr_scalar = lr_scalar_val;
        return res;
    }
};
}    // namespace nn
