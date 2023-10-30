#pragma once

#include "adam.h"

namespace nn {

struct AdamWarmup : public Adam {

    int warmup = 0;

    AdamWarmup(const std::vector<OptimizerEntry>& entries,
               float                              beta1,
               float                              beta2,
               float                              eps,
               int                                warmup)
        : Adam(entries, beta1, beta2, eps)
        , warmup(warmup) {}

    protected:
    void step(OptimizerEntry& entry, int idx, float lr) override {
        float slr = warmup > step_ ? 1e-8 + step_ * lr / warmup : lr;

        Adam::step(entry, idx, slr);
    }
};

}    // namespace nn