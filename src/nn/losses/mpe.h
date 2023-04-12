#pragma once

#include "../../operations/operations.h"
#include "loss.h"

namespace nn {

struct MPE : public Loss {

    float power;
    bool  average_gradients = true;

    MPE(float power = 2, bool averageGradients = true)
        : power(power)
        , average_gradients(averageGradients) {}

    virtual void apply(Tape& output, data::DenseMatrix<float> target) {
        operations::mpe<data::GPU>(output.values,
                                   output.gradients,
                                   target,
                                   loss,
                                   power,
                                   average_gradients ? 1.0f / (output.values.m * output.values.n)
                                                     : 1.0f);
    }
};

}    // namespace nn