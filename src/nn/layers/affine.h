#pragma once

#include "../../operations/operations.h"
#include "../../math/random.h"
#include "layer.h"

namespace nn {

struct Affine : public nn::Layer {

    Layer* prev;

    // weights and biases
    Tape weights {0, 0};
    Tape bias {0, 0};

    Affine(Layer* prev, size_t size)
        : Layer(size)
        , prev(prev) {
        prev->use();
    }

    void compile(size_t batch_size) override {
        weights = Tape(size, prev->size);
        weights.malloc();

        math::normal(weights.values, 0.f, std::sqrtf(2.0f / prev->size));
        weights.values >> data::GPU;

        bias = Tape(size, 1);
        bias.malloc();

        // set output matrix
        compile_suboutput(batch_size, Tape(size, batch_size));
    }

    void forward() override {
        operations::affine<data::GPU>(
            prev->dense_output.values,
            weights.values,
            bias.values,
            dense_output.values);
    }
    void backward() override {
        operations::affine_bp<data::GPU>(
            prev->dense_output.values,
            prev->dense_output.gradients,
            weights.values,
            weights.gradients,
            bias.gradients,
            dense_output.gradients);
    }
};
}    // namespace nn
