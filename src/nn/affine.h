#pragma once

#include "layer.h"
#include "../operations/operations.h"

namespace nn {

struct Affine : public nn::Layer {

    Layer* prev;

    // weights and biases
    Tape weights {0, 0};
    Tape bias {0, 0};

    Affine(Layer* prev, size_t size)
        : Layer(size)
        , prev(prev) {}

    void compile(size_t batch_size) override {
        // TODO add randomization
        weights = Tape(size, prev->size);
        weights.malloc();

        // TODO add randomization
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
            bias.values,
            dense_output.values);
    }
};
}    // namespace nn
