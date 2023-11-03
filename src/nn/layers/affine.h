#pragma once

#include "../../math/random.h"
#include "../../operations/operations.h"
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

        weights = Tape(size, prev->size);
        weights.malloc();

        bias = Tape(size, 1);
        bias.malloc();

        math::kaiming<float>(weights.values, prev->size);
        math::fill<float>(bias.values, 0.0);

        weights.values >> data::GPU;
        bias.values >> data::GPU;
    }

    void compile(size_t batch_size) override {
        // set output matrix
        compile_suboutput(batch_size, Tape(size, batch_size));
    }

    void forward() override {
        operations::affine<data::GPU>(prev->dense_output.values,
                                      weights.values,
                                      bias.values,
                                      dense_output.values);
    }
    void backward() override {
        operations::affine_bp<data::GPU>(prev->dense_output.values,
                                         prev->dense_output.gradients,
                                         weights.values,
                                         weights.gradients,
                                         bias.gradients,
                                         dense_output.gradients);
    }

    std::vector<Tape*> params() override {
        return std::vector<Tape*> {&weights, &bias};
    }
};
}    // namespace nn
