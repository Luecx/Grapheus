#pragma once

#include "../../operations/operations.h"
#include "layer.h"

struct AffineBatched : public nn::Layer {

    size_t     batches;
    nn::Layer* prev;

    nn::Tape   weights {0, 0};
    nn::Tape   bias {0, 0};

    AffineBatched(Layer* prev, size_t size, size_t batches)
        : Layer(size)
        , batches(batches)
        , prev(prev) {
        prev->use();
        ERROR(prev->size % batches == 0);
        ERROR(size % batches == 0);
    }

    void compile(size_t batch_size) override {
        compile_suboutput(batch_size, nn::Tape(size, batch_size));
        // create weights and biases
        // clang-format off
        weights = nn::Tape(size, prev->size / batches);
        bias    = nn::Tape(size, 1);
        weights.malloc();
        bias   .malloc();

        math::normal(weights.values, 0.f, 1.0f / std::sqrtf(prev->size));
        weights.values >> data::GPU;

        // clang-format on
    }

    void compile_suboutput(size_t batch_size, const nn::Tape& output) override {
        Layer::compile_suboutput(batch_size, output);
    }

    void forward() override {
        Layer::forward();
        operations::affine_batched<data::GPU>(prev->dense_output.values,
                                              weights           .values,
                                              bias              .values,
                                              dense_output      .values,
                                              batches);
    }
    void backward() override {
        Layer::backward();

        operations::affine_batched_bp<data::GPU>(prev->dense_output.values,
                                                 prev->dense_output.gradients,
                                                 weights           .values,
                                                 weights           .gradients,
                                                 bias              .gradients,
                                                 dense_output      .gradients,
                                                 batches);

    }
};