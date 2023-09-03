#pragma once

#include "../../operations/operations.h"
#include "layer.h"

namespace nn {

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

        // create weights and biases
        weights = nn::Tape(size, prev->size / batches);
        bias    = nn::Tape(size, 1);
        weights.malloc();
        bias.malloc();

        math::kaiming<float>(weights.values, prev->size / batches);
        math::fill<float>(bias.values, 0.0f);

        const size_t out_batch_size = size / batches;
        for (size_t i = out_batch_size; i < size; i++)
            for (size_t j = 0; j < prev->size / batches; j++)
                weights.values(i, j) = weights.values(i % out_batch_size, j);

        weights.values >> data::GPU;
        bias.values >> data::GPU;
    }

    void compile(size_t batch_size) override {
        compile_suboutput(batch_size, nn::Tape(size, batch_size));
    }

    void compile_suboutput(size_t batch_size, const nn::Tape& output) override {
        Layer::compile_suboutput(batch_size, output);
    }

    void forward() override {
        Layer::forward();
        operations::affine_batched<data::GPU>(prev->dense_output.values,
                                              weights.values,
                                              bias.values,
                                              dense_output.values,
                                              batches);
    }
    void backward() override {
        Layer::backward();

        operations::affine_batched_bp<data::GPU>(prev->dense_output.values,
                                                 prev->dense_output.gradients,
                                                 weights.values,
                                                 weights.gradients,
                                                 bias.gradients,
                                                 dense_output.gradients,
                                                 batches);
    }

    std::vector<Tape*> params() override {
        return std::vector<Tape*> {&weights, &bias};
    }
};

}    // namespace nn