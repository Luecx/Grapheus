#pragma once

#include "layer.h"
#include "../../operations/operations.h"

namespace nn {
struct ReLU : public Layer {
    Layer* prev;
    ReLU(Layer* prev)
        : Layer(prev->size)
        , prev(prev) {
        prev->use();
    }

    void compile(size_t batch_size) override {
        this->compile_suboutput(batch_size, Tape {size, batch_size});
    }

    void forward() override {
        Layer::forward();
        operations::relu<data::GPU>(prev->dense_output.values, dense_output.values);
    }
    void backward() override {
        Layer::backward();
        operations::relu_bp<data::GPU>(prev->dense_output.values,
                                       prev->dense_output.gradients,
                                       dense_output.values,
                                       dense_output.gradients);
    }
};

struct Sigmoid : public Layer {
    Layer* prev;
    float  scalar;
    explicit Sigmoid(Layer* prev, float scalar = 1)
        : Layer(prev->size)
        , prev(prev)
        , scalar(scalar) {
        prev->use();
    }

    void compile(size_t batch_size) override {
        this->compile_suboutput(batch_size, Tape {size, batch_size});
    }

    void forward() override {
        Layer::forward();
        operations::sigmoid<data::GPU>(prev->dense_output.values, dense_output.values, scalar);
    }
    void backward() override {
        Layer::backward();
        operations::sigmoid_bp<data::GPU>(prev->dense_output.values,
                                          prev->dense_output.gradients,
                                          dense_output.values,
                                          dense_output.gradients,
                                          scalar);
    }
};

struct ClippedRelu : public Layer {
    Layer* prev;
    float  max;
    explicit ClippedRelu(Layer* prev, float max = 1)
        : Layer(prev->size)
        , prev(prev)
        , max(max) {
        prev->use();
    }

    void compile(size_t batch_size) override {
        this->compile_suboutput(batch_size, Tape {size, batch_size});
    }

    void forward() override {
        Layer::forward();
        operations::crelu<data::GPU>(prev->dense_output.values, dense_output.values, max);
    }
    void backward() override {
        Layer::backward();
        operations::crelu_bp<data::GPU>(prev->dense_output.values,
                                        prev->dense_output.gradients,
                                        dense_output.values,
                                        dense_output.gradients,
                                        max);
    }
};

struct LeakyReLU : public Layer {
    Layer* prev;
    explicit LeakyReLU(Layer* prev)
        : Layer(prev->size)
        , prev(prev) {
        prev->use();
    }

    void compile(size_t batch_size) override {
        this->compile_suboutput(batch_size, Tape {size, batch_size});
    }

    void forward() override {
        Layer::forward();
        operations::lrelu<data::GPU>(prev->dense_output.values, dense_output.values);
    }
    void backward() override {
        Layer::backward();
        operations::lrelu_bp<data::GPU>(prev->dense_output.values,
                                        prev->dense_output.gradients,
                                        dense_output.values,
                                        dense_output.gradients);
    }
};
}    // namespace nn
