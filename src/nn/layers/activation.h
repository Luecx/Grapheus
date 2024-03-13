#pragma once

#include "../../operations/operations.h"
#include "layer.h"

#include <functional>

namespace nn {

struct ReLU : public Layer {

    Layer*            prev;
    GradientOperation grad_op;
    int               use_id;

    ReLU(Layer* prev)
        : Layer(prev->size)
        , prev(prev) {
        use_id = prev->use();
    }

    void compile(size_t batch_size) override {
        this->compile_suboutput(batch_size, Tape {size, batch_size});
        this->grad_op = use_id == prev->used() ? SET : INCREMENT;
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
                                       dense_output.gradients,
                                       0,
                                       grad_op);
    }
};

struct Linear : public Layer {
    Layer*            prev;
    GradientOperation grad_op;
    int               use_id;
    float             scalar;
    explicit Linear(Layer* prev, float scalar = 1)
        : Layer(prev->size)
        , prev(prev)
        , scalar(scalar) {
        prev->use();
        use_id = prev->use();
    }

    void compile(size_t batch_size) override {
        this->compile_suboutput(batch_size, Tape {size, batch_size});
        this->grad_op = use_id == prev->used() ? SET : INCREMENT;
    }

    void forward() override {
        Layer::forward();
        operations::linear<data::GPU>(prev->dense_output.values, dense_output.values, scalar);
    }

    void backward() override {
        Layer::backward();
        operations::linear_bp<data::GPU>(prev->dense_output.values,
                                          prev->dense_output.gradients,
                                          dense_output.values,
                                          dense_output.gradients,
                                          scalar,
                                          grad_op);
    }
};


struct Sigmoid : public Layer {
    Layer*            prev;
    GradientOperation grad_op;
    int               use_id;
    float             scalar;
    explicit Sigmoid(Layer* prev, float scalar = 1)
        : Layer(prev->size)
        , prev(prev)
        , scalar(scalar) {
        prev->use();
        use_id = prev->use();
    }

    void compile(size_t batch_size) override {
        this->compile_suboutput(batch_size, Tape {size, batch_size});
        this->grad_op = use_id == prev->used() ? SET : INCREMENT;
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
                                          scalar,
                                          grad_op);
    }
};

struct ClippedRelu : public Layer {
    Layer*            prev;
    GradientOperation grad_op;
    int               use_id;
    float             max;
    explicit ClippedRelu(Layer* prev, float max = 1)
        : Layer(prev->size)
        , prev(prev)
        , max(max) {
        use_id = prev->use();
    }

    void compile(size_t batch_size) override {
        this->compile_suboutput(batch_size, Tape {size, batch_size});
        this->grad_op = use_id == prev->used() ? SET : INCREMENT;
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
                                        max,
                                        grad_op);
    }
};

struct SqrClippedRelu : public Layer {
    Layer*            prev;
    GradientOperation grad_op;
    int               use_id;
    float             max;
    explicit SqrClippedRelu(Layer* prev, float max = 1)
        : Layer(prev->size)
        , prev(prev)
        , max(max) {
        use_id = prev->use();
    }

    void compile(size_t batch_size) override {
        this->compile_suboutput(batch_size, Tape {size, batch_size});
        this->grad_op = use_id == prev->used() ? SET : INCREMENT;
    }

    void forward() override {
        Layer::forward();
        operations::sqrcrelu<data::GPU>(prev->dense_output.values, dense_output.values, max);
    }
    void backward() override {
        Layer::backward();
        operations::sqrcrelu_bp<data::GPU>(prev->dense_output.values,
                                        prev->dense_output.gradients,
                                        dense_output.values,
                                        dense_output.gradients,
                                        max,
                                        grad_op);
    }
};

struct LeakyReLU : public Layer {
    Layer*            prev;
    GradientOperation grad_op;
    int               use_id;
    explicit LeakyReLU(Layer* prev)
        : Layer(prev->size)
        , prev(prev) {
        use_id = prev->use();
    }

    void compile(size_t batch_size) override {
        this->compile_suboutput(batch_size, Tape {size, batch_size});
        this->grad_op = use_id == prev->used() ? SET : INCREMENT;
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
                                        dense_output.gradients,
                                        0.1f,
                                        grad_op);
    }
};
}    // namespace nn
