#pragma once
#include "layer.h"

namespace nn {

/**
 * general class for inputs
 * its an abstract class and features the sparse input and the dense input below
 * It does not use any previous layer and serves as a buffer to store either dense
 * or sparse data to be used in the following operations of the network
 */
struct Input : public Layer {
    explicit Input(size_t size)
        : Layer(size) {}

    LayerOutputType input_type() override {
        return DENSE_AND_SPARSE;
    }
};

/**
 * sparse inputs based on the generalised Input class as described above
 * Takes the size (relevant for following layers) as well as the maximum amount of enabled inputs
 * per input. Does only support binary inputs so far. The data format is a custom format which works
 * ideal with the custom kernels and is easy to use. Furthermore its not possible to extract gradients
 * of the sparse features
 */
struct SparseInput : public Input {
    size_t max_inputs;
    SparseInput(size_t size, size_t maxInputs)
        : Input(size)
        , max_inputs(maxInputs) {}

    LayerOutputType output_type() override {
        return SPARSE;
    }
    void compile(size_t batch_size) override {
        this->sparse_output = data::SparseMatrix(size, batch_size, max_inputs);
        this->sparse_output.malloc();
    }
};

/**
 * dense input based on the generalised Input class as described above
 * Takes the size (relevant for following layers). The data is stored in dense matrices once compile()
 * has been called. It is possible to extract gradients of the input features
 */
struct DenseInput : public Input {
    explicit DenseInput(size_t size)
        : Input(size) {}

    LayerOutputType output_type() override {
        return DENSE;
    }
    void compile(size_t batch_size) override {
        this->dense_output = Tape(size, batch_size);
        this->dense_output.malloc();
    }
};

}    // namespace nn

