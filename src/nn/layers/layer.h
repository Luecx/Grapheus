#pragma once

#include "../../data/matrix_sparse.h"
#include "tape.h"

namespace nn {

enum LayerOutputType : int { DENSE = 1, SPARSE = 2, DENSE_AND_SPARSE };

struct Layer {
    size_t size;
    size_t output_used_counter = 0; // count how many times this layer is used.

    explicit Layer(size_t size)
        : size(size) {}

    // store these for every Layer
    // every Layer only uses one of them
    // depending on which one will be used, malloc is going to be called respectively
    Tape               dense_output {0, 0};
    data::SparseMatrix sparse_output {0, 0, 0};

    // basic output managing functions
    virtual LayerOutputType input_type() {
        return DENSE;
    }
    virtual LayerOutputType output_type() {
        return DENSE;
    }

    // main call to configure the layer and initialise all matrices
    // should also call compile_output which sets the output matrix
    virtual void compile(size_t batch_size) = 0;
    virtual void compile_suboutput(size_t batch_size, const Tape& output) {
        dense_output = output;
        if(!dense_output.values.is_allocated<data::BOTH>()){
            dense_output.values.malloc<data::BOTH>();
        }
        if(!dense_output.gradients.is_allocated<data::BOTH>()){
            dense_output.gradients.malloc<data::BOTH>();
        }
    }

    // computation forward and backwards
    virtual void forward() {};
    virtual void backward() {};

    // use and unuse functions for the layers
    int use() {
        output_used_counter++;
        return used();
    }
    int used() {
        return output_used_counter;
    }

    // get all the parameters related to this layer
    virtual std::vector<Tape*> params() {
        return {};
    };
};

}    // namespace nn

