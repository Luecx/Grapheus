//
// Created by finne on 07.04.2023.
//

#pragma once
namespace nn {

struct SelectSingle : Layer {

    Layer* indices;

    Layer*            prev;
    int               use_id;
    GradientOperation grad_op;
    size_t choices;

    SelectSingle(Layer* prev, Layer* indices, size_t choices)
        : Layer(prev->size / choices)
        , prev(prev)
        , indices(indices)
        , choices(choices) {
        use_id = prev->use();
        ERROR(prev->size % choices == 0);
    }

    void compile(size_t batch_size) override {
        this->compile_suboutput(batch_size, Tape {size, batch_size});
        this->grad_op = this->use_id == prev->used() ? SET : INCREMENT;
    }

    void forward() override {
        operations::select_single<data::GPU>(prev->dense_output.values,
                                             dense_output.values,
                                             indices->dense_output.values);
    }
    void backward() override {
        operations::select_single_bp<data::GPU>(prev->dense_output.gradients,
                                                dense_output.gradients,
                                                indices->dense_output.values,
                                                grad_op);
    }
};

}    // namespace nn
