//
// Created by finne on 07.04.2023.
//

#pragma once
namespace nn {

struct Select : Layer {

    Layer*              indices;
    std::vector<Layer*> heads {};

    // pointer to the heads
    data::SArray<float*> input_pointers {0};
    data::SArray<float*> gradient_pointers {0};

    Select(std::vector<Layer*> prev, Layer* indices)
        : Layer(prev[0]->size)
        , heads(prev)
        , indices(indices)
        , input_pointers(prev.size())
        , gradient_pointers(prev.size()) {

        for(Layer* l:prev){
            l->use();
        }

        this->input_pointers.malloc<data::BOTH>();
        this->gradient_pointers.malloc<data::BOTH>();
    }

    void compile(size_t batch_size) override {
        for (int i = 0; i < heads.size(); i++) {
            input_pointers[i]    = heads[i]->dense_output.values.first<data::GPU>();
            gradient_pointers[i] = heads[i]->dense_output.gradients.first<data::GPU>();
        }
        input_pointers    >> data::GPU;
        gradient_pointers >> data::GPU;

        this->dense_output = Tape(size, batch_size);
        this->dense_output.malloc();
    }

    void forward() override {
        // CANNOT SWITCH TO CPU HERE BECAUSE OUTPUT_POINTERS ONLY HAS GPU VALUES!
        operations::select<data::GPU>(input_pointers,
                                      dense_output.values,
                                      indices->dense_output.values);
    }
    void backward() override {
        // CANNOT SWITCH TO CPU HERE BECAUSE OUTPUT_POINTERS ONLY HAS GPU VALUES!
        operations::select_bp<data::GPU>(gradient_pointers,
                                         dense_output.gradients,
                                         indices->dense_output.gradients);
    }
};

}    // namespace nn

