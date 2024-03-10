//
// Created by finne on 07.04.2023.
//

#pragma once
namespace nn {

struct Select : Layer {

    Layer*              indices;
    std::vector<Layer*> heads {};
    std::vector<int   > use_ids {};

    // pointer to the heads
    data::SArray<float*> input_pointers {0};
    data::SArray<float*> gradient_pointers {0};
    data::SArray<GradientOperation> grad_ops {0};

    Select(std::vector<Layer*> prev, Layer* indices)
        : Layer(prev[0]->size)
        , heads(prev)
        , indices(indices)
        , input_pointers(prev.size())
        , gradient_pointers(prev.size())
        , grad_ops(prev.size()){

        for(int i = 0; i< prev.size(); i++){
            use_ids[i] = prev[i]->use();
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

        for(int i = 0; i < heads.size(); i++){
            grad_ops[i] = this->use_ids[i] == heads[i]->used() ? SET : INCREMENT;
        }

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
                                         indices->dense_output.gradients,
                                         grad_ops);
    }
};

}    // namespace nn

