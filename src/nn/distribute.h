//
// Created by finne on 07.04.2023.
//

#pragma once
#include "../operations/operations.h"
#include "layer.h"

namespace nn{

// TODO: discuss how this should be done
//
//struct DistributeHead : public Layer{
//    Layer* prev;
//
//    explicit DistributeHead(Layer* prev)
//        : prev {prev}
//        , Layer(prev->size){}
//
//    void compile(size_t batch_size) override {
//        this->dense_output = Tape {size, batch_size};
//        this->dense_output.malloc();
//    }
//};
//
//struct Distribute : Layer{
//    Layer* prev;
//    Layer* indices;
//    std::vector<DistributeHead> heads{};
//
//    // pointer to the heads
//    data::SArray<float*> output_pointers{0};
//    data::SArray<float*> gradient_pointers{0};
//
//    Distribute(Layer* prev, Layer* indices, size_t heads)
//        : Layer(prev->size * heads)
//        , prev(prev)
//        , indices(indices)
//        , output_pointers(heads)
//        , gradient_pointers(heads) {
//        for(int i = 0; i < heads; i++){
//            this->heads.emplace_back(prev);
//        }
//        this->output_pointers  .malloc<data::BOTH>();
//        this->gradient_pointers.malloc<data::BOTH>();
//    }
//
//
//
//    void compile(size_t batch_size) override {
//        for(int i = 0; i < heads.size(); i++){
//            heads            [i].compile(batch_size);
//            output_pointers  [i] = heads[i].dense_output.values   .first<data::GPU>();
//            gradient_pointers[i] = heads[i].dense_output.gradients.first<data::GPU>();
//        }
//
//        output_pointers   >> data::GPU;
//        gradient_pointers >> data::GPU;
//    }
//
//    DistributeHead* operator[](size_t s) {
//        return &heads[s];
//    }
//
//    void forward() override {
//        // CANNOT SWITCH TO CPU HERE BECAUSE OUTPUT_POINTERS ONLY HAS GPU VALUES!
//        operations::distribute<data::GPU>(prev->dense_output.values,
//                                          output_pointers,
//                                          indices->dense_output.values);
//    }
//    void backward() override {
//        // CANNOT SWITCH TO CPU HERE BECAUSE OUTPUT_POINTERS ONLY HAS GPU VALUES!
//        operations::distribute_bp<data::GPU>(prev->dense_output.gradients,
//                                             output_pointers,
//                                             indices->dense_output.gradients);
//    }
//};

}

