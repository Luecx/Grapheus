#pragma once

#include "../operations/operations.h"
#include "layer.h"

struct AffineBatched : public nn::Layer {

    size_t     batches;
    nn::Layer* prev;

    nn::Tape   weights {0, 0};
    nn::Tape   bias {0, 0};

    // pointers to all the weights, gradients and inputs etc
    data::SArray<float*> wgt_ptr;
    data::SArray<float*> wgt_grd_ptr;
    data::SArray<float*> inp_ptr;
    data::SArray<float*> inp_grd_ptr;
    data::SArray<float*> bia_ptr;
    data::SArray<float*> bia_grd_ptr;
    data::SArray<float*> out_ptr;
    data::SArray<float*> out_grd_ptr;

    AffineBatched(Layer* prev, size_t size, size_t batches)
        : Layer(size)
        , batches(batches)
        , prev(prev)
        , wgt_ptr(batches)
        , wgt_grd_ptr(batches)
        , inp_ptr(batches)
        , inp_grd_ptr(batches)
        , bia_ptr(batches)
        , bia_grd_ptr(batches)
        , out_ptr(batches)
        , out_grd_ptr(batches) {
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

        // setup the pointers
        wgt_ptr    .malloc<data::BOTH>();
        wgt_grd_ptr.malloc<data::BOTH>();
        inp_ptr    .malloc<data::BOTH>();
        inp_grd_ptr.malloc<data::BOTH>();
        bia_ptr    .malloc<data::BOTH>();
        bia_grd_ptr.malloc<data::BOTH>();

        for(int i = 0; i < wgt_ptr.size(); i++){
            wgt_ptr    [i] = weights.values              .first<data::GPU>() + i * size / batches;
            wgt_grd_ptr[i] = weights.gradients           .first<data::GPU>() + i * size / batches;
            inp_ptr    [i] = prev->dense_output.values   .first<data::GPU>() + i * size / batches;
            inp_grd_ptr[i] = prev->dense_output.gradients.first<data::GPU>() + i * size / batches;
            bia_ptr    [i] = bias.values                 .first<data::GPU>() + i * size / batches;
            bia_grd_ptr[i] = bias.gradients              .first<data::GPU>() + i * size / batches;
        }

        wgt_ptr     >> data::GPU;
        wgt_grd_ptr >> data::GPU;
        inp_ptr     >> data::GPU;
        inp_grd_ptr >> data::GPU;
        bia_ptr     >> data::GPU;
        bia_grd_ptr >> data::GPU;
        // clang-format on
    }

    void compile_suboutput(size_t batch_size, const nn::Tape& output) override {
        Layer::compile_suboutput(batch_size, output);
        // clang-format off
        out_ptr    .malloc<data::BOTH>();
        out_grd_ptr.malloc<data::BOTH>();

        for(int i = 0; i < wgt_ptr.size(); i++){
            out_ptr    [i] = dense_output.values   .first<data::GPU>() + i * size / batches;
            out_grd_ptr[i] = dense_output.gradients.first<data::GPU>() + i * size / batches;
        }
        out_ptr     >> data::GPU;
        out_grd_ptr >> data::GPU;
        // clang-format on
    }

    void forward() override {
        Layer::forward();
        operations::affine_batched<data::GPU>(inp_ptr,
                                              wgt_ptr,
                                              bia_ptr,
                                              out_ptr,
                                              size / batches,
                                              dense_output.values.n,
                                              prev->size / batches,
                                              prev->dense_output.values.ld,
                                              weights.values.ld,
                                              dense_output.values.ld);
    }
    void backward() override {
        Layer::backward();

        operations::affine_batched_bp<data::GPU>(inp_ptr,
                                                 inp_grd_ptr,
                                                 wgt_ptr,
                                                 wgt_grd_ptr,
                                                 bia_grd_ptr,
                                                 out_grd_ptr,
                                                 size / batches,
                                                 dense_output.values.n,
                                                 prev->size / batches,
                                                 prev->dense_output.values.ld,
                                                 weights.values.ld,
                                                 dense_output.values.ld);
    }
};