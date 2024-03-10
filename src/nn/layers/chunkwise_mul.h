#pragma once

namespace nn {

#include "layer.h"
#include "../../data/matrix_dense.h"
#include "../../operations/operations.h"

struct ChunkwiseMul : public Layer {
    Layer*            prev;
    int               use_id;
    GradientOperation grad_op;
    int               chunks;

    ChunkwiseMul(Layer* prev, int chunks)
        : Layer(prev->size / 2)
        , prev(prev)
        , chunks(chunks) {
        use_id = prev->use();
        ERROR(prev->size % 2 == 0);
        ERROR(prev->size % chunks == 0);
    }

    void compile(size_t batch_size) override {
        this->compile_suboutput(batch_size, Tape(size, batch_size));
        this->grad_op = this->use_id == prev->used() ? SET : INCREMENT;
    }

    void forward() override {
        size_t m = prev->size / chunks;
        size_t n = dense_output.values.n;
        for(int i = 0; i < chunks / 2; i++){
            data::DenseMatrix m1{prev->dense_output.values,
                    m            , n,
                    m * 2 * i    , 0};
            data::DenseMatrix m2{prev->dense_output.values,
                    m            , n,
                    m * 2 * i + m, 0};
            data::DenseMatrix o{dense_output.values,
                    m            , n,
                    m * i        , 0};
            operations::elemwise_mul<data::GPU>(m1, m2, o);
        }
    }
    void backward() override {

        size_t m = prev->size / chunks;
        size_t n = dense_output.values.n;
        for(int i = 0; i < chunks / 2; i++){
            data::DenseMatrix m1{prev->dense_output.values,
                                  m            , n,
                                  m * 2 * i    , 0};
            data::DenseMatrix m2{prev->dense_output.values,
                                  m            , n,
                                  m * 2 * i + m, 0};
            data::DenseMatrix o_grd{dense_output.gradients,
                                  m            , n,
                                  m * i        , 0};
            data::DenseMatrix m1_grd{prev->dense_output.gradients,
                                  m            , n,
                                  m * 2 * i    , 0};
            data::DenseMatrix m2_grd{prev->dense_output.gradients,
                                  m            , n,
                                  m * 2 * i + m, 0};
            operations::elemwise_mul_bp<data::GPU>(m1, m1_grd, m2, m2_grd, o_grd, grad_op);
        }
    }
};
}

