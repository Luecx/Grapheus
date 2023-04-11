
#include "data/array.h"
#include "data/device.h"
#include "data/matrix.h"
#include "data/matrix_dense.h"
#include "data/sarray.h"
#include "math/functions.h"
#include "math/random.h"
#include "misc/start.h"
#include "misc/timer.h"
#include "nn/layers.h"
#include "nn/tape.h"
#include "operations/function/function.h"
#include "operations/operations.h"

#include <iostream>

using namespace nn;
using namespace data;

int main() {
    init();

    Tape inp{6, 4};
    Tape out{2, 4};
    Tape idx{1, 4};

    inp.malloc();
    out.malloc();
    idx.malloc();

    math::normal(inp.values, 0.f, 1.0f);
    math::normal(inp.gradients, 0.f, 1.0f);
    math::normal(out.gradients, 0.f, 1.0f);
    math::uniform(idx.values, 0.f, 2.f);

    idx.values = math::round(idx.values);
    idx.values >> GPU;
    inp.gradients >> GPU;
    out.gradients >> GPU;

//    operations::select_single<CPU>(inp.values, out.values, idx.values);
    operations::select_single_bp<GPU>(inp.gradients, out.gradients, idx.values);

    inp.gradients >> CPU;
    out.gradients >> CPU;

    std::cout << idx.values << std::endl;
    std::cout << inp.gradients << std::endl;
    std::cout << out.gradients << std::endl;
//
//    DenseInput i1{4};
//    DenseInput i2{5};
//    Merge m{&i1,&i2};
//
//    size_t B = 16384;
//    size_t I = 40200;
//    size_t O = 512;
//
//    size_t E = 40;
//
//    SparseMatrix mat{I, B, E*3};
//    DenseMatrix<float> wgt{O, I};
//    DenseMatrix<float> bias{O,1};
//    DenseMatrix<float> output{O, B};
//
//    DenseMatrix<float> rand_counts{B,1};
//    rand_counts.malloc<CPU>();
//    math::normal(rand_counts, (float)E, 0.1f);
//
//    mat.malloc();
//    wgt.malloc<BOTH>();
//    bias.malloc<BOTH>();
//    output.malloc<BOTH>();
//
//    for(int i = 0; i < O; i++){
//        for(int j = 0; j < I; j++){
//            wgt(i,j) = i + j;
//        }
//    }
//
//    for(int i = 0; i <  B; i++){
//        for(int j = 0; j < rand_counts(i,0); j++){
//            mat.set(i,rand() % I);
//        }
//    }
//
//    wgt >> GPU;
//    mat.values >> GPU;
//
//    Timer t{};
//    t.start();
//
//    for(int i = 0; i < 100; i++){
//        operations::affine_sparse<data::GPU>(wgt, mat, bias, output);
//    }
//
//    output >> CPU;
//    t.stop();
//
//    //    std::cout << wgt << std::endl;
//    //    std::cout << mat << std::endl;
//    //    std::cout << output << std::endl;
//
//    std::cout << "estimated speed: " << std::setprecision(4) << (1638.40f / t.elapsed())
//              << " Mn/s" << std::endl;
//    std::cout << t.elapsed() << std::endl;

    close();
}
