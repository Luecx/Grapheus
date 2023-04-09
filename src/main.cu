
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

    DenseInput inp{2048};
    AffineMulti   affine_multi   {&inp            , 16    , 8};
    ClippedRelu   c1             {&affine_multi};
    AffineBatched affine_batched1{&c1             , 32 * 8, 8};
    AffineBatched affine_batched3{&affine_batched1,      8, 8};

    inp.compile(16384);
    c1.compile(16384);
    affine_multi.compile(16384);
    affine_batched1.compile(16384);
    affine_batched3.compile(16384);

    Timer t{};
    t.start();

    for(int i = 0; i < 100; i++){
        affine_multi.forward();
        c1.forward();
        affine_batched1.forward();
        affine_batched3.forward();

        affine_batched3.backward();
        affine_batched1.backward();
        c1.backward();
        affine_multi.backward();
    }

    affine_batched3.dense_output.values >> CPU;
    t.stop();

    std::cout << "estimated speed: "<< std::fixed << (int)(1e5 * 16384 / t.elapsed()) << "n/s" << std::endl;
    std::cout << t.elapsed() << std::endl;




    close();
}
