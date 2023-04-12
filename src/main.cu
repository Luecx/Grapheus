
#include "misc/timer.h"
#include "nn/nn.h"
#include "operations/operations.h"

using namespace nn;
using namespace data;

int main() {
    init();

    Model model{};


    auto i1 = model.add<SparseInput>(12 * 64 * 16, 64);
    auto i2 = model.add<SparseInput>(12 * 64 * 16, 64);
    auto i3 = model.add<DenseInput>(1);
    auto ft = model.add<FeatureTransformer>(i1, i2, 512);
    auto rl = model.add<ReLU>(ft);
    auto s1 = model.add<AffineMulti>(rl, 16, 8);
    auto a1 = model.add<ReLU>(s1);
    auto s2 = model.add<AffineBatched>(a1, 32 * 8, 8);
    auto a2 = model.add<ReLU>(s2);
    auto s3 = model.add<AffineBatched>(a2, 1 * 8, 8);
    auto sl = model.add<SelectSingle>(s3, i3, 8);
    auto a3 = model.add<Sigmoid>(sl, 2.5 / 400);

    model.set_loss(MSE {true});
    model.set_lr_schedule(StepDecayLRSchedule{1,1,1});
    model.add_optimizer(Adam({
             {OptimizerEntry{&ft->weights}},
             {OptimizerEntry{&ft->bias   }},
             {OptimizerEntry{&s1->weights}},
             {OptimizerEntry{&s1->bias   }},
             {OptimizerEntry{&s2->weights}},
             {OptimizerEntry{&s2->bias   }},
             {OptimizerEntry{&s3->weights}},
             {OptimizerEntry{&s3->bias   }},
    }, 0.9, 0.999, 1e-7));

    model.compile(16384);

    // setup input
    for(int i = 0; i < 16384; i++){
        int offset = 12 * 64 * (rand() % 16);
        for(int j = 0; j < 10 + (rand() % 50); j++){
            i1->sparse_output.set(i, (rand() % (12 * 64)) + offset);
            i2->sparse_output.set(i, (rand() % (12 * 64)) + offset);
        }
    }

    DenseMatrix<float> target{a3->size,16384};
    target.malloc<BOTH>();
    math::uniform( target, 0.f, 1.f);
    target >> GPU;

    for(int k = 0; k < 10; k++){
        Timer t{};
        t.start();

        for(int i = 0; i < 1000; i++){
            model.batch(target);
        }

        t.stop();

        std::cout << (int)(16384 * 1e6 / t.elapsed()) << std::endl;
    }


    close();
}