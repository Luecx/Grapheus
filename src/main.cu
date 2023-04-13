
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
    auto ft = model.add<FeatureTransformer>(i1, i2, 512);
    auto rl = model.add<ReLU>(ft);
    auto s1 = model.add<AffineMulti>(rl, 16, 8);
    auto a3 = model.add<Sigmoid>(s1, 2.5 / 400);

    model.set_loss(MSE {true});
    model.set_lr_schedule(StepDecayLRSchedule{0.001,1,1});
    model.add_optimizer(Adam({
             {OptimizerEntry{&ft->weights}.min(-5).max(5)},
             {OptimizerEntry{&ft->bias   }.min(-5).max(5)},
             {OptimizerEntry{&s1->weights}.min(-5).max(5)},
             {OptimizerEntry{&s1->bias   }.min(-5).max(5)},
    }, 0.9, 0.999, 1e-7));

    model.compile(8);

    // setup input
    for(int i = 0; i < 8; i++){
        int offset = 12 * 64 * (rand() % 5);
        for(int j = 0; j < 10 + (rand() % 50); j++){
            i1->sparse_output.set(i, (rand() % (12 * 64)) + offset);
            i2->sparse_output.set(i, (rand() % (12 * 64)) + offset);
        }
    }

    DenseMatrix<float> target{a3->size,8};
    target.malloc<BOTH>();
    math::uniform( target, 0.4f, 0.6f);
    target >> GPU;

    for(int k = 0; k < 1; k++){
        Timer t{};
        t.start();

        for(int i = 0; i < 1000; i++){
            std::cout << model.batch(target) << std::endl;
        }

        t.stop();

        std::cout << (int)(8 * 1e5 / t.elapsed()) << std::endl;
    }
//    s2->dense_output.values >> CPU;
//    a2->dense_output.values >> CPU;
//    s3->weights.values >> CPU;
//    s3->bias.values >> CPU;

//    std::cout << s2->dense_output.values << std::endl;
//    std::cout << a2->dense_output.values << std::endl;
//    std::cout << s3->weights.values << std::endl;
//    std::cout << s3->bias.values << std::endl;

    close();
}