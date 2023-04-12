#pragma once

#include "../losses/loss.h"
#include "../optimizer/lrschedule.h"
#include "../optimizer/optimizer.h"

#include <type_traits>

namespace nn {

struct Model {

    using LayerPtr = std::shared_ptr<Layer>;
    using InputPtr = std::shared_ptr<Input>;
    using LossPtr  = std::shared_ptr<Loss>;
    using OptiPtr  = std::shared_ptr<Optimizer>;
    using LRSPtr   = std::shared_ptr<LRSchedule>;

    std::vector<LayerPtr> layers {};
    std::vector<InputPtr> inputs {};
    std::vector<OptiPtr>  optimizers {};
    LossPtr               loss {};
    LRSPtr                lr_schedule {};

    size_t epoch = 0;

    Model() {
        layers.reserve(1024);
        inputs.reserve(1024);
    }

    template<typename LT, typename... ARGS>
    LT* add(ARGS&&... args) {
        std::shared_ptr<LT> ptr = std::make_shared<LT>(std::forward<ARGS>(args)...);
        static_assert(std::is_base_of_v<Input, LT> || std::is_base_of_v<Layer, LT>,
                      "Invalid argument type for add function");
        if constexpr (std::is_base_of_v<Input, LT>) {
            inputs.emplace_back(ptr);
        } else {
            layers.emplace_back(ptr);
        }
        return ptr.get();
    }

    template<typename LT>
    void set_loss(const LT& loss) {
        static_assert(std::is_base_of_v<Loss, LT>, "Invalid argument type for set_loss function");
        LossPtr ptr = std::make_shared<LT>(loss);
        this->loss = ptr;
    }

    template<typename LT>
    void add_optimizer(const LT& opt) {
        static_assert(std::is_base_of_v<Optimizer, LT>, "LT must be derived from Optimizer.");
        std::shared_ptr<LT> ptr = std::make_shared<LT>(opt);
        optimizers.emplace_back(ptr);
    }

    template<typename LT>
    void set_lr_schedule(const LT& lr) {
        static_assert(std::is_base_of_v<LRSchedule, LT>, "LT must be derived from LRSchedule.");
        std::shared_ptr<LT> ptr = std::make_shared<LT>(lr);
        lr_schedule             = ptr;
    }

    void compile(int batch_size) {
        this->loss.get()->compile();

        for (auto& l : inputs) {
            l->compile(batch_size);
        }

        for (auto& l : layers) {
            l->compile(batch_size);
        }

        for (auto& l : optimizers) {
            l->compile();
        }
    }

    void upload_inputs() {
        for (auto& l : inputs) {
            if (l->output_type() == DENSE) {
                l->dense_output.values >> data::GPU;
            } else {
                l->sparse_output.values >> data::GPU;
            }
        }
    }

    void forward() {
        for (auto& l : layers) {
            l->forward();
        }
    }

    void backward() {
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            (*it)->backward();
        }
    }

    float batch(data::DenseMatrix<float>& target){
        // step 1: sync with cpu and upload inputs and reset loss
        this->loss->loss[0] = 0;
        this->loss->loss >> data::GPU;
        upload_inputs();

        // step 2: forward
        forward();

        // step 3: loss function
        this->loss->apply(this->layers.back()->dense_output, target);

        // step 4: backwards
        backward();

        // step 5: optimize
        float lr = lr_schedule.get()->get_lr(epoch++);
        for(auto& opt:optimizers){
            opt->step(lr);
        }
        // step 5: sync with cpu and return loss
        this->loss->loss >> data::CPU;
        return this->loss->loss.get(0);
    }
};

}    // namespace nn