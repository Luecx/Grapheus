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

    std::vector<LayerPtr> m_layers {};
    std::vector<InputPtr> m_inputs {};
    std::vector<OptiPtr>  m_optimizers {};
    LossPtr               m_loss {};
    LRSPtr                m_lr_schedule {};

    size_t epoch = 0;

    Model() {
        m_layers.reserve(1024);
        m_inputs.reserve(1024);
    }

    template<typename LT, typename... ARGS>
    LT* add(ARGS&&... args) {
        std::shared_ptr<LT> ptr = std::make_shared<LT>(std::forward<ARGS>(args)...);
        static_assert(std::is_base_of_v<Input, LT> || std::is_base_of_v<Layer, LT>,
                      "Invalid argument type for add function");
        if constexpr (std::is_base_of_v<Input, LT>) {
            m_inputs.emplace_back(ptr);
        } else {
            m_layers.emplace_back(ptr);
        }
        return ptr.get();
    }

    template<typename LT>
    void set_loss(const LT& loss) {
        static_assert(std::is_base_of_v<Loss, LT>, "Invalid argument type for set_loss function");
        LossPtr ptr = std::make_shared<LT>(loss);
        this->m_loss = ptr;
    }

    template<typename LT>
    void add_optimizer(const LT& opt) {
        static_assert(std::is_base_of_v<Optimizer, LT>, "LT must be derived from Optimizer.");
        std::shared_ptr<LT> ptr = std::make_shared<LT>(opt);
        m_optimizers.emplace_back(ptr);
    }

    template<typename LT>
    void set_lr_schedule(const LT& lr) {
        static_assert(std::is_base_of_v<LRSchedule, LT>, "LT must be derived from LRSchedule.");
        std::shared_ptr<LT> ptr = std::make_shared<LT>(lr);
        m_lr_schedule             = ptr;
    }

    void compile(int batch_size) {
        this->m_loss.get()->compile(m_layers.back().get(), batch_size);

        for (auto& l : m_inputs) {
            l->compile(batch_size);
        }

        for (auto& l : m_layers) {
            l->compile(batch_size);
        }

        for (auto& l : m_optimizers) {
            l->compile();
        }
    }

    void upload_inputs() {
        for (auto& l : m_inputs) {
            if (l->output_type() == DENSE) {
                l->dense_output.values >> data::GPU;
            } else {
                l->sparse_output.values >> data::GPU;
            }
        }
    }

    void upload_target() {
        this->m_loss->target >> data::GPU;
    }

    void reset_loss(){
        this->m_loss->loss[0] = 0;
        this->m_loss->loss >> data::GPU;
    }

    void forward() {
        for (auto& l : m_layers) {
            l->forward();
        }
    }

    void backward() {
        for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
            (*it)->backward();
        }
    }

    float loss_of_last_batch(){
        this->m_loss->loss >> data::CPU;
        return this->m_loss->loss.get(0);
    }

    void batch(){
        // step 1: sync with cpu and upload inputs and reset loss
        reset_loss();
        upload_target();
        upload_inputs();

        // step 2: forward
        forward();

        // step 3: loss function
        this->m_loss->apply(this->m_layers.back()->dense_output);

        // step 4: backwards
        backward();

        // step 5: optimize
        float lr = m_lr_schedule.get()->get_lr(epoch);
        for(auto& opt:m_optimizers){
            opt->step(lr);
        }
    }

    float loss(){
        // step 1: sync with cpu and upload inputs and reset loss
        upload_target();
        reset_loss();
        upload_inputs();

        // step 2: forward
        forward();

        // step 3: loss function
        this->m_loss->apply(this->m_layers.back()->dense_output);

        // step 5: sync with cpu and return loss
        return loss_of_last_batch();
    }

    void next_epoch(){
        epoch++;
    }
};

}    // namespace nn