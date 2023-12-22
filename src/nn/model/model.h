#pragma once

#include "../losses/loss.h"
#include "../optimizer/lrschedule.h"
#include "../optimizer/optimizer.h"
#include "quantization.h"

#include <filesystem>
#include <type_traits>

namespace nn {

struct Model {

    using LayerPtr = std::shared_ptr<Layer>;
    using InputPtr = std::shared_ptr<Input>;
    using LossPtr  = std::shared_ptr<Loss>;
    using OptiPtr  = std::shared_ptr<Optimizer>;
    using LRSPtr   = std::shared_ptr<LRSchedule>;

    std::vector<LayerPtr>  m_layers {};
    std::vector<InputPtr>  m_inputs {};
    std::vector<OptiPtr>   m_optimizers {};
    std::vector<Quantizer> m_quantizers {};
    LossPtr                m_loss {};
    LRSPtr                 m_lr_schedule {};
    CSVWriter              m_csv {};

    size_t                 m_epoch          = 1;
    size_t                 m_save_frequency = 50;
    std::filesystem::path  m_path;

    Model() {
        m_layers.reserve(1024);
        m_inputs.reserve(1024);
    }

    // ----------------------------------------------------------------------------------------------
    // functions to set up the model
    // ----------------------------------------------------------------------------------------------

    // add a layer to the model
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

    // sets the loss of the model
    template<typename LT>
    void set_loss(const LT& loss) {
        static_assert(std::is_base_of_v<Loss, LT>, "Invalid argument type for set_loss function");
        LossPtr ptr  = std::make_shared<LT>(loss);
        this->m_loss = ptr;
    }

    // adds an optimiser
    template<typename LT>
    void add_optimizer(const LT& opt) {
        static_assert(std::is_base_of_v<Optimizer, LT>, "LT must be derived from Optimizer.");
        std::shared_ptr<LT> ptr = std::make_shared<LT>(opt);
        m_optimizers.emplace_back(ptr);
    }

    // sets the lr schedule. the lr schedule feeds into the optimisers
    template<typename LT>
    void set_lr_schedule(const LT& lr) {
        static_assert(std::is_base_of_v<LRSchedule, LT>, "LT must be derived from LRSchedule.");
        std::shared_ptr<LT> ptr = std::make_shared<LT>(lr);
        m_lr_schedule           = ptr;
    }

    // sets the default output path where quantised networks etc will be placed
    void set_file_output(const std::string& outpath) {
        m_path = outpath;
        std::filesystem::create_directories(outpath);
        m_csv.open((std::filesystem::path(m_path) / std::filesystem::path("loss.csv")).string());
        m_csv.write("epoch", "training loss", "validation loss");
    }

    // adds a quantization
    void add_quantization(const Quantizer& quantizer) {
        m_quantizers.push_back(quantizer);
        m_quantizers.back().set_path(m_path.string());
    }

    // sets the frequency at which states will be saved
    void set_save_frequency(size_t frequency) {
        ERROR(frequency > 0);
        m_save_frequency = frequency;
    }

    // ----------------------------------------------------------------------------------------------
    // compilation
    // ----------------------------------------------------------------------------------------------
    void compile(int batch_size) {
        ERROR(this->m_loss != nullptr);
        ERROR(this->m_inputs.size());
        ERROR(this->m_layers.size());

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

    // ----------------------------------------------------------------------------------------------
    // training
    // ----------------------------------------------------------------------------------------------
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

    void reset_loss() {
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

    float loss_of_last_batch() {
        this->m_loss->loss >> data::CPU;
        return this->m_loss->loss.get(0);
    }

    float batch() {
        // step 1: sync with cpu and upload inputs and reset loss
        reset_loss();
        upload_target();
        upload_inputs();

        // step 2: forward
        forward();

        // step 3: loss function
        this->m_loss->apply(this->m_layers.back()->dense_output);
        float batch_loss = loss_of_last_batch();

        // step 4: backwards
        backward();

        // step 5: optimize
        float lr = m_lr_schedule.get()->get_lr(m_epoch);
        for (auto& opt : m_optimizers) {
            opt->step(lr);
        }

        return batch_loss;
    }

    float loss() {
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

    void next_epoch(float epoch_loss, float validation_loss = 0.0) {
        // quantitize weights
        quantize();
        quantize("latest.net");
        write_epoch_result(epoch_loss, validation_loss);
        // save weights
        if (m_epoch % m_save_frequency == 0)
            save_weights(this->m_path / "weights" / (std::to_string(m_epoch) + ".state"));
        save_weights(this->m_path / "weights" / "latest.state");

        m_epoch++;
    }

    // ----------------------------------------------------------------------------------------------
    // saving and writing results
    // ----------------------------------------------------------------------------------------------
    void quantize(const std::string& name = "") {
        for (auto& h : m_quantizers) {
            if (name == "") {
                h.save(m_epoch);
            } else {
                h.save(name);
            }
        }
    }

    void write_epoch_result(float train_loss, float validation_loss = 0.0) {
        this->m_csv.write(m_epoch, train_loss, validation_loss);
    }

    void save_weights(const std::filesystem::path& name) {

        std::filesystem::create_directory(absolute(name).parent_path());
        std::ofstream file(name, std::ios::binary);
        // layers
        for (auto& l : m_layers) {
            for (auto* p : l->params()) {
                p->values >> data::CPU;
                file.write(reinterpret_cast<const char*>(p->values.first<data::CPU>()),
                           p->values.m * p->values.n * sizeof(float));
            }
        }

        file.close();
    }

    void load_weights(const std::filesystem::path& name) {

        // FILE* f = fopen(name.string().c_str(), "rb");

        // // figure out how many entries we will store
        // uint64_t count = 0;
        // for (auto& l : m_layers) {
        //     for (Tape* t : l->params()) {
        //         count += t->values.size();
        //     }
        // }

        // uint64_t fileCount = 0;
        // fread(&fileCount, sizeof(uint64_t), 1, f);
        // ASSERT(count == fileCount);

        // for (auto& l : m_layers) {
        //     for (Tape* t : l->params()) {
        //         fread(t->values.address<data::CPU>(), sizeof(float), t->values.size(), f);
        //         t->values >> data::GPU;
        //     }
        // }
        // fclose(f);

        std::ifstream file(name, std::ios::binary);
        // layers
        for(auto& l:m_layers){
            for(auto* p:l->params()){
                file.read(reinterpret_cast<char*>(p->values.first<data::CPU>()), p->values.m *
                p->values.n * sizeof(float)); p->values >> data::GPU;
            }
        }

        file.close();
    }
};

}    // namespace nn