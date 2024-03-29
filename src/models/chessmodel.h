#pragma once

#include "../chess/chess.h"
#include "../dataset/batchloader.h"
#include "../dataset/dataset.h"
#include "../dataset/io.h"
#include "../dataset/process.h"
#include "../misc/csv.h"
#include "../misc/timer.h"
#include "../nn/nn.h"
#include "../operations/operations.h"

#include <algorithm>
#include <optional>

namespace model {

using namespace nn;
using namespace data;

struct ChessModel : Model {

    /**
     * @brief Versatile base class for Chess models with training setup.
     * Override `setup_inputs_and_outputs` to define inputs.
     * `lambda`: CP Score to WDL ratio.
     */

    float lambda;

    ChessModel(float lambda_)
        : lambda(lambda_) {}

    /**
     * @brief Set up inputs and outputs for the model.
     * Override this function to define inputs for the chess model.
     * @param positions Pointer to the dataset of chess positions.
     * This function should be implemented in derived classes to specify how inputs and outputs are
     * prepared.
     */
    virtual void setup_inputs_and_outputs(dataset::DataSet<chess::Position>* positions) = 0;

    using BatchLoader = dataset::BatchLoader<chess::Position>;

    /**
     * @brief Trains the model using the provided train and validation loaders for a specified number
     * of epochs.
     *
     * @param train_loader The batch loader for training data.
     * @param val_loader The batch loader for validation data (optional).
     * @param epochs Number of training epochs (default: 1500).
     * @param epoch_size Number of batches per epoch (default: 1e8).
     */
    void train(BatchLoader&                train_loader,
               std::optional<BatchLoader>& val_loader,
               int                         epochs         = 1500,
               int                         epoch_size     = 1e8,
               int                         val_epoch_size = 1e7) {

        this->compile(train_loader.batch_size);

        Timer t {};
        for (int i = 1; i <= epochs; i++) {
            t.start();

            uint64_t prev_print_tm    = 0;
            float    total_epoch_loss = 0;
            float    total_val_loss   = 0;

            // Training phase
            for (int b = 1; b <= epoch_size / train_loader.batch_size; b++) {
                auto* ds = train_loader.next();
                setup_inputs_and_outputs(ds);

                float batch_loss = batch();
                total_epoch_loss += batch_loss;
                float epoch_loss = total_epoch_loss / b;

                t.stop();
                uint64_t elapsed = t.elapsed();
                if (elapsed - prev_print_tm > 1000 || b == epoch_size / train_loader.batch_size) {
                    prev_print_tm = elapsed;

                    printf("\rep = [%4d], epoch_loss = [%1.8f], batch = [%5d], batch_loss = [%1.8f], "
                           "speed = [%7.2f it/s], time = [%3ds]",
                           i,
                           epoch_loss,
                           b,
                           batch_loss,
                           1000.0f * b / elapsed,
                           (int) (elapsed / 1000.0f));
                    std::cout << std::flush;
                }
            }

            // Validation phase (if validation loader is provided)
            if (val_loader.has_value()) {
                for (int b = 1; b <= val_epoch_size / val_loader->batch_size; b++) {
                    auto* ds = val_loader->next();
                    setup_inputs_and_outputs(ds);

                    float val_batch_loss = loss();
                    total_val_loss += val_batch_loss;
                }
            }

            float epoch_loss = total_epoch_loss / (val_epoch_size / train_loader.batch_size);
            float val_loss   = (val_loader.has_value())
                                   ? total_val_loss / (val_epoch_size / val_loader->batch_size)
                                   : 0;

            printf(", val_loss = [%1.8f] ", val_loss);
            next_epoch(epoch_loss, val_loss);
            std::cout << std::endl;
        }
    }

    /// @brief Predict output and display layer activations for a chess position.
    /// @param fen FEN string of the chess position.
    /// Compiles the model, sets up inputs, performs a forward pass, and prints layer activations.
    void test_fen(const std::string& fen) {
        this->compile(1);

        chess::Position                   pos = chess::parse_fen(fen);
        dataset::DataSet<chess::Position> ds {};
        ds.positions.push_back(pos);
        ds.header.entry_count = 1;

        // setup inputs of network
        setup_inputs_and_outputs(&ds);

        // forward pass
        this->upload_inputs();
        this->forward();

        // go through the layers and download values

        std::cout
            << "==================================================================================\n";
        std::cout << "testing fen: " << fen << std::endl;

        int idx = 0;
        for (auto layer : m_layers) {
            layer->dense_output.values >> data::CPU;

            std::cout << "LAYER " << ++idx << std::endl;
            for (int i = 0; i < std::min((size_t) 16, layer->size); i++) {
                std::cout << std::setw(10) << layer->dense_output.values(i, 0);
            }
            if (layer->size > 16) {
                std::cout << " ......... " << layer->dense_output.values(layer->size - 1, 0);
            }
            std::cout << "\n";
        }
    }

    /// @brief Display layer output statistics.
    /// Computes and prints min/max values, sparsity, and parameter bounds for each layer.
    /// @param loader Batch loader for dataset input.
    /// @param batches Number of batches for processing.
    /// Iterates through batches, computes layer output stats, and prints to console.
    void distribution(dataset::BatchLoader<chess::Position>& loader, int batches = 32) {
        this->compile(loader.batch_size);

        using namespace data;

        std::vector<DenseMatrix<float>>            max_values {};
        std::vector<DenseMatrix<float>>            min_values {};
        std::vector<std::pair<uint64_t, uint64_t>> sparsity {};

        for (auto l : m_layers) {
            max_values.emplace_back(l->dense_output.values.m, 1);
            min_values.emplace_back(l->dense_output.values.m, 1);
            max_values.back().malloc<data::CPU>();
            min_values.back().malloc<data::CPU>();

            math::fill<float>(max_values.back(), -std::numeric_limits<float>::max());
            math::fill<float>(min_values.back(), std::numeric_limits<float>::max());

            sparsity.push_back(std::pair(0, 0));
        }

        for (int b = 0; b < batches; b++) {
            auto* ds = loader.next();
            setup_inputs_and_outputs(ds);
            this->upload_inputs();
            this->forward();
            std::cout << "\r" << b << " / " << batches << std::flush;

            // get minimum and maximum values
            for (int i = 0; i < m_layers.size(); i++) {
                auto layer = m_layers[i].get();
                layer->dense_output.values >> data::CPU;
                for (int m = 0; m < layer->dense_output.values.m; m++) {
                    for (int n = 0; n < layer->dense_output.values.n; n++) {
                        max_values[i](m, 0) =
                            std::max(max_values[i](m, 0), layer->dense_output.values(m, n));
                        min_values[i](m, 0) =
                            std::min(min_values[i](m, 0), layer->dense_output.values(m, n));

                        sparsity[i].first++;
                        sparsity[i].second += (layer->dense_output.values(m, n) > 0);
                    }
                }
            }
        }
        std::cout << std::endl;

        for (int i = 0; i < m_layers.size(); i++) {
            std::cout << "------------ LAYER " << i + 1 << " --------------------" << std::endl;
            std::cout << "min: ";
            for (int j = 0; j < std::min((size_t) 16, min_values[i].size()); j++) {
                std::cout << std::setw(10) << min_values[i](j);
            }
            if (min_values[i].size() > 16) {
                std::cout << " ......... " << min_values[i](min_values.size() - 1);
            }
            std::cout << "\n";

            std::cout << "max: ";
            for (int j = 0; j < std::min((size_t) 16, max_values[i].size()); j++) {
                std::cout << std::setw(10) << max_values[i](j);
            }
            if (max_values[i].size() > 16) {
                std::cout << " ......... " << max_values[i](max_values.size() - 1);
            }

            std::cout << "\n";
            float min = 10000000;
            float max = -10000000;
            for (int m = 0; m < min_values.size(); m++) {
                min = std::min(min, min_values[i](m));
                max = std::max(max, max_values[i](m));
            }
            std::cout << "output bounds: [" << min << " ; " << max << "]\n";

            int died = 0;
            for (int j = 0; j < max_values[i].size(); j++) {
                if (std::abs(max_values[i](j) - min_values[i](j)) < 1e-8) {
                    died++;
                }
            }

            std::cout << "died: " << died << " / " << max_values[i].size();
            std::cout << "\n";

            float sparsity_total  = sparsity[i].first;
            float sparsity_active = sparsity[i].second;

            std::cout << "sparsity: " << sparsity_active / sparsity_total;
            std::cout << "\n";

            for (auto p : m_layers[i]->params()) {
                float min = 10000000;
                float max = -10000000;
                for (int m = 0; m < p->values.m; m++) {
                    for (int n = 0; n < p->values.n; n++) {
                        min = std::min(min, p->values(m, n));
                        max = std::max(max, p->values(m, n));
                    }
                }

                std::cout << "param bounds: [" << min << " ; " << max << "]\n";
            }
        }
    }
};
}    // namespace model