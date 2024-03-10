#include "argparse.hpp"
#include "chess/chess.h"
#include "dataset/batchloader.h"
#include "dataset/dataset.h"
#include "dataset/io.h"
#include "dataset/process.h"
#include "misc/csv.h"
#include "misc/timer.h"
#include "nn/nn.h"
#include "operations/operations.h"

#include <fstream>
#include <limits>

namespace fs = std::filesystem;

using namespace nn;
using namespace data;

struct ChessModel : nn::Model {
    float lambda;

    ChessModel(float lambda_)
        : lambda(lambda_) {}

    // seting inputs
    virtual void setup_inputs_and_outputs(dataset::DataSet<chess::Position>* positions) = 0;

    // train function
    void train(dataset::BatchLoader<chess::Position>& loader,
               int                                    epochs     = 1500,
               int                                    epoch_size = 1e8) {
        this->compile(loader.batch_size);

        Timer t {};
        for (int i = 1; i <= epochs; i++) {
            t.start();

            uint64_t prev_print_tm    = 0;
            float    total_epoch_loss = 0;

            for (int b = 1; b <= epoch_size / loader.batch_size; b++) {
                auto* ds = loader.next();
                setup_inputs_and_outputs(ds);

                float batch_loss = batch();
                total_epoch_loss += batch_loss;
                float epoch_loss = total_epoch_loss / b;

                t.stop();
                uint64_t elapsed = t.elapsed();
                if (elapsed - prev_print_tm > 1000 || b == epoch_size / loader.batch_size) {
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

            std::cout << std::endl;

            float epoch_loss = total_epoch_loss / (epoch_size / loader.batch_size);
            next_epoch(epoch_loss, 0.0);
        }
    }

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
            layer->dense_output.values >> CPU;

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

    void distribution(dataset::BatchLoader<chess::Position>& loader, int batches = 32) {
        this->compile(loader.batch_size);

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

struct BerserkModel : ChessModel {
    SparseInput* in1;
    SparseInput* in2;

    const float  sigmoid_scale = 1.0 / 160.0;
    const float  quant_one     = 32.0;
    const float  quant_two     = 32.0;

    const size_t n_features    = 16 * 12 * 64;
    const size_t n_l1          = 16;
    const size_t n_l2          = 32;
    const size_t n_out         = 1;

    BerserkModel(size_t n_ft, float lambda, size_t save_rate)
        : ChessModel(lambda) {

        in1                    = add<SparseInput>(n_features, 32);
        in2                    = add<SparseInput>(n_features, 32);

        auto ft                = add<FeatureTransformer>(in1, in2, n_ft);
        auto fta               = add<ClippedRelu>(ft);
        ft->ft_regularization  = 1.0 / 16384.0 / 4194304.0;
        fta->max               = 127.0;

        auto        l1         = add<Affine>(fta, n_l1);
        auto        l1a        = add<ReLU>(l1);

        auto        l2         = add<Affine>(l1a, n_l2);
        auto        l2a        = add<ReLU>(l2);

        auto        pos_eval   = add<Affine>(l2a, n_out);
        auto        sigmoid    = add<Sigmoid>(pos_eval, sigmoid_scale);

        const float hidden_max = 127.0 / quant_two;
        add_optimizer(AdamWarmup({{OptimizerEntry {&ft->weights}},
                                  {OptimizerEntry {&ft->bias}},
                                  {OptimizerEntry {&l1->weights}.clamp(-hidden_max, hidden_max)},
                                  {OptimizerEntry {&l1->bias}},
                                  {OptimizerEntry {&l2->weights}},
                                  {OptimizerEntry {&l2->bias}},
                                  {OptimizerEntry {&pos_eval->weights}},
                                  {OptimizerEntry {&pos_eval->bias}}},
                                 0.95,
                                 0.999,
                                 1e-8,
                                 5 * 16384));

        set_save_frequency(save_rate);
        add_quantization(Quantizer {
            "quant",
            save_rate,
            QuantizerEntry<int16_t>(&ft->weights.values, quant_one, true),
            QuantizerEntry<int16_t>(&ft->bias.values, quant_one),
            QuantizerEntry<int8_t>(&l1->weights.values, quant_two),
            QuantizerEntry<int32_t>(&l1->bias.values, quant_two),
            QuantizerEntry<float>(&l2->weights.values, 1.0),
            QuantizerEntry<float>(&l2->bias.values, quant_two),
            QuantizerEntry<float>(&pos_eval->weights.values, 1.0),
            QuantizerEntry<float>(&pos_eval->bias.values, quant_two),
        });
    }

    inline int king_square_index(int relative_king_square) {
        constexpr int indices[64] {
            -1, -1, -1, -1, 14, 14, 15, 15,    //
            -1, -1, -1, -1, 14, 14, 15, 15,    //
            -1, -1, -1, -1, 12, 12, 13, 13,    //
            -1, -1, -1, -1, 12, 12, 13, 13,    //
            -1, -1, -1, -1, 8,  9,  10, 11,    //
            -1, -1, -1, -1, 8,  9,  10, 11,    //
            -1, -1, -1, -1, 4,  5,  6,  7,     //
            -1, -1, -1, -1, 0,  1,  2,  3,     //
        };

        return indices[relative_king_square];
    }

    inline int index(chess::Square piece_square,
                     chess::Piece  piece,
                     chess::Square king_square,
                     chess::Color  view) {

        const chess::PieceType piece_type  = chess::type_of(piece);
        const chess::Color     piece_color = chess::color_of(piece);

        piece_square ^= 56;
        king_square ^= 56;

        const int oP  = piece_type + 6 * (piece_color != view);
        const int oK  = (7 * !(king_square & 4)) ^ (56 * view) ^ king_square;
        const int oSq = (7 * !(king_square & 4)) ^ (56 * view) ^ piece_square;

        return king_square_index(oK) * 12 * 64 + oP * 64 + oSq;
    }

    void setup_inputs_and_outputs(dataset::DataSet<chess::Position>* positions) {
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        auto& target = m_loss->target;

#pragma omp parallel for schedule(static) num_threads(16)
        for (int b = 0; b < positions->header.entry_count; b++) {
            chess::Position* pos = &positions->positions[b];
            // fill in the inputs and target values

            chess::Square wKingSq = pos->get_king_square<chess::WHITE>();
            chess::Square bKingSq = pos->get_king_square<chess::BLACK>();

            chess::BB     bb {pos->m_occupancy};
            int           idx = 0;

            while (bb) {
                chess::Square sq                    = chess::lsb(bb);
                chess::Piece  pc                    = pos->m_pieces.get_piece(idx);

                auto          piece_index_white_pov = index(sq, pc, wKingSq, chess::WHITE);
                auto          piece_index_black_pov = index(sq, pc, bKingSq, chess::BLACK);

                if (pos->m_meta.stm() == chess::WHITE) {
                    in1->sparse_output.set(b, piece_index_white_pov);
                    in2->sparse_output.set(b, piece_index_black_pov);
                } else {
                    in2->sparse_output.set(b, piece_index_white_pov);
                    in1->sparse_output.set(b, piece_index_black_pov);
                }

                bb = chess::lsb_reset(bb);
                idx++;
            }

            float p_value = pos->m_result.score;
            float w_value = pos->m_result.wdl;

            // flip if black is to move -> relative network style
            if (pos->m_meta.stm() == chess::BLACK) {
                p_value = -p_value;
                w_value = -w_value;
            }

            float p_target = 1 / (1 + expf(-p_value * sigmoid_scale));
            float w_target = (w_value + 1) / 2.0f;

            target(b)      = lambda * p_target + (1.0 - lambda) * w_target;

            // layer_selector->dense_output.values(b, 0) =
            //     (int) ((chess::popcount(pos->m_occupancy) - 1) / 4);
        }
    }
};

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("Grapheus");

    program.add_argument("data").required().help("Directory containing training files");
    program.add_argument("--output").required().help("Output directory for network files");
    program.add_argument("--resume").help("Weights file to resume from");
    program.add_argument("--epochs")
        .default_value(1000)
        .help("Total number of epochs to train for")
        .scan<'i', int>();
    program.add_argument("--save-rate")
        .default_value(50)
        .help("How frequently to save quantized networks + weights")
        .scan<'i', int>();
    program.add_argument("--ft-size")
        .default_value(1024)
        .help("Number of neurons in the Feature Transformer")
        .scan<'i', int>();
    program.add_argument("--lambda")
        .default_value(0.0f)
        .help("Ratio of evaluation scored to use while training")
        .scan<'f', float>();
    program.add_argument("--lr")
        .default_value(0.001f)
        .help("The starting learning rate for the optimizer")
        .scan<'f', float>();
    program.add_argument("--batch-size")
        .default_value(16384)
        .help("Number of positions in a mini-batch during training")
        .scan<'i', int>();
    program.add_argument("--lr-drop-epoch")
        .default_value(500)
        .help("Epoch to execute an LR drop at")
        .scan<'i', int>();
    program.add_argument("--lr-drop-ratio")
        .default_value(0.025f)
        .help("How much to scale down LR when dropping")
        .scan<'f', float>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    math::seed(0);

    init();

    std::vector<std::string> files {};

    for (const auto& entry : fs::directory_iterator(program.get("data"))) {
        const std::string path = entry.path().string();
        files.push_back(path);
    }

    uint64_t total_positions = 0;
    for (const auto& file_path : files) {
        FILE*                  fin = fopen(file_path.c_str(), "rb");

        dataset::DataSetHeader h {};
        fread(&h, sizeof(dataset::DataSetHeader), 1, fin);

        total_positions += h.entry_count;
        fclose(fin);
    }

    std::cout << "Loading a total of " << files.size() << " files with " << total_positions
              << " total position(s)" << std::endl;

    const int   total_epochs  = program.get<int>("--epochs");
    const int   save_rate     = program.get<int>("--save-rate");
    const int   ft_size       = program.get<int>("--ft-size");
    const float lambda        = program.get<float>("--lambda");
    const float lr            = program.get<float>("--lr");
    const int   batch_size    = program.get<int>("--batch-size");
    const int   lr_drop_epoch = program.get<int>("--lr-drop-epoch");
    const float lr_drop_ratio = program.get<float>("--lr-drop-ratio");

    std::cout << "Epochs: " << total_epochs << "\n"
              << "Save Rate: " << save_rate << "\n"
              << "FT Size: " << ft_size << "\n"
              << "Lambda: " << lambda << "\n"
              << "LR: " << lr << "\n"
              << "Batch: " << batch_size << "\n"
              << "LR Drop @ " << lr_drop_epoch << "\n"
              << "LR Drop R " << lr_drop_ratio << std::endl;

    dataset::BatchLoader<chess::Position> loader {files, batch_size};
    loader.start();

    BerserkModel model {static_cast<size_t>(ft_size), lambda, static_cast<size_t>(save_rate)};
    model.set_loss(MPE {2.5, true});
    model.set_lr_schedule(StepDecayLRSchedule {lr, lr_drop_ratio, lr_drop_epoch});

    auto output_dir = program.get("--output");
    model.set_file_output(output_dir);
    for (auto& quantizer : model.m_quantizers)
        quantizer.set_path(output_dir);

    std::cout << "Files will be saved to " << output_dir << std::endl;

    if (auto previous = program.present("--resume")) {
        model.load_weights(*previous);
        std::cout << "Loaded weights from previous " << *previous << std::endl;
    }

    model.train(loader, total_epochs);

    loader.kill();

    close();
    return 0;
}