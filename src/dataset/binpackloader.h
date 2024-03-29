#pragma on

// turn off warnings for this
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "../binpack/nnue_data_binpack_format.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>

namespace binpackloader {

using DataEntry = binpack::binpack::TrainingDataEntry;
using DataSet   = std::vector<DataEntry>;
using binpack::binpack::CompressedTrainingDataEntryParallelReader;

// Data filtering strategy taken from Stockfish's nnue-pytorch/trainingdataloader.cpp
std::function<bool(const DataEntry&)> skipPredicate = [](const DataEntry& entry) {
    static constexpr int    VALUE_NONE                      = 32002;

    static constexpr double desired_piece_count_weights[33] = {
        1.000000, 1.121094, 1.234375, 1.339844, 1.437500, 1.527344, 1.609375, 1.683594, 1.750000,
        1.808594, 1.859375, 1.902344, 1.937500, 1.964844, 1.984375, 1.996094, 2.000000, 1.996094,
        1.984375, 1.964844, 1.937500, 1.902344, 1.859375, 1.808594, 1.750000, 1.683594, 1.609375,
        1.527344, 1.437500, 1.339844, 1.234375, 1.121094, 1.000000};

    static constexpr double desired_piece_count_weights_total = []() {
        double tot = 0;
        for (auto w : desired_piece_count_weights)
            tot += w;
        return tot;
    }();

    static thread_local std::mt19937 gen(std::random_device {}());

    static thread_local double       alpha                            = 1;
    static thread_local double       piece_count_history_all[33]      = {0};
    static thread_local double       piece_count_history_passed[33]   = {0};
    static thread_local double       piece_count_history_all_total    = 0;
    static thread_local double       piece_count_history_passed_total = 0;

    static constexpr double          max_skipping_rate                = 10.0;

    auto                             do_wld_skip                      = [&entry]() {
        auto&                       prng = rng::get_thread_local_rng();

        std::bernoulli_distribution distrib(1.0
                                            - entry.score_result_prob() * entry.score_result_prob());
        return distrib(prng);
    };

    if (entry.score == VALUE_NONE) {
        return true;
    }

    if (entry.ply <= 16) {
        return true;
    }

    if ((entry.isCapturingMove() && (entry.score == 0 || entry.seeGE(0))) || entry.isInCheck()) {
        return true;
    }

    if (do_wld_skip()) {
        return true;
    }

    return false;
};

/// @brief Multithreaded dataloader to load data in Stockfish's binpack format
struct BinpackLoader {

    static constexpr std::size_t                               ChunkSize = (1 << 22);

    std::vector<std::string>                                   paths;
    std::unique_ptr<CompressedTrainingDataEntryParallelReader> reader;

    std::vector<std::size_t>                                   permute_shuffle;
    DataSet                                                    buffer;
    DataSet                                                    active_buffer;
    DataSet                                                    active_batch;

    std::thread                                                readingThread;
    int                                                        batch_size;
    int                                                        current_batch_index  = 0;
    size_t                                                     total_positions_read = 0;
    int                                                        concurrency          = 8;

    static constexpr auto openmode = std::ios::in | std::ios::binary;

    BinpackLoader(const std::vector<std::string>& filename, int batch_size, int concurrency)
        : reader(std::make_unique<CompressedTrainingDataEntryParallelReader>(concurrency,
                                                                             filename,
                                                                             openmode,
                                                                             false,
                                                                             skipPredicate))
        , batch_size(batch_size)
        , paths(filename)
        , concurrency(concurrency) {
        buffer.reserve(ChunkSize);
        active_buffer.reserve(ChunkSize);
        permute_shuffle.resize(ChunkSize);
        active_batch.reserve(batch_size);
    }

    void start() {

        current_batch_index = 0;

        shuffle();
        loadNext();
        loadToActiveBuffer();
        readingThread = std::thread(&BinpackLoader::loadNext, this);
    }

    void loadToActiveBuffer() {
        active_buffer.clear();
        for (int i = 0; i < buffer.size(); i++) {
            active_buffer.push_back(buffer[i]);
        }
    }

    void loadNext() {
        buffer.clear();

        auto k = reader->fill(buffer, ChunkSize);

        if (ChunkSize != k) {
            reader = std::make_unique<binpack::binpack::CompressedTrainingDataEntryParallelReader>(
                concurrency,
                paths,
                openmode,
                false,
                skipPredicate);
        }
    }

    DataSet& next() {
        active_batch.clear();

        for (int i = 0; i < batch_size; i++) {
            if (current_batch_index >= active_buffer.size()) {

                current_batch_index = 0;

                if (readingThread.joinable()) {
                    readingThread.join();
                }

                loadToActiveBuffer();
                shuffle();

                readingThread = std::thread(&BinpackLoader::loadNext, this);
            }

            active_batch.push_back(active_buffer[permute_shuffle[current_batch_index++]]);
        }

        return active_batch;
    }

    void shuffle() {
        std::iota(permute_shuffle.begin(), permute_shuffle.end(), 0);
        std::shuffle(permute_shuffle.begin(),
                     permute_shuffle.end(),
                     std::mt19937(std::random_device()()));
    }
};

}    // namespace binpackloader