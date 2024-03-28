#include "argparse.hpp"
#include "models/berserk.h"

#include <fstream>
#include <limits>

using namespace nn;
using namespace data;
namespace fs = std::filesystem;

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

    model::BerserkModel model {static_cast<size_t>(ft_size), lambda, static_cast<size_t>(save_rate)};
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