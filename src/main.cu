#include "argparse.hpp"
#include "dataset/io.h"
#include "models/ricemodel.h"

#include <fstream>
#include <limits>

using namespace nn;
using namespace data;

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("Grapheus");

    program.add_argument("data").required().help("Directory containing training files");
    program.add_argument("--val-data").help("Directory containing validation files");
    program.add_argument("--output").required().help("Output directory for network files");
    program.add_argument("--resume").help("Weights file to resume from");
    program.add_argument("--epochs")
        .default_value(1000)
        .help("Total number of epochs to train for")
        .scan<'i', int>();
    program.add_argument("--concurrency")
        .default_value(1)
        .help("Sets the number of threads the sf binpack dataloader will use (if using the sf "
              "binpack dataloader.)")
        .scan<'i', int>();
    program.add_argument("--epoch-size")
        .default_value(100000000)
        .help("Total positions in each epoch")
        .scan<'i', int>();
    program.add_argument("--val-size")
        .default_value(10000000)
        .help("Total positions for each validation epoch")
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
    program.add_argument("--startlambda")
        .default_value(0.7f)
        .help("Ratio of evaluation at the start of the training (if applicable to the model being "
              "used)")
        .scan<'f', float>();
    program.add_argument("--endlambda")
        .default_value(0.7f)
        .help("Ratio of evaluation interpolated by the end of the training (if applicable to the "
              "model being used)")
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

    // Fetch training dataset paths
    std::vector<std::string> train_files = dataset::fetch_dataset_paths(program.get("data"));
    bool                     is_binpack  = false;

    // Print training dataset file list if files are found
    if (!train_files.empty()) {
        std::cout << "Training Dataset Files:" << std::endl;

        for (const auto& file : train_files) {
            std::cout << file << std::endl;

            if (file.find(".binpack") != std::string::npos) {
                is_binpack = true;
            }
        }

        std::cout << "Total training files: " << train_files.size() << std::endl;

        // can't count total positions in binpack files
        if (!is_binpack) {
            std::cout << "Total training positions: " << dataset::count_total_positions(train_files)
                      << std::endl
                      << std::endl;
        }
    } else {
        std::cout << "No training files found in " << program.get("data") << std::endl << std::endl;
        exit(0);
    }

    // Fetch validation dataset paths
    std::vector<std::string> val_files;

    if (program.present("--val-data")) {
        val_files = dataset::fetch_dataset_paths(program.get("--val-data"));
    }

    // Print validation dataset file list if files are found
    if (!val_files.empty()) {
        std::cout << "Validation Dataset Files:" << std::endl;

        for (const auto& file : val_files) {
            std::cout << file << std::endl;

            if (file.find(".binpack") != std::string::npos && !is_binpack) {
                std::cerr << "Validation dataset is binpack but training dataset is not. Exiting."
                          << std::endl;
                exit(1);
            }
        }
        std::cout << "Total validation files: " << val_files.size() << std::endl;

        // can't count total positions in binpack files
        if (!is_binpack) {
            std::cout << "Total validation positions: " << dataset::count_total_positions(val_files)
                      << std::endl
                      << std::endl;
        }
    }

    const int   total_epochs              = program.get<int>("--epochs");
    const int   epoch_size                = program.get<int>("--epoch-size");
    const int   val_epoch_size            = program.get<int>("--val-size");
    const int   save_rate                 = program.get<int>("--save-rate");
    const int   ft_size                   = program.get<int>("--ft-size");
    const float lambda                    = program.get<float>("--lambda");
    const float startlambda               = program.get<float>("--startlambda");
    const float endlambda                 = program.get<float>("--endlambda");
    const float lr                        = program.get<float>("--lr");
    const int   batch_size                = program.get<int>("--batch-size");
    const int   lr_drop_epoch             = program.get<int>("--lr-drop-epoch");
    const float lr_drop_ratio             = program.get<float>("--lr-drop-ratio");
    const int   binpackloader_concurrency = program.get<int>("--concurrency");

    std::cout << "Epochs: " << total_epochs << "\n"
              << "Epochs Size: " << epoch_size << "\n"
              << "Validation Size: " << val_epoch_size << "\n"
              << "Save Rate: " << save_rate << "\n"
              << "FT Size: " << ft_size << "\n"
              << "Lambda: " << lambda << "\n"
              << "Start lambda: " << startlambda << "\n"
              << "End lambda: " << endlambda << "\n"
              << "LR: " << lr << "\n"
              << "Batch: " << batch_size << "\n"
              << "LR Drop @ " << lr_drop_epoch << "\n"
              << "LR Drop R " << lr_drop_ratio << "\n"
              << std::endl;

    if (is_binpack) {
        std::cout << "Binpackloader Concurrency: " << binpackloader_concurrency << std::endl;
    }

    // using BatchLoader = dataset::BatchLoader<chess::Position>;
    using BatchLoader = binpackloader::BinpackLoader;

    BatchLoader train_loader {train_files, batch_size, binpackloader_concurrency};
    train_loader.start();

    std::optional<BatchLoader> val_loader;
    if (val_files.size() > 0) {
        val_loader.emplace(val_files, batch_size, binpackloader_concurrency);
        val_loader->start();
    }

    model::RiceModel model {static_cast<size_t>(ft_size),
                            startlambda,
                            endlambda,
                            static_cast<size_t>(save_rate)};
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

    model.train(train_loader, val_loader, total_epochs, epoch_size, val_epoch_size);

    // train_loader.kill();
    // val_loader->kill();

    close();
    return 0;
}