#pragma once

#include "../chess/chess.h"
#include "dataset.h"
#include "process.h"

#include <chrono>
#include <fstream>


namespace dataset {
inline uint64_t convert_text_to_bin(std::string input, std::string output) {

    std::ifstream                     filestream(input, std::ios::in);
    std::string                       line;
    dataset::DataSet<chess::Position> positions {};

    if (!filestream) {
        std::cout << "couldn't find file" << std::endl;
    } else {
        
        while (std::getline(filestream, line)) {
            chess::Position pos = chess::parse_fen(line);
            positions.positions.push_back(pos);
            positions.header.entry_count++;
        }

        dataset::write(output, positions);
    }

    return positions.header.entry_count;
}

inline void print_convert_usage() {
    std::cout << "convert <input file> <output file>     Converts data in fen [wdl] score format to "
                 "Binary format.\n";
}

inline void print_convert_multiple_usage() {
    std::cout << "convert_multiple <input dir> <output dir>     Converts all data in fen [wdl] score "
                 "format in <input dir> to Binary format and outputs to <output dir>. Assumes all "
                 "files are in .txt or .plain type.\n";
}

inline void print_shuffle_usage() {
    std::cout << "shuffle <input dir> <output_format> <num_files>     Mixes and shuffles all files "
                 "in a given directory and outputs in <output_format>. It assumes the "
                 "<output_format> contains at least one $ (dollar sign) which will be replaced with "
                 "a number ranging from 1 to <num_files>. The input files should in .bin format.\n";
}

inline void print_usage() {
    std::cout << "USAGE: subcommand <args>\n\n\n";
    std::cout << "List of available commands: \n";
    print_convert_usage();
    print_convert_multiple_usage();
    print_shuffle_usage();
}

inline void start_utils(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Grapheus Utils\n";
        print_usage();
        return;
    }

    std::string input {argv[1]};

    if (input == "-h" || input == "--help") {
        print_usage();
        return;
    }

    if (input == "convert") {
        if (argc < 4) {
            print_convert_usage();
            return;
        }

        auto     start               = std::chrono::high_resolution_clock::now();
        uint64_t positions_convreted = convert_text_to_bin(std::string {argv[2]}, std::string {argv[3]});
        auto     end                 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Sucessfully converted " << positions_convreted << "!" << std::endl;
        std::cout << "Done in " << duration << " ms." << std::endl;

        return;
    }

    if (input == "convert_multiple") {
        if (argc < 4) {
            print_convert_multiple_usage();
            return;
        }

        if (!(std::filesystem::is_directory(argv[2]) && std::filesystem::is_directory(argv[3]))) {
            std::cout << "Input/Output must be a directory!" << std::endl;
            return;
        }

        std::string out_dir = std::string {argv[3]};

        out_dir += '/';

        uint32_t count {0};
        uint64_t position_count {0};

        auto     start = std::chrono::high_resolution_clock::now();

        for (auto& file : std::filesystem::recursive_directory_iterator(argv[2])) {
            if (file.path().extension() == ".txt" || file.path().extension() == ".plain") {
                std::string file_path = file.path().string();

                std::string out_path  = out_dir + file.path().filename().string() + ".bin";

                position_count += convert_text_to_bin(file_path, out_path);
                count++;
            }
        }

        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Converted " << count << " files and " << position_count << " positions!"
                  << std::endl;
        std::cout << "Done in " << duration << " ms." << std::endl;

        return;
    }

    if (input == "shuffle") {
        if (argc < 5) {
            print_shuffle_usage();
            return;
        }

        std::vector<std::string> input_files;

        for (auto& file : std::filesystem::recursive_directory_iterator(argv[2])) {
            if (file.path().extension() == ".bin") {
                std::string file_path = file.path().string();
                input_files.push_back(file_path);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();

        dataset::mix_and_shuffle_2<chess::Position>(input_files,
                                                          std::string {argv[3]},
                                                          std::stoi(argv[4]));

        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Shuffled and mixed " << input_files.size() << " and outputted " << argv[4]
                  << " files!" << std::endl;
        std::cout << "Done in " << duration << " ms." << std::endl;
    }
}
}    // namespace dataset