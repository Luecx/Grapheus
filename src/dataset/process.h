#pragma once

#include "dataset.h"
#include "io.h"

#include <iostream>
#include <regex>

namespace dataset {

/**
 * shuffles all the files and writes num_files output files.
 * The output files will be generated using the out_format.
 * It assumes the out_format contains at least one "$" (dollar sign) which will be replaced
 * with a number ranging from 1 to num_files
 * @param files
 * @param out_format
 * @param num_files
 */
template<typename TYPE>
inline void mix_and_shuffle_2(std::vector<std::string>& files,
                              const std::string&        out_format,
                              const int                 num_files = 32) {

    static_assert(std::is_base_of<DataSetEntry, TYPE>::value,
                  "TYPE must be a subclass of DataSetEntry");

    std::vector<FILE*> outfiles {};
    std::vector<int>   sizes {};

    outfiles.resize(num_files);
    sizes.resize(num_files);

    for (int i = 0; i < num_files; i++) {
        // replace the out_format's dollar signs with the index
        std::string file_name = out_format;
        file_name = std::regex_replace(file_name, std::regex("\\$"), std::to_string(i + 1));

        // open the file and store it in the outfiles
        FILE* f     = fopen(file_name.c_str(), "wb");
        outfiles[i] = f;
        sizes[i]    = 0;

        // write the header
        DataSetHeader header {};
        fwrite(&header, sizeof(DataSetHeader), 1, f);
    }

    srand(time(NULL));

    // going through each file and writing the output files
    int count = 0;
    for (std::string s : files) {
        DataSet<TYPE> ds = read<TYPE>(s);
        for (TYPE& p : ds.positions) {
            // select a outfile
            int idx = rand() % num_files;
            // write it to the given file
            fwrite(&p, sizeof(TYPE), 1, outfiles[idx]);
            sizes[idx]++;
            count++;
        }

        // printing
        if (count % 1000000 == 0) {
            std::cout << count << std::endl;
        }
    }

    for (int i = 0; i < num_files; i++) {
        // correcting the size and closing the file
        // seek to the beginning
        fseek(outfiles[i], 0, SEEK_SET);
        // create new header and set position count
        DataSetHeader header {};
        header.entry_count = sizes[i];
        // overwrite the header at the start
        fwrite(&header, sizeof(DataSetHeader), 1, outfiles[i]);
        // close
        fclose(outfiles[i]);
    }

    // final intra-file shuffling
    for (int i = 0; i < num_files; i++) {
        // regenerate the file name
        std::string file_name = out_format;
        file_name = std::regex_replace(file_name, std::regex("\\$"), std::to_string(i + 1));
        // read
        DataSet<TYPE> ds = read<TYPE>(file_name);
        // shuffle
        ds.shuffle();
        // write
        write(file_name, ds);
    }
}
}    // namespace dataset