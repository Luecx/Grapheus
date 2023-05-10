#pragma once

#include "../misc/assert.h"
#include "dataset.h"
#include "io.h"

#include <fstream>
#include <random>
#include <thread>
#include <utility>
#include <vector>

namespace dataset {

template<typename TYPE>
struct BatchLoader {
    int            batch_size;
    DataSet<TYPE>  active_batch;

    volatile bool  keep_loading      = true;
    volatile bool  next_batch_loaded = false;
    DataSet<TYPE> load_buffer{};

    std::thread*   loading_thread;

    // files to load
    std::vector<std::string> files {};
    std::ifstream            file {};
    int                      positions_left_in_file = 0;
    int                      current_file_index     = 0;

    BatchLoader(std::vector<std::string> p_files, int batch_size, int validate_files = true)
        : batch_size(batch_size) {

        load_buffer .resize(batch_size);
        active_batch.resize(batch_size);

        files                              = std::move(p_files);
        current_file_index                 = -1;
        positions_left_in_file             = 0;
        next_batch_loaded                  = false;

        if (validate_files) {
            files.erase(std::remove_if(files.begin(),
                                       files.end(),
                                       [](const std::string& s) { return !is_readable<TYPE>(s); }),
                        files.end());
        }

        ERROR(files.size() > 0);
    }

    virtual ~BatchLoader() {
        kill();

        delete loading_thread;
    }

    void start() {
        loading_thread = new std::thread(&BatchLoader::bg_loading, this);
    }

    void kill() {
        // tell the other thread to stop looping
        keep_loading = false;

        // break the other thread from potentially
        // being in it's busy loop
        next_batch_loaded = false;

        if (loading_thread->joinable())
            loading_thread->join();
    }

    void next_file() {
        if (file.is_open())
            file.close();

        while (!file.is_open()) {
            current_file_index = (current_file_index + 1) % files.size();

            file               = std::ifstream {files[current_file_index], std::ios::binary};

            if (!file.is_open()) {
//                logging::write("Could not open file: " + files[current_file_index]);
                file.close();
            }
        }

        // get the positions in file
        DataSetHeader header {};
        file.read(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
        positions_left_in_file = header.entry_count;
    }

    void fill_buffer() {
        int fens_to_fill = batch_size;
        int read_offset  = 0;

        while (fens_to_fill > 0) {
            if (positions_left_in_file == 0)
                next_file();

            // read as many positions as possible from current file
            int filling = std::min(fens_to_fill, positions_left_in_file);
            positions_left_in_file -= filling;

            file.read(reinterpret_cast<char*>(&(load_buffer.positions[read_offset])),
                      sizeof(TYPE) * filling);

            if (file.gcount() != sizeof(TYPE) * filling) {
                exit(-1);
            }

            read_offset += filling;
            fens_to_fill -= filling;
        }
    }

    void bg_loading() {
        while (keep_loading) {
            fill_buffer();

            // mark batch as loaded and wait
            next_batch_loaded = true;
            while (next_batch_loaded)
                ;
        }
    }

    DataSet<TYPE>* next() {
        // wait until loaded
        while (!next_batch_loaded)
            ;

        // copy to active batch
        active_batch.positions.assign(load_buffer.positions.begin(), load_buffer.positions.end());
        next_batch_loaded = false;
        return &active_batch;
    }
};

}    // namespace dataset