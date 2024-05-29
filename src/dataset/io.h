#pragma once

#include "dataset.h"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace dataset {

template<typename TYPE>
DataSet<TYPE> read(const std::string& file, uint64_t count = (1ULL << 54)) {
    static_assert(std::is_base_of<DataSetEntry, TYPE>::value,
                  "TYPE must be a subclass of DataSetEntry");

    constexpr uint64_t CHUNK_SIZE = (1 << 20);

    // open the file
    FILE* f;
    f = fopen(file.c_str(), "rb");

    // create the dataset
    DataSet<TYPE> data_set {};

    // check if opening has worked
    if (f == nullptr) {
        std::cout << "could not open: " << file << std::endl;
        return DataSet<TYPE> {};
    }

    // read the header
    fread(&data_set.header, sizeof(DataSetHeader), 1, f);

    // compute how much data to read
    auto data_to_read = std::min(count, data_set.header.entry_count);
    data_set.positions.resize(data_to_read);
    int chunks = std::ceil(data_to_read / (float) CHUNK_SIZE);

    // actually load
    for (int c = 0; c < chunks; c++) {

        int start = c * CHUNK_SIZE;
        int end   = c * CHUNK_SIZE + CHUNK_SIZE;
        if (end > data_set.positions.size())
            end = data_set.positions.size();
        fread(&data_set.positions[start], sizeof(DataSetEntry), end - start, f);
        printf("\r[Reading positions] Current count=%d", end);
        fflush(stdout);
    }
    std::cout << std::endl;

    fclose(f);
    return std::move(data_set);
}

template<typename TYPE>
inline void write(const std::string& file, const DataSet<TYPE>& data_set, uint64_t count = -1) {
    static_assert(std::is_base_of<DataSetEntry, TYPE>::value,
                  "TYPE must be a subclass of DataSetEntry");

    constexpr uint64_t CHUNK_SIZE = (1 << 20);

    // open the file
    FILE* f = fopen(file.c_str(), "wb");
    if (f == nullptr) {
        return;
    }

    // write the data count
    auto data_to_write = std::min(count, data_set.positions.size());

    // copy the header and replace the data count
    DataSetHeader header = data_set.header;
    header.entry_count   = data_to_write;

    // write the header
    fwrite(&header, sizeof(DataSetHeader), 1, f);

    // compute how much data to read
    int chunks = std::ceil(data_to_write / (float) CHUNK_SIZE);

    // actually write
    for (int c = 0; c < chunks; c++) {
        int start = c * CHUNK_SIZE;
        int end   = c * CHUNK_SIZE + CHUNK_SIZE;
        if (end > data_set.positions.size())
            end = data_set.positions.size();
        fwrite(&data_set.positions[start], sizeof(TYPE), end - start, f);
        printf("\r[Writing positions] Current count=%d", end);
        fflush(stdout);
    }
    std::cout << std::endl;

    fclose(f);
}

template<typename TYPE>
bool is_readable(const std::string& file) {
    static_assert(std::is_base_of<DataSetEntry, TYPE>::value,
                  "TYPE must be a subclass of DataSetEntry");

    if (!std::filesystem::exists(file))
        return false;

    std::filesystem::path p {file};
    auto                  size = std::filesystem::file_size(p);

    FILE*                 f;
    f = fopen(file.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }

    DataSetHeader header {};
    auto freadresult = fread(&header, sizeof(DataSetHeader), 1, f);

    if (freadresult != 1) {
        fclose(f);
        return false;
    }

    auto expected_size = header.entry_count * sizeof(TYPE) + sizeof(DataSetHeader);

    fclose(f);

    return expected_size == size;
}

/**
 * @brief Counts total positions across all datasets from provided file paths.
 *
 * Iterates through file paths, reads headers, and sums positions for total count.
 *
 * @param files Vector of dataset file paths.
 * @return Total positions across all datasets.
 */
uint64_t count_total_positions(const std::vector<std::string>& files) {
    uint64_t total_positions = 0;

    // Iterate through each file path and read dataset headers to count positions
    for (const auto& path : files) {
        std::ifstream fin(path, std::ios::binary);
        DataSetHeader h {};
        fin.read(reinterpret_cast<char*>(&h), sizeof(DataSetHeader));
        total_positions += h.entry_count;
    }

    return total_positions;
}

/**
 * @brief Retrieves dataset file paths from the specified directory.
 *
 * Iterates through the directory and collects paths of dataset files.
 *
 * @param directory The directory containing dataset files.
 * @return Vector of dataset file paths.
 */
auto fetch_dataset_paths(const std::string& directory) {
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const std::string path = entry.path().string();
        files.push_back(path);
    }
    return files;
}

}    // namespace dataset