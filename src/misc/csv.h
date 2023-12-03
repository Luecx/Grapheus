#pragma once

#include <cstdarg>
#include <fstream>

struct CSVWriter {
    std::ofstream csv_file {};
    char          separator;

    CSVWriter() = default;

    ~CSVWriter() {
        close();
    }

    void open(std::string res, char separator = ',') {
        this->separator = separator;
        csv_file.open(res);
    }

    void close() {
        csv_file.close();
    }

    template<typename T, typename... Args>
    void write(T arg, Args... args) {
        csv_file << "\"" << arg << "\"";

        if constexpr (sizeof...(args) > 0) {
            csv_file << separator;
            write(args...);
        } else {
            // new line and flush output
            csv_file << std::endl;
        }
    }
};
