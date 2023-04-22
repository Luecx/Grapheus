#pragma once

#include <cstdarg>
#include <fstream>

struct CSVWriter {
    std::ofstream csv_file {};
    char separator;

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



    void write(std::initializer_list<std::string> args) {
        for (auto col = args.begin(); col != args.end(); ++col) {
            if (col != args.begin())
                csv_file << separator;

            csv_file << "\"" << *col << "\"";
        }

        // new line and flush output
        csv_file << std::endl;
    }
};
