#pragma once

#include <cstdint>  // For fixed-width integer types
#include <iostream>  // For std::ostream
#include <algorithm> // For std::clamp

namespace chess {


enum GameResult { WIN = 1, DRAW = 0, LOSS = -1 };

class Result {
    private:
    uint16_t data;  // 16-bit storage for both score and wdl

    static const uint16_t SCORE_MASK = 0xFFFC; // 1111 1111 1111 1100 (14 bits for score)
    static const uint16_t WDL_MASK = 0x0003;   // 0000 0000 0000 0011 (2 bits for wdl)

    public:
    // Constructor to initialize data
    Result() : data(0) {}

    // Set score (14 bits)
    void set_score(int16_t score) {
        score = std::clamp(score, (int16_t) -8192, (int16_t) 8191);

        data = (data & ~SCORE_MASK) | ((score << 2) & SCORE_MASK);
    }

    // Get score (14 bits)
    int16_t score() const {
        // Shift back right and handle potential negative values
        int16_t score = (data >> 2);
        // Adjust for signed bit field if negative values are used
        if (score & (1 << 13)) { // Check the sign bit in original 14 bits
            score |= 0xC000; // Extend the sign bit
        }
        return score;
    }

    // Set wdl (2 bits)
    void set_wdl(int8_t wdl) {
        data = (data & ~WDL_MASK) | ((wdl + 1) & WDL_MASK);
    }

    // Get wdl (2 bits)
    int8_t wdl() const {
        return (data & WDL_MASK) - 1;
    }

    // stream output
    friend std::ostream& operator<<(std::ostream& os, const Result& result) {
        os << "Score: " << result.score() << ", WDL: " << result.wdl();
        return os;
    }
};

}  // namespace chess
