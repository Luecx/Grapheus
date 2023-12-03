#pragma once

namespace chess {

enum GameResult { WIN = 1, DRAW = 0, LOSS = -1 };

struct Result {
    int16_t score;
    int8_t  wdl;
};

}    // namespace chess
