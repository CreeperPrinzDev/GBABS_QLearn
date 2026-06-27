#pragma once
#include <cstdint>
#include <vector>

inline bool isForbiddenGbabsByte(uint8_t b) {
    const bool A = (b & 0x80) != 0; // bit 7
    const bool V = (b & 0x08) != 0; // bit 3

    return !A && V;
}

inline bool isValidGbabsByte(uint8_t b) {
    return !isForbiddenGbabsByte(b);
}

inline std::vector<uint8_t> buildValidSymbols() {
    std::vector<uint8_t> symbols;
    symbols.reserve(192);

    for (int i = 0; i < 256; ++i) {
        uint8_t b = static_cast<uint8_t>(i);
        if (isValidGbabsByte(b)) {
            symbols.push_back(b);
        }
    }

    return symbols;
}