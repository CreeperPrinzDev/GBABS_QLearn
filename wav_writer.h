#pragma once

#include <cstdint>
#include <string>
#include <vector>

bool writeStereo16Wav(
    const std::string& path,
    uint16_t sampleRate,
    const std::vector<int16_t>& interleavedStereo,
    std::string& error
);