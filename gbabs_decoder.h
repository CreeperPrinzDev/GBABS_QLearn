#pragma once

#include "qltc.h"

#include <cstdint>
#include <string>
#include <vector>

class GbabsDecoder {
public:
    struct Result {
        bool ok = false;
        std::string error;
        uint32_t frames = 0;
    };

    Result decodeQltcToWav(
        const std::string& inputQltcPath,
        const std::string& outputWavPath
    );

private:
    static double calculateCapRate(uint16_t sampleRate);
    static double highPass(double in, double& capacitor, double capRate);

    static std::vector<int16_t> decodePayloadToStereo16(
        const std::vector<uint8_t>& payload,
        uint16_t sampleRate
    );
}; 
