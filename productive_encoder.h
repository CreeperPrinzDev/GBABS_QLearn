#pragma once

#include <cstdint>
#include <string>
#include <vector>

class ProductiveEncoder {
public:
    struct Options {
        uint16_t sampleRate = 16384;
        std::string modelPath = "model.qtbl";
        bool keepTempPcm = false;
    };

    struct Result {
        bool ok = false;
        std::string error;
        uint32_t frames = 0;
        uint64_t payloadBytes = 0;
        uint16_t sampleRate = 0;
        std::string outputPath;
    };

    Result encodeWavToQltc(
        const std::string& inputWavPath,
        const std::string& outputQltcPath,
        const Options& options = Options{}
    );

private:
    bool preprocessWavToF32Stereo(
        const std::string& wavPath,
        const std::string& pcmPath,
        uint16_t sampleRate,
        std::string& error
    ) const;

    bool loadF32StereoPcm(
        const std::string& pcmPath,
        std::vector<float>& interleaved,
        uint32_t& frames,
        std::string& error
    ) const;

    bool writeQltcTrainingOrigin(
        const std::string& path,
        const std::vector<uint8_t>& payload,
        uint16_t sampleRate,
        std::string& error
    ) const;
};