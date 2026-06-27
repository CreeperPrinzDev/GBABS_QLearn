#pragma once

#include "qltc.h"
#include "gbabs_symbols.h"

#include <cstdint>
#include <string>
#include <vector>

class TeacherEncoder {
public:
    struct Options {
        uint16_t sampleRate = 16384;
        bool keepTempPcm = false;
        std::string tempPcmPath = "teacher_temp_f32le_stereo.pcm";
		float lesaGate = 23.0f / 240.0f;
    };

    struct Result {
        bool ok = false;
        std::string error;
        uint32_t frames = 0;
        uint32_t payloadBytes = 0;
    };

    Result encodeToQltc(
        const std::string& inputAudioPath,
        const std::string& outputQltcPath,
        const Options& options = Options{}
    );

private:
    static uint8_t q3(float x);
    static float softclip(float x);
    static bool runFfmpegPreprocess(
        const std::string& inputAudioPath,
        const std::string& pcmPath,
        uint16_t sampleRate,
        std::string& error
    );

    static bool loadF32StereoPcm(
        const std::string& pcmPath,
        std::vector<float>& interleaved,
        uint32_t& frames,
        std::string& error
    );

    static void repairGlobalPhase(std::vector<float>& interleaved, uint32_t frames);

    static std::vector<uint8_t> encodeIrbmsPayload(
        const std::vector<float>& interleaved,
        uint32_t frames,
        float gate
    );
};