#include "productive_encoder.h"

#include "qtable.h"
#include "pcm_state.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {
    void writeU16LE(std::ofstream& out, uint16_t value) {
        const uint8_t bytes[2] = {
            static_cast<uint8_t>(value & 0xFF),
            static_cast<uint8_t>((value >> 8) & 0xFF)
        };

        out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    }

    void writeU32LE(std::ofstream& out, uint32_t value) {
        const uint8_t bytes[4] = {
            static_cast<uint8_t>(value & 0xFF),
            static_cast<uint8_t>((value >> 8) & 0xFF),
            static_cast<uint8_t>((value >> 16) & 0xFF),
            static_cast<uint8_t>((value >> 24) & 0xFF)
        };

        out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    }

    std::filesystem::path makeTempPcmPath(
        const std::filesystem::path& outputQltcPath,
        uint16_t sampleRate
    ) {
        const std::string tempName =
            outputQltcPath.stem().string()
            + "_encode_tmp_"
            + std::to_string(sampleRate)
            + "_stereo_f32le.pcm";

        if (outputQltcPath.parent_path().empty()) {
            return std::filesystem::path(tempName);
        }

        return outputQltcPath.parent_path() / tempName;
    }
}

ProductiveEncoder::Result ProductiveEncoder::encodeWavToQltc(
    const std::string& inputWavPath,
    const std::string& outputQltcPath,
    const Options& options
) {
    namespace fs = std::filesystem;

    Result result;
    result.sampleRate = options.sampleRate;
    result.outputPath = outputQltcPath;

    if (options.sampleRate == 0) {
        result.error = "Sample rate must not be zero.";
        return result;
    }

    if (!fs::exists(inputWavPath)) {
        result.error = "Input WAV does not exist: " + inputWavPath;
        return result;
    }

    if (!fs::exists(options.modelPath)) {
        result.error = "QTable model does not exist: " + options.modelPath;
        return result;
    }

    QTable table;
    std::string error;

    if (!table.load(options.modelPath, error)) {
        result.error = "Could not load QTable: " + error;
        return result;
    }

    const fs::path outPath(outputQltcPath);

    if (!outPath.parent_path().empty()) {
        std::error_code ec;
        fs::create_directories(outPath.parent_path(), ec);
    }

    const fs::path tempPcmPath = makeTempPcmPath(outPath, options.sampleRate);

    if (!preprocessWavToF32Stereo(
        inputWavPath,
        tempPcmPath.string(),
        options.sampleRate,
        error
    )) {
        result.error = error;
        return result;
    }

    std::vector<float> pcmInterleaved;
    uint32_t frames = 0;

    if (!loadF32StereoPcm(tempPcmPath.string(), pcmInterleaved, frames, error)) {
        if (!options.keepTempPcm) {
            std::error_code ec;
            fs::remove(tempPcmPath, ec);
        }

        result.error = error;
        return result;
    }

    if (!options.keepTempPcm) {
        std::error_code ec;
        fs::remove(tempPcmPath, ec);
    }

    std::vector<uint8_t> payload;
    payload.resize(frames);

    uint8_t prevGeneratedByte = 0x00;
    float prevL = 0.0f;
    float prevR = 0.0f;

    for (uint32_t i = 0; i < frames; ++i) {
        const float currL = pcmInterleaved[2ULL * i];
        const float currR = pcmInterleaved[2ULL * i + 1];

        const uint32_t state = makePcmEnergySideState32(
            currL,
            currR,
            prevL,
            prevR,
            prevGeneratedByte
        );

        const size_t actionIndex = table.bestAction(state);
        const uint8_t generatedByte = table.actionToByte(actionIndex);

        payload[i] = generatedByte;

        prevGeneratedByte = generatedByte;
        prevL = currL;
        prevR = currR;
    }

    if (!writeQltcTrainingOrigin(
        outputQltcPath,
        payload,
        options.sampleRate,
        error
    )) {
        result.error = error;
        return result;
    }

    result.frames = frames;
    result.payloadBytes = static_cast<uint64_t>(payload.size());
    result.ok = true;
    return result;
}

bool ProductiveEncoder::preprocessWavToF32Stereo(
    const std::string& wavPath,
    const std::string& pcmPath,
    uint16_t sampleRate,
    std::string& error
) const {
    std::stringstream ffm;

    ffm << "ffmpeg -hide_banner -loglevel error -y -i \"" << wavPath
        << "\" -ac 2 "
        << "-af \"highpass=f=40,"
        << "lowpass=f=7500,"
        << "equalizer=f=100:t=q:w=1:g=-6,"
        << "dynaudnorm=f=75:p=0.75:g=5:m=15,"
        << "alimiter=limit=0.97,"
        << "aresample=" << sampleRate
        << ":resampler=soxr:precision=33:dither_method=none,"
        << "afftdn=nf=-28\" "
        << "-f f32le -acodec pcm_f32le -ar " << sampleRate
        << " \"" << pcmPath << "\""
        << " >nul 2>&1";

    const int rc = std::system(ffm.str().c_str());

    if (rc != 0) {
        error = "FFmpeg preprocessing failed.";
        return false;
    }

    return true;
}

bool ProductiveEncoder::loadF32StereoPcm(
    const std::string& pcmPath,
    std::vector<float>& interleaved,
    uint32_t& frames,
    std::string& error
) const {
    std::ifstream src(pcmPath, std::ios::binary);

    if (!src) {
        error = "PCM not found: " + pcmPath;
        return false;
    }

    src.seekg(0, std::ios::end);
    const std::streamoff byteCount = src.tellg();
    src.seekg(0, std::ios::beg);

    if (byteCount <= 0 || (byteCount % (2 * static_cast<std::streamoff>(sizeof(float)))) != 0) {
        error = "PCM size invalid. Expected f32le stereo.";
        return false;
    }

    const uint64_t frameCount64 =
        static_cast<uint64_t>(byteCount) / (2ULL * sizeof(float));

    if (frameCount64 > std::numeric_limits<uint32_t>::max()) {
        error = "PCM too large.";
        return false;
    }

    frames = static_cast<uint32_t>(frameCount64);
    interleaved.resize(static_cast<size_t>(frames) * 2ULL);

    src.read(
        reinterpret_cast<char*>(interleaved.data()),
        byteCount
    );

    if (!src) {
        error = "Failed to read PCM data.";
        return false;
    }

    return true;
}

bool ProductiveEncoder::writeQltcTrainingOrigin(
    const std::string& path,
    const std::vector<uint8_t>& payload,
    uint16_t sampleRate,
    std::string& error
) const {
    if (payload.size() > std::numeric_limits<uint32_t>::max()) {
        error = "Payload too large for QLTC length field.";
        return false;
    }

    std::ofstream out(path, std::ios::binary);

    if (!out) {
        error = "Could not open output QLTC: " + path;
        return false;
    }

    out.write("QLTC", 4);
    writeU32LE(out, static_cast<uint32_t>(payload.size()));
    writeU16LE(out, sampleRate);
    out.put('T');
    out.put(static_cast<char>(0xFF));

    if (!payload.empty()) {
        out.write(
            reinterpret_cast<const char*>(payload.data()),
            static_cast<std::streamsize>(payload.size())
        );
    }

    if (!out) {
        error = "Failed to write QLTC.";
        return false;
    }

    return true;
}