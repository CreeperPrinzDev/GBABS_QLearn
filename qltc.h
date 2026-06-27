#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct QltcFile {
    uint32_t length = 0;
    uint16_t sampleRate = 16384;
    uint8_t originId = 'R';
    std::vector<uint8_t> payload;
};

bool readQltc(const std::string& path, QltcFile& out, std::string& error);
bool writeQltc(const std::string& path, const QltcFile& file, std::string& error);
bool validateQltc(const QltcFile& file, std::string& error);