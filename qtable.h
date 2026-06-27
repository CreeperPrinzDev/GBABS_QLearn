#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>

class QTable {
public:
    static constexpr size_t ACTION_COUNT = 192;

    QTable();

    void reset();

    size_t actionCount() const;
    size_t stateCount() const;

    uint8_t actionToByte(size_t actionIndex) const;
    size_t byteToAction(uint8_t byte) const;

    float get(uint32_t state, size_t action) const;

    void update(
        uint32_t state,
        size_t action,
        float reward,
        uint32_t nextState,
        float alpha,
        float gamma
    );

    size_t bestAction(uint32_t state) const;

    bool save(const std::string& path, std::string& error) const;
    bool load(const std::string& path, std::string& error);

private:
    struct Row {
        std::array<float, ACTION_COUNT> q{};
        float bestValue = 0.0f;
        size_t bestAction = 0;
    };

    std::unordered_map<uint32_t, Row> q;

    std::array<uint8_t, ACTION_COUNT> actions{};
    std::array<int16_t, 256> byteToActionIndex{};

    void buildActionMap();

    Row& getOrCreateRow(uint32_t state);
    const Row* getRow(uint32_t state) const;

    void refreshBest(Row& row);
};