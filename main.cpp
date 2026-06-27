#include "qltc.h"
#include "hitpot.h"
#include "gbabs_symbols.h"
#include "teacher_encoder.h"
#include "gbabs_decoder.h"
#include "trainer.h"
#include "productive_encoder.h"

#include <iostream>
#include <string>
#include <array>
#include <cstdint>

static void printUsage() {
    std::cout
        << "GBABS Q-Learning Encoder v1.0\n"
        << "\n"
        << "Usage:\n"
        << "  gbabs_qlearn validate <file.qltc>\n"
        << "  gbabs_qlearn compare <teacher.qltc> <student.qltc>\n"
        << "  gbabs_qlearn teacher <input_audio> <output.qltc> [samplerate]\n"
        << "  gbabs_qlearn encode <input.wav> <output.qltc> <model.qtbl> [samplerate]\n"
        << "  gbabs_qlearn decode <input.qltc> <output.wav>\n"
        << "  gbabs_qlearn train <dataset_folder> <model.qtbl> <episodes> [options]\n"
        << "\n"
        << "Train options:\n"
        << "  --reset                         Start with a fresh QTable (default)\n"
        << "  --resume                        Resume from existing model.qtbl\n"
        << "  --working-set <n>               Number of training files kept in RAM at once (default: 4)\n"
        << "  --pcm-cache                     Use disk PCM cache (default)\n"
        << "  --no-pcm-cache                  Disable disk PCM cache\n"
        << "  --keep-pcm-cache                Keep generated PCM cache files after training (default)\n"
        << "  --delete-pcm-cache              Delete generated PCM cache files after training\n"
        << "  --student-forcing               Enable Teacher -> Student forcing switch (default)\n"
        << "  --no-student-forcing            Disable Student forcing, use Teacher forcing only\n"
        << "  --teacher-ratio <0.0-1.0>       Share of reset run used for Teacher forcing (default: 0.70)\n"
        << "  --student-epsilon <0.0-1.0>     Minimum epsilon when Student forcing starts (default: 0.25)\n"
        << "  --boost-student-epsilon         Raise epsilon at Student switch (default)\n"
        << "  --no-boost-student-epsilon      Do not raise epsilon at Student switch\n";
}

static bool parseU32(const std::string& s, uint32_t& out) {
    try {
        size_t pos = 0;
        unsigned long v = std::stoul(s, &pos, 10);

        if (pos != s.size()) {
            return false;
        }

        if (v > UINT32_MAX) {
            return false;
        }

        out = static_cast<uint32_t>(v);
        return true;
    }
    catch (...) {
        return false;
    }
}

static bool parseFloat(const std::string& s, float& out) {
    try {
        size_t pos = 0;
        float v = std::stof(s, &pos);

        if (pos != s.size()) {
            return false;
        }

        out = v;
        return true;
    }
    catch (...) {
        return false;
    }
}

static int cmdTrain(
    const std::string& folder,
    const std::string& modelPath,
    uint32_t episodes,
    const Trainer::Options& parsedOptions
) {
    Trainer trainer;

    Trainer::Options opt = parsedOptions;
    opt.episodes = episodes;
    opt.seed = 1337;
    opt.modelPath = modelPath;

    auto result = trainer.trainQTable(folder, opt);

    if (!result.ok) {
        std::cerr << "[FAIL] Training failed: " << result.error << "\n";
        return 1;
    }

    std::cout << "\n[OK] QTable training complete\n";
    std::cout << "Files loaded: " << result.filesLoaded << "\n";
    result.stats.print();
    std::cout << "Model written: " << modelPath << "\n";

    return 0;
}

static int cmdTeacher(
    const std::string& input,
    const std::string& output,
    uint16_t sampleRate
) {
    TeacherEncoder encoder;

    TeacherEncoder::Options opt;
    opt.sampleRate = sampleRate;
    opt.keepTempPcm = false;
    opt.tempPcmPath = "teacher_temp_f32le_stereo.pcm";

    auto result = encoder.encodeToQltc(input, output, opt);

    if (!result.ok) {
        std::cerr << "[FAIL] Teacher encode failed: " << result.error << "\n";
        return 1;
    }

    std::cout << "[OK] Teacher QLTC written\n";
    std::cout << "Frames: " << result.frames << "\n";
    std::cout << "Payload bytes: " << result.payloadBytes << "\n";
    std::cout << "Samplerate: " << sampleRate << "\n";

    return 0;
}

static int cmdDecode(
    const std::string& input,
    const std::string& output
) {
    GbabsDecoder decoder;

    auto result = decoder.decodeQltcToWav(input, output);

    if (!result.ok) {
        std::cerr << "[FAIL] Decode failed: " << result.error << "\n";
        return 1;
    }

    std::cout << "[OK] WAV written\n";
    std::cout << "Frames: " << result.frames << "\n";

    return 0;
}

static int cmdValidate(const std::string& path) {
    QltcFile file;
    std::string error;

    if (!readQltc(path, file, error)) {
        std::cerr << "[FAIL] " << error << "\n";
        return 1;
    }

    std::cout << "[OK] QLTC valid\n";
    std::cout << "Samplerate: " << file.sampleRate << "\n";
    std::cout << "OriginID: " << char(file.originId) << "\n";
    std::cout << "Payload bytes: " << file.payload.size() << "\n";

    return 0;
}

static int cmdCompare(
    const std::string& teacherPath,
    const std::string& studentPath
) {
    QltcFile teacher;
    QltcFile student;
    std::string error;

    if (!readQltc(teacherPath, teacher, error)) {
        std::cerr << "[FAIL] Teacher: " << error << "\n";
        return 1;
    }

    if (!readQltc(studentPath, student, error)) {
        std::cerr << "[FAIL] Student: " << error << "\n";
        return 1;
    }

    const size_t n = std::min(
        teacher.payload.size(),
        student.payload.size()
    );

    if (teacher.payload.size() != student.payload.size()) {
        std::cout << "[WARN] Payload sizes differ. Comparing first " << n << " bytes.\n";
    }

    std::array<uint64_t, 7> states{};
    uint64_t totalReward = 0;

    for (size_t i = 0; i < n; ++i) {
        auto r = scoreSymbol(
            teacher.payload[i],
            student.payload[i]
        );

        totalReward += r.reward;

        switch (r.state) {
        case HitPotState::POT_HIT:     states[0]++; break;
        case HitPotState::VERY_HOT:    states[1]++; break;
        case HitPotState::HOT:         states[2]++; break;
        case HitPotState::MEDIUM:      states[3]++; break;
        case HitPotState::COLD:        states[4]++; break;
        case HitPotState::BLIZZARD:    states[5]++; break;
        case HitPotState::ZERO_KELVIN: states[6]++; break;
        }
    }

    double accuracy = n > 0
        ? (double(totalReward) / double(n * 100ULL)) * 100.0
        : 0.0;

    std::cout << "Compared bytes: " << n << "\n";
    std::cout << "Accuracy: " << accuracy << "%\n";
    std::cout << "POT_HIT: " << states[0] << "\n";
    std::cout << "VERY_HOT: " << states[1] << "\n";
    std::cout << "HOT: " << states[2] << "\n";
    std::cout << "MEDIUM: " << states[3] << "\n";
    std::cout << "COLD: " << states[4] << "\n";
    std::cout << "BLIZZARD: " << states[5] << "\n";
    std::cout << "0KELVIN: " << states[6] << "\n";

    if (accuracy > 95.0) {
        std::cout << "[SUS] Accuracy > 95%. Check for overfitting or data leakage.\n";
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 0;
    }

    std::string mode = argv[1];

    if (mode == "encode") {
        if (argc < 5) {
            std::cerr << "Usage: " << argv[0]
                << " encode <input.wav> <output.qltc> <model.qtbl> [samplerate]\n";
            return 1;
        }

        ProductiveEncoder::Options opt;
        opt.modelPath = argv[4];
        opt.sampleRate = 16384;

        if (argc >= 6) {
            opt.sampleRate = static_cast<uint16_t>(std::stoi(argv[5]));
        }

        ProductiveEncoder encoder;
        const auto result = encoder.encodeWavToQltc(argv[2], argv[3], opt);

        if (!result.ok) {
            std::cerr << "[ERROR] " << result.error << "\n";
            return 1;
        }

        std::cout << "[OK] Productive encode complete\n";
        std::cout << "Frames: " << result.frames << "\n";
        std::cout << "Payload bytes: " << result.payloadBytes << "\n";
        std::cout << "Samplerate: " << result.sampleRate << "\n";
        std::cout << "OriginID: T\n";

        return 0;
    }

    if (mode == "train") {
        if (argc < 5) {
            printUsage();
            return 1;
        }

        const std::string folder = argv[2];
        const std::string modelPath = argv[3];

        uint32_t episodes = 0;

        if (!parseU32(argv[4], episodes) || episodes == 0) {
            std::cerr << "[FAIL] episodes must be > 0.\n";
            return 1;
        }

        Trainer::Options opt;
        opt.reset = true;
        opt.workingSetSize = 4;
        opt.usePcmCache = true;
        opt.keepPcmCache = true;

        opt.useStudentForcing = true;
        opt.teacherForcingRatio = 0.70f;
        opt.studentForcingEpsilon = 0.25f;
        opt.boostEpsilonOnStudentSwitch = true;

        for (int i = 5; i < argc; ++i) {
            std::string flag = argv[i];

            if (flag == "--reset") {
                opt.reset = true;
            }
            else if (flag == "--resume") {
                opt.reset = false;
            }
            else if (flag == "--working-set") {
                if (i + 1 >= argc) {
                    std::cerr << "[FAIL] --working-set requires a number.\n";
                    return 1;
                }

                uint32_t ws = 0;

                if (!parseU32(argv[i + 1], ws) || ws == 0) {
                    std::cerr << "[FAIL] --working-set must be > 0.\n";
                    return 1;
                }

                opt.workingSetSize = ws;
                ++i;
            }
            else if (flag == "--pcm-cache") {
                opt.usePcmCache = true;
            }
            else if (flag == "--no-pcm-cache") {
                opt.usePcmCache = false;
            }
            else if (flag == "--keep-pcm-cache") {
                opt.keepPcmCache = true;
            }
            else if (flag == "--delete-pcm-cache") {
                opt.keepPcmCache = false;
            }
            else if (flag == "--student-forcing") {
                opt.useStudentForcing = true;
            }
            else if (flag == "--no-student-forcing") {
                opt.useStudentForcing = false;
            }
            else if (flag == "--teacher-ratio") {
                if (i + 1 >= argc) {
                    std::cerr << "[FAIL] --teacher-ratio requires a value.\n";
                    return 1;
                }

                float ratio = 0.0f;

                if (!parseFloat(argv[i + 1], ratio) || ratio < 0.0f || ratio > 1.0f) {
                    std::cerr << "[FAIL] --teacher-ratio must be between 0.0 and 1.0.\n";
                    return 1;
                }

                opt.teacherForcingRatio = ratio;
                ++i;
            }
            else if (flag == "--student-epsilon") {
                if (i + 1 >= argc) {
                    std::cerr << "[FAIL] --student-epsilon requires a value.\n";
                    return 1;
                }

                float eps = 0.0f;

                if (!parseFloat(argv[i + 1], eps) || eps < 0.0f || eps > 1.0f) {
                    std::cerr << "[FAIL] --student-epsilon must be between 0.0 and 1.0.\n";
                    return 1;
                }

                opt.studentForcingEpsilon = eps;
                ++i;
            }
            else if (flag == "--boost-student-epsilon") {
                opt.boostEpsilonOnStudentSwitch = true;
            }
            else if (flag == "--no-boost-student-epsilon") {
                opt.boostEpsilonOnStudentSwitch = false;
            }
            else {
                std::cerr << "[FAIL] Unknown train option: " << flag << "\n";
                return 1;
            }
        }

        return cmdTrain(
            folder,
            modelPath,
            episodes,
            opt
        );
    }

    if (mode == "teacher") {
        if (argc != 4 && argc != 5) {
            printUsage();
            return 1;
        }

        uint16_t sampleRate = 16384;

        if (argc == 5) {
            int sr = 0;

            try {
                sr = std::stoi(argv[4]);
            }
            catch (...) {
                std::cerr << "[FAIL] Samplerate must be a number.\n";
                return 1;
            }

            if (sr <= 0 || sr > 65535) {
                std::cerr << "[FAIL] Samplerate must fit into u16.\n";
                return 1;
            }

            sampleRate = static_cast<uint16_t>(sr);
        }

        return cmdTeacher(
            argv[2],
            argv[3],
            sampleRate
        );
    }

    if (mode == "decode") {
        if (argc != 4) {
            printUsage();
            return 1;
        }

        return cmdDecode(
            argv[2],
            argv[3]
        );
    }

    if (mode == "validate") {
        if (argc != 3) {
            printUsage();
            return 1;
        }

        return cmdValidate(argv[2]);
    }

    if (mode == "compare") {
        if (argc != 4) {
            printUsage();
            return 1;
        }

        return cmdCompare(
            argv[2],
            argv[3]
        );
    }

    printUsage();
    return 1;
}