#include "workload.h"

#include "workload_extra.h"
#include "workload_monkey.h"

namespace monkey {

const char* workload_name(WorkloadType t) {
    switch (t) {
    case WorkloadType::Monkey:
        return "monkey";
    case WorkloadType::Bbp:
        return "bbp";
    case WorkloadType::MillerRabin:
        return "miller-rabin";
    }
    return "unknown";
}

std::unique_ptr<IWorkload> make_workload(const Config& cfg) {
    switch (cfg.workload) {
    case WorkloadType::Monkey:
        return std::make_unique<MonkeyWorkload>();
    case WorkloadType::Bbp:
        return std::make_unique<BBPPiWorkload>();
    case WorkloadType::MillerRabin:
        return std::make_unique<MillerRabinWorkload>();
    }
    return nullptr;
}

} // namespace monkey
