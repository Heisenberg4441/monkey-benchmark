#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels.h"

// GPU-исполнитель диапазонов индексов — общий интерфейс Vulkan- и
// CUDA-бэкендов. Устройство создаётся один раз (instance, device, pipeline'ы),
// дальше prepare() на каждый тест и сколько угодно run_range(). Для одного и
// того же диапазона run_range() обязан вернуть ровно то же, что
// cpu_run_range() — это проверяет test_parity.
//
// Любая ошибка API — исключение GpuError с понятным текстом, а не тихий
// return: вызывающий код останавливает прогон и выходит с ненулевым кодом.

namespace monkey {

class GpuError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct GpuInfo {
    std::string api;            // "vulkan" | "cuda"
    std::string name;           // имя устройства
    std::string vendor;         // производитель
    std::string driver;         // имя и версия драйвера
    uint32_t compute_units = 0; // SM/CU, если API их сообщает (0 — неизвестно)
    std::string u64_mode;       // vulkan: "native" (shaderInt64) | "emulated" (uvec2)
    bool cpu_device = false;    // программная реализация (lavapipe, SwiftShader)
    double init_time_s = 0;     // создание устройства и pipeline'ов
};

class GpuDevice {
public:
    virtual ~GpuDevice() = default;
    virtual const GpuInfo& info() const = 0;

    // Загрузка параметров теста: эталон, буфер сумм BBP (обнуляется).
    virtual void prepare(const KernelParams& p) = 0;

    // Синхронно обрабатывает индексы [begin, begin + count).
    virtual RangeResult run_range(uint64_t begin, uint64_t count) = 0;

    // Суммы BBP по позициям окна, накопленные с prepare()/reset_bbp_acc().
    virtual void read_bbp_acc(std::vector<uint64_t>& acc) = 0;
    virtual void reset_bbp_acc() = 0;
};

struct VulkanOptions {
    bool allow_cpu_device = false; // разрешить lavapipe/SwiftShader (тесты, CI)
    bool force_emulated_u64 = false;
    int device_index = -1;         // -1 — лучшее по типу устройство
    double timeout_s = 20.0;       // предел ожидания одного диспатча
};

// Собран ли бинарь с Vulkan.
bool vulkan_compiled();
// nullptr + *err, если Vulkan недоступен (нет loader'а, устройства, фич).
std::unique_ptr<GpuDevice> create_vulkan_device(const VulkanOptions& opt, std::string* err);
// Имена всех Vulkan-устройств (для `info`).
std::vector<std::string> list_vulkan_devices();
// Загруженная библиотека Vulkan (пусто, если не найдена или не загружалась).
std::string vulkan_runtime();

struct CudaOptions {
    int device_index = 0;
};
bool cuda_compiled();
std::unique_ptr<GpuDevice> create_cuda_device(const CudaOptions& opt, std::string* err);
std::vector<std::string> list_cuda_devices();

} // namespace monkey
