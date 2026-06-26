#pragma once

#include <cstdint>
#include <memory>

#include "backend.h"

// Подключаемый workload: "что именно считает бенчмарк". Бэкенды (CPU/CUDA)
// отвечают за потоки, счётчики и остановку; workload — за чистую функцию от
// counter'а. Тяжёлая математика workload'а живёт в header-only ядрах
// (src/workloads/*_kernel.h) как __host__ __device__, поэтому один и тот же код
// исполняется и на CPU (здесь), и на GPU (в kernel'ах CUDA).

namespace monkey {

// Аккумулятор результата батча. execute_batch ДОБАВЛЯЕТ в него (не перезаписывает),
// поэтому один объект переиспользуется потоком между батчами без аллокаций.
struct WorkloadResult {
    uint64_t iterations = 0;     // сколько counter'ов обработано (для Ops/sec)
    uint64_t checksum = 0;       // аккумулятор выходов: воспроизводим + анти-DCE
    bool solution_found = false; // ранний выход (monkey: точное совпадение)
    uint64_t solution_counter = 0;

    void reset() {
        iterations = 0;
        checksum = 0;
        solution_found = false;
        solution_counter = 0;
    }
};

// Контракт workload'а. Без verify() workload не допускается в бенч.
class IWorkload {
public:
    virtual ~IWorkload() = default;

    virtual const char* name() const = 0;

    // Однократная инициализация из конфига (target, таблицы и т.п.).
    virtual void prepare(const Config& cfg) = 0;

    // Pure, без аллокаций: обрабатывает counter'ы [counter_start, +batch_size),
    // аккумулируя в out. Может выйти раньше, выставив solution_found.
    // const + без общего состояния => безопасно вызывать из всех потоков.
    virtual void execute_batch(uint64_t counter_start, uint64_t batch_size,
                               WorkloadResult& out) const = 0;

    // Самопроверка корректности против независимого эталона (gate допуска).
    virtual bool verify() const = 0;
};

// Фабрика по cfg.workload. nullptr, если workload ещё не реализован.
std::unique_ptr<IWorkload> make_workload(const Config& cfg);

// Имя workload'а для вывода/парсинга.
const char* workload_name(WorkloadType t);

} // namespace monkey
