#include "backend.h"
#include "prng.h"

#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

// CUDA-бэкенд. Сравнение, как и на CPU, свёрнуто к индексам символов в алфавите,
// поэтому на устройство передаётся только массив target_idx — строки не нужны.

namespace monkey {

namespace {

constexpr int kThreadsPerBlock = 256;
// Сколько кандидатов проверяет один поток за один запуск ядра. Амортизирует
// оверхед запуска и копирования счётчиков между host и device.
constexpr unsigned long long kItersPerThread = 4096;
constexpr int kMaxLen = 256;

#define CUDA_OK(call)                                                          \
    do {                                                                      \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            std::fprintf(stderr, "[cuda] %s: %s\n", #call,                  \
                         cudaGetErrorString(_e));                            \
            return;                                                           \
        }                                                                     \
    } while (0)

__global__ void random_kernel(const int* __restrict__ target, int len, int n,
                              unsigned long long iters, unsigned long long seed,
                              unsigned long long* attempts, int* found) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Поток получает собственный поток случайных чисел через seed; key —
    // порядковый номер попытки внутри потока.
    const uint32_t thread_seed = static_cast<uint32_t>(seed) + tid * 2654435761u;

    unsigned long long local = 0;
    for (unsigned long long k = 0; k < iters; ++k) {
        if (*found) break;
        bool match = true;
        for (int i = 0; i < len; ++i) {
            int c = rand_index(thread_seed, static_cast<uint32_t>(k),
                               static_cast<uint32_t>(i), n);
            if (c != target[i]) { match = false; break; }
        }
        ++local;
        if (match) atomicExch(found, 1);
    }
    atomicAdd(attempts, local);
}

__global__ void brute_kernel(const int* __restrict__ target, int len, int n,
                             unsigned long long base, unsigned long long total,
                             unsigned long long iters,
                             unsigned long long* attempts, int* found) {
    const unsigned long long gid = base + blockIdx.x * blockDim.x + threadIdx.x;
    int comb[kMaxLen];

    unsigned long long local = 0;
    for (unsigned long long step = 0; step < iters; ++step) {
        if (*found) break;
        // Глобальный индекс кандидата -> комбинация по основанию n.
        unsigned long long idx = gid + step * total;
        for (int i = 0; i < len; ++i) {
            comb[len - 1 - i] = static_cast<int>(idx % n);
            idx /= n;
        }
        bool match = true;
        for (int j = 0; j < len; ++j) {
            if (comb[j] != target[j]) { match = false; break; }
        }
        ++local;
        if (match) atomicExch(found, 1);
    }
    atomicAdd(attempts, local);
}

} // namespace

bool cuda_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return e == cudaSuccess && count > 0;
}

void run_cuda(const Config& cfg, Control& ctrl) {
    // Лимит длины актуален только для brute (ядро держит комбинацию в
    // локальном массиве). random-ядро сравнивает с эталоном в глобальной
    // памяти и не ограничено длиной.
    if (cfg.mode == Mode::Brute && cfg.len > kMaxLen) {
        std::fprintf(stderr,
                     "[cuda] brute на GPU ограничен длиной %d (эталон %d); "
                     "используйте random или CPU\n",
                     kMaxLen, cfg.len);
        return;
    }

    int device = 0;
    CUDA_OK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, device));

    // Сетка масштабируется под число SM устройства.
    const int blocks = prop.multiProcessorCount * 32;
    const unsigned long long total_threads =
        static_cast<unsigned long long>(blocks) * kThreadsPerBlock;
    std::fprintf(stderr, "[cuda] устройство: %s | SM: %d | потоков: %llu\n",
                 prop.name, prop.multiProcessorCount, total_threads);

    int* d_target = nullptr;
    unsigned long long* d_attempts = nullptr;
    int* d_found = nullptr;
    CUDA_OK(cudaMalloc(&d_target, cfg.len * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_attempts, sizeof(unsigned long long)));
    CUDA_OK(cudaMalloc(&d_found, sizeof(int)));
    CUDA_OK(cudaMemcpy(d_target, cfg.target_idx.data(), cfg.len * sizeof(int),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_attempts, 0, sizeof(unsigned long long)));
    CUDA_OK(cudaMemset(d_found, 0, sizeof(int)));

    // seed детерминирован (cfg.seed, CLI --seed) и согласован с CPU-бэкендом.
    unsigned long long seed = cfg.seed;
    unsigned long long brute_base = 0;

    while (!ctrl.stop.load(std::memory_order_relaxed)) {
        if (cfg.mode == Mode::Random) {
            random_kernel<<<blocks, kThreadsPerBlock>>>(
                d_target, cfg.len, cfg.n, kItersPerThread, seed, d_attempts,
                d_found);
            seed += total_threads;
        } else {
            brute_kernel<<<blocks, kThreadsPerBlock>>>(
                d_target, cfg.len, cfg.n, brute_base, total_threads,
                kItersPerThread, d_attempts, d_found);
            brute_base += total_threads * kItersPerThread;
        }
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());

        unsigned long long attempts = 0;
        int found = 0;
        CUDA_OK(cudaMemcpy(&attempts, d_attempts, sizeof(attempts),
                           cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(&found, d_found, sizeof(found),
                           cudaMemcpyDeviceToHost));
        ctrl.gpu_attempts.store(attempts, std::memory_order_relaxed);
        if (found) {
            ctrl.found.store(true, std::memory_order_release);
            ctrl.stop.store(true, std::memory_order_release);
            break;
        }
    }

    cudaFree(d_target);
    cudaFree(d_attempts);
    cudaFree(d_found);
}

} // namespace monkey
