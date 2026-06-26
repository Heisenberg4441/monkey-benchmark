#include "backend.h"
#include "prng.h"
#include "uint128.h"

#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

// CUDA-бэкенд. Сравнение, как и на CPU, свёрнуто к индексам символов в алфавите,
// поэтому на устройство передаётся только массив target_idx — строки не нужны.

namespace monkey {

namespace {

constexpr int kThreadsPerBlock = 256;
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

// Редукция счётчика попыток: каждый поток копит local, складывает его в
// общий для блока счётчик в shared memory, а в global делается ОДИН atomicAdd
// на блок (а не на тред) — это убирает contention на глобальном атомике.
__global__ void random_kernel(const int* __restrict__ target, int len, int n,
                              unsigned long long iters, unsigned long long seed,
                              unsigned long long* attempts, int* found) {
    __shared__ unsigned long long block_sum;
    if (threadIdx.x == 0) block_sum = 0;
    __syncthreads();

    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
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

    atomicAdd(&block_sum, local);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(attempts, block_sum);
}

// 128-битный brute-перебор. base_lo/hi — стартовый глобальный индекс блока,
// total — полное число потоков в гриде (64 бита, число потоков всегда < 2^32).
// Индексация спроектирована так, чтобы каждый перезапуск ядра продолжал с
// того же места, не дублируя работу.
__global__ void brute_kernel(const int* __restrict__ target, int len, int n,
                             uint64_t base_lo, uint64_t base_hi,
                             unsigned long long total, unsigned long long iters,
                             unsigned long long* attempts, int* found) {
    __shared__ unsigned long long block_sum;
    if (threadIdx.x == 0) block_sum = 0;
    __syncthreads();

    const Counter block_offset = counter_from_u64(blockIdx.x * kThreadsPerBlock +
                                                  threadIdx.x);
    Counter gid = {{base_lo, base_hi}};
    gid = gid + block_offset;

    unsigned long long local = 0;
    for (unsigned long long step = 0; step < iters; ++step) {
        if (*found) break;
        const Counter offset = counter_from_u64(step * total);
        Counter idx = gid + offset;

        // Извлечение цифр n-ричного числа (idx) в локальный массив.
        int comb[kMaxLen];
        const uint32_t base = static_cast<uint32_t>(n);
        for (int i = 0; i < len; ++i) {
            uint32_t rem = 0;
            idx = counter_divmod(idx, base, &rem);
            comb[len - 1 - i] = static_cast<int>(rem);
        }
        bool match = true;
        for (int j = 0; j < len; ++j) {
            if (comb[j] != target[j]) { match = false; break; }
        }
        ++local;
        if (match) atomicExch(found, 1);
    }

    atomicAdd(&block_sum, local);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(attempts, block_sum);
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

    // Сколько кандидатов проверяет один поток за запуск ядра (CLI --batch-size):
    // амортизирует оверхед запуска и копирования счётчиков host<->device.
    const unsigned long long iters_per_thread = cfg.batch_size;

    // seed детерминирован (cfg.seed, CLI --seed) и согласован с CPU-бэкендом.
    unsigned long long seed = cfg.seed;
    Counter brute_base = counter_from_u64(0);

    while (!ctrl.stop.load(std::memory_order_relaxed)) {
        if (cfg.mode == Mode::Random) {
            random_kernel<<<blocks, kThreadsPerBlock>>>(
                d_target, cfg.len, cfg.n, iters_per_thread, seed, d_attempts,
                d_found);
            seed += total_threads;
        } else {
            const uint64_t base_lo = counter_lo(brute_base);
            const uint64_t base_hi = counter_hi(brute_base);
            brute_kernel<<<blocks, kThreadsPerBlock>>>(
                d_target, cfg.len, cfg.n, base_lo, base_hi, total_threads,
                iters_per_thread, d_attempts, d_found);
            const Counter stride =
                counter_from_u64(total_threads) * iters_per_thread;
            brute_base = brute_base + stride;
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
