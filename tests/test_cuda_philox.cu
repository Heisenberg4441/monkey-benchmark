// Cross-backend determinism: исполняет Philox на CUDA-устройстве через тот же
// philox.h (__host__ __device__) и сверяет первые 2^16 выходов с CPU-эталоном.
// Если CUDA-устройства нет в рантайме — WARN + pass.
#include "catch.hpp"
#include "philox.h"

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr uint32_t kSeed = 0x9e3779b9u;
constexpr uint32_t kCount = 1u << 16;

__global__ void dump_kernel(uint32_t* out, uint32_t seed, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    // Тот же маппинг, что в test_glsl_philox / test_vulkan_philox.
    out[i] = prng::index_stream(seed, i >> 4, i & 0xFu);
}

} // namespace

TEST_CASE("CUDA Philox matches CPU over 2^16 outputs", "[cuda][determinism]") {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        WARN("no CUDA device — skipping cross-backend determinism test");
        return;
    }

    const size_t bytes = static_cast<size_t>(kCount) * sizeof(uint32_t);
    uint32_t* d_out = nullptr;
    REQUIRE(cudaMalloc(&d_out, bytes) == cudaSuccess);

    constexpr uint32_t kBlock = 256;
    dump_kernel<<<(kCount + kBlock - 1) / kBlock, kBlock>>>(d_out, kSeed, kCount);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<uint32_t> gpu(kCount);
    REQUIRE(cudaMemcpy(gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(d_out);

    bool all_equal = true;
    uint32_t first_bad = 0;
    for (uint32_t i = 0; i < kCount; ++i) {
        if (gpu[i] != prng::index_stream(kSeed, i >> 4, i & 0xFu)) {
            all_equal = false;
            first_bad = i;
            break;
        }
    }
    INFO("first mismatch at index " << first_bad);
    REQUIRE(all_equal);
}
