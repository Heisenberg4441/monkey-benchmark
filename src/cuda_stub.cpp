#include "gpu_device.h"

// Заглушка, собирается когда проект скомпилирован без CUDA Toolkit.

namespace monkey {

bool cuda_compiled() { return false; }

std::unique_ptr<GpuDevice> create_cuda_device(const CudaOptions&, std::string* err) {
    if (err) *err = "cuda: this binary was built without CUDA";
    return nullptr;
}

std::vector<std::string> list_cuda_devices() { return {}; }

} // namespace monkey
