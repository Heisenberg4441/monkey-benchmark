#include "gpu_device.h"

// Заглушка, собирается когда проект скомпилирован без Vulkan SDK.

namespace monkey {

bool vulkan_compiled() { return false; }

std::unique_ptr<GpuDevice> create_vulkan_device(const VulkanOptions&, std::string* err) {
    if (err) *err = "vulkan: this binary was built without Vulkan";
    return nullptr;
}

std::vector<std::string> list_vulkan_devices() { return {}; }

} // namespace monkey
