#include "backend.h"

// Заглушка, собирается когда проект скомпилирован без Vulkan SDK.

namespace monkey {

bool vulkan_available() { return false; }

void run_vulkan(const Config&, Control&) {}

} // namespace monkey
