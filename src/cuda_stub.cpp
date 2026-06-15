#include "backend.h"

// Заглушка, собирается когда проект скомпилирован без CUDA Toolkit.
// Сигнализирует, что GPU-бэкенд недоступен; вызывающий код делает откат на CPU.

namespace monkey {

bool cuda_available() { return false; }

void run_cuda(const Config&, Control&) {}

} // namespace monkey
