#include "vulkan_loader.h"

#include <mutex>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace monkey {
namespace vkl {

#define MONKEY_VK_DEFINE(name) PFN_##name name = nullptr;
MONKEY_VK_GLOBAL_FUNCS(MONKEY_VK_DEFINE)
MONKEY_VK_INSTANCE_FUNCS(MONKEY_VK_DEFINE)
MONKEY_VK_DEFINE(vkEnumerateInstanceVersion)
MONKEY_VK_DEFINE(vkGetPhysicalDeviceProperties2)
#undef MONKEY_VK_DEFINE

namespace {

PFN_vkGetInstanceProcAddr get_proc = nullptr;
std::string lib_name;
std::string lib_error;
std::once_flag once;

void* open_lib(const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA(name));
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* find_sym(void* lib, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

void load_once() {
    const char* candidates[] = {
#if defined(_WIN32)
        "vulkan-1.dll",
#elif defined(__APPLE__)
        "@executable_path/libMoltenVK.dylib", // релизный архив: MoltenVK рядом с бинарём
        "@executable_path/libvulkan.1.dylib",
        "libvulkan.1.dylib",
        "/opt/homebrew/lib/libvulkan.1.dylib",
        "/usr/local/lib/libvulkan.1.dylib",
        "libMoltenVK.dylib",
        "/opt/homebrew/lib/libMoltenVK.dylib",
        "/usr/local/lib/libMoltenVK.dylib",
#else
        "libvulkan.so.1",
        "libvulkan.so",
#endif
    };
    for (const char* name : candidates) {
        void* lib = open_lib(name);
        if (!lib) continue;
        get_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(find_sym(lib, "vkGetInstanceProcAddr"));
        if (get_proc) {
            lib_name = name;
            break;
        }
    }
    if (!get_proc) {
        lib_error = "vulkan: no Vulkan runtime found (vulkan-1.dll / libvulkan.so.1 / MoltenVK); "
                    "install a GPU driver with Vulkan support";
        return;
    }
#define MONKEY_VK_LOAD_GLOBAL(name) name = reinterpret_cast<PFN_##name>(get_proc(nullptr, #name));
    MONKEY_VK_GLOBAL_FUNCS(MONKEY_VK_LOAD_GLOBAL)
    MONKEY_VK_LOAD_GLOBAL(vkEnumerateInstanceVersion)
#undef MONKEY_VK_LOAD_GLOBAL
    if (!vkCreateInstance || !vkEnumerateInstanceExtensionProperties) {
        lib_error = "vulkan: runtime " + lib_name + " lacks core entry points";
        get_proc = nullptr;
    }
}

} // namespace

bool load_library(std::string* err) {
    std::call_once(once, load_once);
    if (!get_proc && err) *err = lib_error;
    return get_proc != nullptr;
}

bool load_instance(VkInstance inst, std::string* err) {
    bool ok = true;
#define MONKEY_VK_LOAD(name)                                                                       \
    name = reinterpret_cast<PFN_##name>(get_proc(inst, #name));                                    \
    ok = ok && name != nullptr;
    MONKEY_VK_INSTANCE_FUNCS(MONKEY_VK_LOAD)
#undef MONKEY_VK_LOAD
    vkGetPhysicalDeviceProperties2 =
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(get_proc(inst, "vkGetPhysicalDeviceProperties2"));
    if (!vkGetPhysicalDeviceProperties2)
        vkGetPhysicalDeviceProperties2 =
            reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(get_proc(inst, "vkGetPhysicalDeviceProperties2KHR"));
    if (!ok && err) *err = "vulkan: runtime " + lib_name + " lacks required functions";
    return ok;
}

const std::string& library_name() { return lib_name; }

} // namespace vkl
} // namespace monkey
