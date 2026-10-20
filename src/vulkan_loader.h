#pragma once

// Загрузка Vulkan в рантайме (dlopen/LoadLibrary) вместо линковки с loader'ом.
// Бинарь без установленного Vulkan (сервер без GPU, macOS без MoltenVK)
// запускается и честно сообщает, что GPU недоступен, а CPU-категории идут как
// обычно. Таблица — глобальные указатели с именами функций Vulkan, поэтому код
// бэкенда вызывает их как обычно.
//
// Порядок поиска: Windows — vulkan-1.dll; Linux — libvulkan.so.1;
// macOS — libMoltenVK.dylib рядом с бинарём (релиз), loader из Homebrew или
// /usr/local, затем MoltenVK из стандартных путей.

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <string>

#define MONKEY_VK_GLOBAL_FUNCS(X)                                                                  \
    X(vkCreateInstance)                                                                            \
    X(vkEnumerateInstanceExtensionProperties)

#define MONKEY_VK_INSTANCE_FUNCS(X)                                                                \
    X(vkDestroyInstance)                                                                           \
    X(vkEnumeratePhysicalDevices)                                                                  \
    X(vkGetPhysicalDeviceProperties)                                                               \
    X(vkGetPhysicalDeviceFeatures)                                                                 \
    X(vkGetPhysicalDeviceQueueFamilyProperties)                                                    \
    X(vkGetPhysicalDeviceMemoryProperties)                                                         \
    X(vkEnumerateDeviceExtensionProperties)                                                        \
    X(vkCreateDevice)                                                                              \
    X(vkDestroyDevice)                                                                             \
    X(vkGetDeviceQueue)                                                                            \
    X(vkCreateDescriptorSetLayout)                                                                 \
    X(vkDestroyDescriptorSetLayout)                                                                \
    X(vkCreatePipelineLayout)                                                                      \
    X(vkDestroyPipelineLayout)                                                                     \
    X(vkCreateDescriptorPool)                                                                      \
    X(vkDestroyDescriptorPool)                                                                     \
    X(vkAllocateDescriptorSets)                                                                    \
    X(vkUpdateDescriptorSets)                                                                      \
    X(vkCreateShaderModule)                                                                        \
    X(vkDestroyShaderModule)                                                                       \
    X(vkCreateComputePipelines)                                                                    \
    X(vkDestroyPipeline)                                                                           \
    X(vkCreateCommandPool)                                                                         \
    X(vkDestroyCommandPool)                                                                        \
    X(vkAllocateCommandBuffers)                                                                    \
    X(vkCreateFence)                                                                               \
    X(vkDestroyFence)                                                                              \
    X(vkResetFences)                                                                               \
    X(vkWaitForFences)                                                                             \
    X(vkCreateBuffer)                                                                              \
    X(vkDestroyBuffer)                                                                             \
    X(vkGetBufferMemoryRequirements)                                                               \
    X(vkAllocateMemory)                                                                            \
    X(vkFreeMemory)                                                                                \
    X(vkBindBufferMemory)                                                                          \
    X(vkMapMemory)                                                                                 \
    X(vkUnmapMemory)                                                                               \
    X(vkResetCommandBuffer)                                                                        \
    X(vkBeginCommandBuffer)                                                                        \
    X(vkEndCommandBuffer)                                                                          \
    X(vkCmdBindPipeline)                                                                           \
    X(vkCmdBindDescriptorSets)                                                                     \
    X(vkCmdPushConstants)                                                                          \
    X(vkCmdDispatch)                                                                               \
    X(vkCmdPipelineBarrier)                                                                        \
    X(vkQueueSubmit)                                                                               \
    X(vkDeviceWaitIdle)

namespace monkey {
namespace vkl {

#define MONKEY_VK_DECLARE(name) extern PFN_##name name;
MONKEY_VK_GLOBAL_FUNCS(MONKEY_VK_DECLARE)
MONKEY_VK_INSTANCE_FUNCS(MONKEY_VK_DECLARE)
MONKEY_VK_DECLARE(vkEnumerateInstanceVersion)    // nullptr у loader'ов Vulkan 1.0
MONKEY_VK_DECLARE(vkGetPhysicalDeviceProperties2) // nullptr без Vulkan 1.1
#undef MONKEY_VK_DECLARE

// Загружает библиотеку и глобальные функции. false + *err — Vulkan не найден.
bool load_library(std::string* err);

// Функции уровня instance/device для созданного instance.
bool load_instance(VkInstance inst, std::string* err);

// Какая библиотека загружена (для `info`).
const std::string& library_name();

} // namespace vkl
} // namespace monkey
