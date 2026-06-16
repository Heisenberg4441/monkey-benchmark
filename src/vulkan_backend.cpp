#include "backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <vulkan/vulkan.h>

// Скомпилированный SPIR-V шейдер (генерируется из src/monkey.comp на этапе
// сборки glslangValidator'ом, переменная monkey_comp_spv).
#include "monkey_comp_spv.h"

// Vulkan compute-бэкенд: работает на любом GPU с драйвером Vulkan — AMD, Intel,
// Apple (через MoltenVK), а также NVIDIA (как фоллбэк/для сравнения с CUDA).

namespace monkey {

namespace {

constexpr uint32_t kLocalSize = 256; // должно совпадать с шейдером
constexpr uint32_t kGroups = 2048;   // число рабочих групп

struct PushConstants {
    uint32_t seed;
    uint32_t iters;
    int32_t len;
    int32_t n;
    uint32_t mode;
    uint32_t base;
    uint32_t total;
};

#define VK_OK(call)                                                            \
    do {                                                                       \
        VkResult _r = (call);                                                  \
        if (_r != VK_SUCCESS) {                                                \
            std::fprintf(stderr, "[vulkan] %s -> %d\n", #call, (int)_r);       \
            return;                                                            \
        }                                                                      \
    } while (0)

// Создаёт instance с поддержкой portability (нужно для MoltenVK на macOS).
bool create_instance(VkInstance* out) {
    uint32_t ext_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, exts.data());

    std::vector<const char*> enabled;
    VkInstanceCreateFlags flags = 0;
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, "VK_KHR_portability_enumeration") == 0) {
            enabled.push_back("VK_KHR_portability_enumeration");
            flags |= 0x00000001; // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
        }
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "monkey_bench";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.flags = flags;
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = (uint32_t)enabled.size();
    ci.ppEnabledExtensionNames = enabled.data();
    return vkCreateInstance(&ci, nullptr, out) == VK_SUCCESS;
}

// Очки устройства: дискретный GPU предпочтительнее интегрированного.
int device_score(const VkPhysicalDeviceProperties& p) {
    switch (p.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
        default: return 0; // CPU / OTHER (например mock ICD) — пропускаем
    }
}

// Выбирает лучшее физическое устройство с очередью compute.
bool pick_device(VkInstance inst, VkPhysicalDevice* out_dev, uint32_t* out_queue,
                 VkPhysicalDeviceProperties* out_props) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, nullptr);
    if (count == 0) return false;
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(inst, &count, devs.data());

    int best = -1;
    for (auto d : devs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);
        int score = device_score(props);
        if (score == 0) continue;

        uint32_t qf = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qf, nullptr);
        std::vector<VkQueueFamilyProperties> families(qf);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qf, families.data());
        for (uint32_t i = 0; i < qf; ++i) {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                if (score > best) {
                    best = score;
                    *out_dev = d;
                    *out_queue = i;
                    *out_props = props;
                }
                break;
            }
        }
    }
    return best > 0;
}

uint32_t find_memory_type(VkPhysicalDevice dev, uint32_t type_bits,
                          VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(dev, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

// Хост-видимый когерентный буфер (target маленький, found крошечный —
// производительность этих буферов не важна, ядро ALU-bound).
bool make_buffer(VkPhysicalDevice phys, VkDevice dev, VkDeviceSize size,
                 VkBuffer* buf, VkDeviceMemory* mem, void** mapped) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bi, nullptr, buf) != VK_SUCCESS) return false;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, *buf, &req);
    uint32_t type = find_memory_type(
        phys, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX) return false;

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(dev, &ai, nullptr, mem) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(dev, *buf, *mem, 0) != VK_SUCCESS) return false;
    return vkMapMemory(dev, *mem, 0, size, 0, mapped) == VK_SUCCESS;
}

} // namespace

bool vulkan_available() {
    VkInstance inst;
    if (!create_instance(&inst)) return false;
    VkPhysicalDevice dev;
    uint32_t qf;
    VkPhysicalDeviceProperties props;
    bool ok = pick_device(inst, &dev, &qf, &props);
    vkDestroyInstance(inst, nullptr);
    return ok;
}

void run_vulkan(const Config& cfg, Control& ctrl) {
    VkInstance inst;
    if (!create_instance(&inst)) {
        std::fprintf(stderr, "[vulkan] не удалось создать instance\n");
        return;
    }

    VkPhysicalDevice phys;
    uint32_t queue_family;
    VkPhysicalDeviceProperties props;
    if (!pick_device(inst, &phys, &queue_family, &props)) {
        std::fprintf(stderr, "[vulkan] подходящий GPU не найден\n");
        vkDestroyInstance(inst, nullptr);
        return;
    }

    const uint32_t total_threads = kGroups * kLocalSize;
    std::fprintf(stderr, "[vulkan] устройство: %s | потоков: %u\n",
                 props.deviceName, total_threads);

    // Логическое устройство + очередь compute.
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    VkDevice dev;
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS) {
        std::fprintf(stderr, "[vulkan] vkCreateDevice failed\n");
        vkDestroyInstance(inst, nullptr);
        return;
    }
    VkQueue queue;
    vkGetDeviceQueue(dev, queue_family, 0, &queue);

    // Буферы: target (индексы эталона) и found (флаг совпадения).
    VkBuffer target_buf = VK_NULL_HANDLE, found_buf = VK_NULL_HANDLE;
    VkDeviceMemory target_mem = VK_NULL_HANDLE, found_mem = VK_NULL_HANDLE;
    void* target_ptr = nullptr;
    void* found_ptr = nullptr;
    VkDeviceSize target_size = sizeof(int) * (cfg.len > 0 ? cfg.len : 1);
    if (!make_buffer(phys, dev, target_size, &target_buf, &target_mem, &target_ptr) ||
        !make_buffer(phys, dev, sizeof(uint32_t), &found_buf, &found_mem, &found_ptr)) {
        std::fprintf(stderr, "[vulkan] не удалось создать буферы\n");
        vkDestroyDevice(dev, nullptr);
        vkDestroyInstance(inst, nullptr);
        return;
    }
    std::memcpy(target_ptr, cfg.target_idx.data(), sizeof(int) * cfg.len);

    // Дескрипторы: два storage-буфера.
    VkDescriptorSetLayoutBinding binds[2]{};
    for (int i = 0; i < 2; ++i) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl{};
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 2;
    dl.pBindings = binds;
    VkDescriptorSetLayout set_layout;
    VK_OK(vkCreateDescriptorSetLayout(dev, &dl, nullptr, &set_layout));

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &pool_size;
    VkDescriptorPool pool;
    VK_OK(vkCreateDescriptorPool(dev, &pci, nullptr, &pool));

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &set_layout;
    VkDescriptorSet set;
    VK_OK(vkAllocateDescriptorSets(dev, &dsai, &set));

    VkDescriptorBufferInfo binfo[2] = {{target_buf, 0, VK_WHOLE_SIZE},
                                       {found_buf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[2]{};
    for (int i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &binfo[i];
    }
    vkUpdateDescriptorSets(dev, 2, writes, 0, nullptr);

    // Пайплайн.
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(monkey_comp_spv);
    smci.pCode = monkey_comp_spv;
    VkShaderModule shader;
    VK_OK(vkCreateShaderModule(dev, &smci, nullptr, &shader));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pipe_layout;
    VK_OK(vkCreatePipelineLayout(dev, &plci, nullptr, &pipe_layout));

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName = "main";
    cpci.layout = pipe_layout;
    VkPipeline pipeline;
    VK_OK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline));

    // Командный буфер и fence.
    VkCommandPoolCreateInfo cpc{};
    cpc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpc.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpc.queueFamilyIndex = queue_family;
    VkCommandPool cmd_pool;
    VK_OK(vkCreateCommandPool(dev, &cpc, nullptr, &cmd_pool));

    VkCommandBufferAllocateInfo cba{};
    cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = cmd_pool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_OK(vkAllocateCommandBuffers(dev, &cba, &cmd));

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_OK(vkCreateFence(dev, &fci, nullptr, &fence));

    // Цикл диспатчей. iters/инвокацию задаётся --batch-size (см. cfg.batch_size).
    const uint32_t iters_per_thread = static_cast<uint32_t>(cfg.batch_size);
    PushConstants pc{};
    pc.iters = iters_per_thread;
    pc.len = cfg.len;
    pc.n = cfg.n;
    pc.mode = (cfg.mode == Mode::Random) ? 0u : 1u;
    pc.total = total_threads;
    pc.seed = cfg.seed; // детерминирован (CLI --seed), согласован с CPU/CUDA
    pc.base = 0;

    unsigned long long attempts = 0;

    while (!ctrl.stop.load(std::memory_order_relaxed)) {
        *reinterpret_cast<uint32_t*>(found_ptr) = 0;

        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0,
                                1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(PushConstants), &pc);
        vkCmdDispatch(cmd, kGroups, 1, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkResetFences(dev, 1, &fence);
        vkQueueSubmit(queue, 1, &si, fence);
        vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);

        attempts += (unsigned long long)total_threads * iters_per_thread;
        ctrl.gpu_attempts.store(attempts, std::memory_order_relaxed);

        if (*reinterpret_cast<uint32_t*>(found_ptr) != 0) {
            ctrl.found.store(true, std::memory_order_release);
            ctrl.stop.store(true, std::memory_order_release);
            break;
        }

        pc.seed += total_threads;
        pc.base += total_threads * iters_per_thread;
    }

    vkDeviceWaitIdle(dev);
    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, cmd_pool, nullptr);
    vkDestroyPipeline(dev, pipeline, nullptr);
    vkDestroyPipelineLayout(dev, pipe_layout, nullptr);
    vkDestroyShaderModule(dev, shader, nullptr);
    vkDestroyDescriptorPool(dev, pool, nullptr);
    vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
    vkDestroyBuffer(dev, target_buf, nullptr);
    vkDestroyBuffer(dev, found_buf, nullptr);
    vkFreeMemory(dev, target_mem, nullptr);
    vkFreeMemory(dev, found_mem, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
}

} // namespace monkey
