// Слой 2 проверки GLSL-бэкенда: реально исполняет скомпилированный SPIR-V
// (tests/philox_dump.comp -> SPIR-V) на доступном Vulkan-устройстве и сверяет
// первые 2^16 выходов с CPU-эталоном (philox.h). Это ловит расхождения
// компилятора GLSL->SPIR-V, недоступные C++-транслитерации.
//
// Если в рантайме нет Vulkan (нет loader/устройства) — тест помечается WARN и
// проходит: в CI устройство обеспечивает Mesa lavapipe (см. tests.yml).
#include "catch.hpp"
#include "philox.h"

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "philox_dump_spv.h" // сгенерированный массив SPIR-V (CMake)

namespace {

constexpr uint32_t kSeed = 0x9e3779b9u;
constexpr uint32_t kCount = 1u << 16;
constexpr uint32_t kLocalSize = 256;

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

} // namespace

TEST_CASE("Vulkan Philox shader matches CPU over 2^16 outputs", "[vulkan][glsl][determinism]") {
    // --- Instance ----------------------------------------------------------
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
#ifdef __APPLE__
    // MoltenVK: portability enumeration требуется начиная с Vulkan 1.3 loader.
    const char* exts[] = {"VK_KHR_portability_enumeration"};
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = exts;
    ici.flags = 0x00000001; // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
#endif
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        WARN("Vulkan loader/instance unavailable — skipping real-shader test");
        return;
    }

    // --- Physical device + compute queue -----------------------------------
    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    if (ndev == 0) {
        WARN("no Vulkan physical device — skipping real-shader test");
        vkDestroyInstance(inst, nullptr);
        return;
    }
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, devs.data());
    VkPhysicalDevice phys = devs[0];

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            qfi = i;
            break;
        }
    }
    REQUIRE(qfi != UINT32_MAX);

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice dev = VK_NULL_HANDLE;
    REQUIRE(vkCreateDevice(phys, &dci, nullptr, &dev) == VK_SUCCESS);
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev, qfi, 0, &queue);

    // --- Output buffer (host-visible) --------------------------------------
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(kCount) * sizeof(uint32_t);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf = VK_NULL_HANDLE;
    REQUIRE(vkCreateBuffer(dev, &bci, nullptr, &buf) == VK_SUCCESS);

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(dev, buf, &mr);
    const uint32_t mt = find_memory_type(
        phys, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    REQUIRE(mt != UINT32_MAX);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mt;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    REQUIRE(vkAllocateMemory(dev, &mai, nullptr, &mem) == VK_SUCCESS);
    REQUIRE(vkBindBufferMemory(dev, buf, mem, 0) == VK_SUCCESS);

    // --- Descriptor set layout + pipeline ----------------------------------
    VkDescriptorSetLayoutBinding dslb{};
    dslb.binding = 0;
    dslb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dslb.descriptorCount = 1;
    dslb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 1;
    dslci.pBindings = &dslb;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    REQUIRE(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl) == VK_SUCCESS);

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 2 * sizeof(uint32_t); // {seed, count}
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    REQUIRE(vkCreatePipelineLayout(dev, &plci, nullptr, &pl) == VK_SUCCESS);

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = sizeof(philox_dump_spv);
    smci.pCode = philox_dump_spv;
    VkShaderModule sm = VK_NULL_HANDLE;
    REQUIRE(vkCreateShaderModule(dev, &smci, nullptr, &sm) == VK_SUCCESS);

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline pipe = VK_NULL_HANDLE;
    REQUIRE(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe) ==
            VK_SUCCESS);

    // --- Descriptor pool + set ---------------------------------------------
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    REQUIRE(vkCreateDescriptorPool(dev, &dpci, nullptr, &pool) == VK_SUCCESS);
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    REQUIRE(vkAllocateDescriptorSets(dev, &dsai, &ds) == VK_SUCCESS);

    VkDescriptorBufferInfo dbi{buf, 0, bytes};
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.dstSet = ds;
    wds.dstBinding = 0;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

    // --- Command buffer: dispatch ------------------------------------------
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qfi;
    VkCommandPool cmdpool = VK_NULL_HANDLE;
    REQUIRE(vkCreateCommandPool(dev, &cpi, nullptr, &cmdpool) == VK_SUCCESS);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cmdpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    REQUIRE(vkAllocateCommandBuffers(dev, &cbai, &cmd) == VK_SUCCESS);

    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    const uint32_t pc[2] = {kSeed, kCount};
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cmd, (kCount + kLocalSize - 1) / kLocalSize, 1, 1);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    REQUIRE(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS);
    REQUIRE(vkQueueWaitIdle(queue) == VK_SUCCESS);

    // --- Readback + compare to CPU canon -----------------------------------
    void* mapped = nullptr;
    REQUIRE(vkMapMemory(dev, mem, 0, bytes, 0, &mapped) == VK_SUCCESS);
    const uint32_t* gpu = static_cast<const uint32_t*>(mapped);
    bool all_equal = true;
    uint32_t first_bad = 0;
    for (uint32_t i = 0; i < kCount; ++i) {
        const uint32_t cpu = prng::index_stream(kSeed, i >> 4, i & 0xFu);
        if (gpu[i] != cpu) {
            all_equal = false;
            first_bad = i;
            break;
        }
    }
    vkUnmapMemory(dev, mem);
    INFO("first mismatch at index " << first_bad);
    REQUIRE(all_equal);

    // --- Teardown ----------------------------------------------------------
    vkDestroyCommandPool(dev, cmdpool, nullptr);
    vkDestroyDescriptorPool(dev, pool, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkFreeMemory(dev, mem, nullptr);
    vkDestroyBuffer(dev, buf, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
}
