#include "gpu_device.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

// SPIR-V src/shaders/kernels.comp в двух вариантах (генерирует CMake):
// kernels_native_spv — uint64_t (shaderInt64), kernels_emu_spv — uvec2.
#include "kernels_emu_spv.h"
#include "kernels_native_spv.h"

// Vulkan compute-исполнитель: любой GPU с драйвером Vulkan — AMD, Intel, Apple
// (через MoltenVK), NVIDIA; программные lavapipe/SwiftShader — только по
// явному разрешению (тесты, CI).

namespace monkey {

namespace {

constexpr uint32_t kLocalSize = 256;      // = local_size_x в kernels.comp
constexpr uint32_t kPartialUints = 8;     // слот результата workgroup'ы
constexpr uint32_t kMaxGroups = 65535;    // гарантированный минимум maxComputeWorkGroupCount
constexpr uint64_t kThreadsTarget = 1u << 20;

struct PushConstants {
    uint32_t begin_lo, begin_hi;
    uint32_t count_lo, count_hi;
    uint32_t steps;
    uint32_t seed;
    uint32_t n;
    uint32_t len;
    uint32_t mr_base_lo, mr_base_hi;
    uint32_t mr_step_lo, mr_step_hi;
    uint32_t bbp_pos0;
    uint32_t bbp_block_terms;
    uint32_t bbp_blocks;
};

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        throw GpuError(std::string("vulkan: ") + what + " failed (VkResult " + std::to_string(r) + ")");
    }
}

bool has_extension(const std::vector<VkExtensionProperties>& exts, const char* name) {
    for (const auto& e : exts)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

// Instance с portability enumeration (нужно MoltenVK начиная с loader 1.3.216).
VkInstance create_instance() {
    uint32_t n = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr) != VK_SUCCESS) return VK_NULL_HANDLE;
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data());

    std::vector<const char*> enabled;
    VkInstanceCreateFlags flags = 0;
    if (has_extension(exts, "VK_KHR_portability_enumeration")) {
        enabled.push_back("VK_KHR_portability_enumeration");
        flags |= 0x00000001; // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
    }
    if (has_extension(exts, "VK_KHR_get_physical_device_properties2"))
        enabled.push_back("VK_KHR_get_physical_device_properties2");

    // 1.2, если loader умеет: иначе MoltenVK сообщает устройство как 1.1 и не
    // отдаёт VkPhysicalDeviceDriverProperties (имя и версию драйвера).
    uint32_t loader = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion(&loader) != VK_SUCCESS) loader = VK_API_VERSION_1_0;
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "monkey_bench";
    app.apiVersion = loader >= VK_API_VERSION_1_2 ? VK_API_VERSION_1_2 : VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.flags = flags;
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = static_cast<uint32_t>(enabled.size());
    ci.ppEnabledExtensionNames = enabled.data();
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) return VK_NULL_HANDLE;
    return inst;
}

std::vector<VkPhysicalDevice> physical_devices(VkInstance inst) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    if (n) vkEnumeratePhysicalDevices(inst, &n, devs.data());
    return devs;
}

int type_score(VkPhysicalDeviceType t) {
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
    default: return 0;
    }
}

int compute_queue_family(VkPhysicalDevice d) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fam(n);
    vkGetPhysicalDeviceQueueFamilyProperties(d, &n, fam.data());
    for (uint32_t i = 0; i < n; ++i)
        if (fam[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return static_cast<int>(i);
    return -1;
}

const char* vendor_name(uint32_t id) {
    switch (id) {
    case 0x1002: return "AMD";
    case 0x10DE: return "NVIDIA";
    case 0x8086: return "Intel";
    case 0x106B: return "Apple";
    case 0x13B5: return "ARM";
    case 0x5143: return "Qualcomm";
    case 0x1010: return "Imagination";
    case 0x10005: return "Mesa";
    default: return "unknown";
    }
}

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* ptr = nullptr;
    VkDeviceSize size = 0;
};

class VulkanDevice final : public GpuDevice {
public:
    explicit VulkanDevice(const VulkanOptions& opt) : opt_(opt) {}
    ~VulkanDevice() override { destroy(); }

    void init();
    const GpuInfo& info() const override { return info_; }
    void prepare(const KernelParams& p) override;
    RangeResult run_range(uint64_t begin, uint64_t count) override;
    void read_bbp_acc(std::vector<uint64_t>& acc) override;
    void reset_bbp_acc() override;

private:
    void make_buffer(Buffer& b, VkDeviceSize size);
    void free_buffer(Buffer& b);
    void write_descriptors();
    VkPipeline pipeline_for(KernelId k);
    void destroy();

    VulkanOptions opt_;
    GpuInfo info_;
    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t qfi_ = 0;
    bool native_u64_ = false;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkShaderModule shader_ = VK_NULL_HANDLE;
    VkPipeline pipelines_[4] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    Buffer target_, partials_, bbp_acc_;
    KernelParams params_;
    bool prepared_ = false;
};

void VulkanDevice::init() {
    const auto t0 = std::chrono::steady_clock::now();
    inst_ = create_instance();
    if (!inst_) throw GpuError("vulkan: no Vulkan loader or instance creation failed");

    // Выбор устройства: явный индекс или лучший тип; CPU-реализации — только
    // если разрешены.
    const auto devs = physical_devices(inst_);
    int best = -1, best_score = 0;
    for (size_t i = 0; i < devs.size(); ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);
        int score = type_score(p.deviceType);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && !opt_.allow_cpu_device) score = 0;
        if (compute_queue_family(devs[i]) < 0) score = 0;
        if (opt_.device_index >= 0) score = (static_cast<int>(i) == opt_.device_index && score > 0) ? 1 : 0;
        if (score > best_score) {
            best_score = score;
            best = static_cast<int>(i);
        }
    }
    if (best < 0) throw GpuError("vulkan: no suitable GPU (software devices need --vk-allow-cpu)");
    phys_ = devs[best];
    qfi_ = static_cast<uint32_t>(compute_queue_family(phys_));

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys_, &props);
    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(phys_, &feats);
    native_u64_ = feats.shaderInt64 == VK_TRUE && !opt_.force_emulated_u64;

    info_.api = "vulkan";
    info_.name = props.deviceName;
    info_.vendor = vendor_name(props.vendorID);
    info_.cpu_device = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
    info_.u64_mode = native_u64_ ? "native" : "emulated";
    {
        char buf[64];
        std::snprintf(buf, sizeof buf, "Vulkan %u.%u.%u, driver 0x%x", VK_API_VERSION_MAJOR(props.apiVersion),
                      VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion),
                      props.driverVersion);
        info_.driver = buf;
    }
    if (props.apiVersion >= VK_API_VERSION_1_2) {
        VkPhysicalDeviceDriverProperties drv{};
        drv.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        VkPhysicalDeviceProperties2 p2{};
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &drv;
        vkGetPhysicalDeviceProperties2(phys_, &p2);
        if (drv.driverName[0]) info_.driver = std::string(drv.driverName) + " " + drv.driverInfo;
    }

    // Логическое устройство. VK_KHR_portability_subset обязателен к включению,
    // если устройство его объявляет (MoltenVK).
    uint32_t n_ext = 0;
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n_ext, nullptr);
    std::vector<VkExtensionProperties> dev_exts(n_ext);
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n_ext, dev_exts.data());
    std::vector<const char*> enabled;
    if (has_extension(dev_exts, "VK_KHR_portability_subset")) enabled.push_back("VK_KHR_portability_subset");

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qfi_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceFeatures want{};
    want.shaderInt64 = native_u64_ ? VK_TRUE : VK_FALSE;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(enabled.size());
    dci.ppEnabledExtensionNames = enabled.data();
    dci.pEnabledFeatures = &want;
    check(vkCreateDevice(phys_, &dci, nullptr, &dev_), "vkCreateDevice");
    vkGetDeviceQueue(dev_, qfi_, 0, &queue_);

    // Дескрипторы: эталон, слоты групп, суммы BBP.
    VkDescriptorSetLayoutBinding binds[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl{};
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 3;
    dl.pBindings = binds;
    check(vkCreateDescriptorSetLayout(dev_, &dl, nullptr, &set_layout_), "vkCreateDescriptorSetLayout");

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set_layout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    check(vkCreatePipelineLayout(dev_, &plci, nullptr, &pipe_layout_), "vkCreatePipelineLayout");

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    check(vkCreateDescriptorPool(dev_, &dpci, nullptr, &pool_), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &set_layout_;
    check(vkAllocateDescriptorSets(dev_, &dsai, &set_), "vkAllocateDescriptorSets");

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = native_u64_ ? sizeof(kernels_native_spv) : sizeof(kernels_emu_spv);
    smci.pCode = native_u64_ ? kernels_native_spv : kernels_emu_spv;
    check(vkCreateShaderModule(dev_, &smci, nullptr, &shader_), "vkCreateShaderModule");

    VkCommandPoolCreateInfo cpc{};
    cpc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpc.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpc.queueFamilyIndex = qfi_;
    check(vkCreateCommandPool(dev_, &cpc, nullptr, &cmd_pool_), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cba{};
    cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = cmd_pool_;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(dev_, &cba, &cmd_), "vkAllocateCommandBuffers");
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    check(vkCreateFence(dev_, &fci, nullptr, &fence_), "vkCreateFence");

    make_buffer(target_, 1024);
    make_buffer(partials_, static_cast<VkDeviceSize>(kMaxGroups) * kPartialUints * 4);
    make_buffer(bbp_acc_, 64);
    write_descriptors();

    // Все pipeline'ы создаются сразу: их компиляция не должна попасть в замер.
    for (uint32_t k = 0; k < 4; ++k) pipeline_for(static_cast<KernelId>(k));
    info_.init_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void VulkanDevice::make_buffer(Buffer& b, VkDeviceSize size) {
    free_buffer(b);
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(dev_, &bi, nullptr, &b.buf), "vkCreateBuffer");
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev_, b.buf, &req);
    // Хост-видимая когерентная память; на UMA (Apple, iGPU) она же device-local.
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = find_memory_type(phys_, req.memoryTypeBits, host | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) type = find_memory_type(phys_, req.memoryTypeBits, host);
    if (type == UINT32_MAX) throw GpuError("vulkan: no host-visible coherent memory");
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    check(vkAllocateMemory(dev_, &ai, nullptr, &b.mem), "vkAllocateMemory");
    check(vkBindBufferMemory(dev_, b.buf, b.mem, 0), "vkBindBufferMemory");
    check(vkMapMemory(dev_, b.mem, 0, size, 0, &b.ptr), "vkMapMemory");
    std::memset(b.ptr, 0, size);
    b.size = size;
}

void VulkanDevice::free_buffer(Buffer& b) {
    if (!dev_) return;
    if (b.ptr) vkUnmapMemory(dev_, b.mem);
    if (b.buf) vkDestroyBuffer(dev_, b.buf, nullptr);
    if (b.mem) vkFreeMemory(dev_, b.mem, nullptr);
    b = Buffer{};
}

void VulkanDevice::write_descriptors() {
    VkDescriptorBufferInfo bi[3] = {{target_.buf, 0, VK_WHOLE_SIZE},
                                    {partials_.buf, 0, VK_WHOLE_SIZE},
                                    {bbp_acc_.buf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = set_;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(dev_, 3, w, 0, nullptr);
}

VkPipeline VulkanDevice::pipeline_for(KernelId k) {
    const auto idx = static_cast<uint32_t>(k);
    if (pipelines_[idx]) return pipelines_[idx];
    VkSpecializationMapEntry entry{0, 0, sizeof(uint32_t)};
    VkSpecializationInfo spec{1, &entry, sizeof(uint32_t), &idx};
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader_;
    cpci.stage.pName = "main";
    cpci.stage.pSpecializationInfo = &spec;
    cpci.layout = pipe_layout_;
    check(vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipelines_[idx]),
          "vkCreateComputePipelines");
    return pipelines_[idx];
}

void VulkanDevice::prepare(const KernelParams& p) {
    params_ = p;
    const VkDeviceSize tsize = std::max<VkDeviceSize>(4, p.target.size() * sizeof(int32_t));
    const VkDeviceSize asize = std::max<VkDeviceSize>(8, static_cast<VkDeviceSize>(p.bbp.positions) * 8);
    bool rewrite = false;
    if (tsize > target_.size) { make_buffer(target_, tsize); rewrite = true; }
    if (asize > bbp_acc_.size) { make_buffer(bbp_acc_, asize); rewrite = true; }
    if (rewrite) write_descriptors();
    if (!p.target.empty()) std::memcpy(target_.ptr, p.target.data(), p.target.size() * sizeof(int32_t));
    if (p.kernel == KernelId::Bbp && bbp_units(p.bbp) >= (1ull << 32))
        throw GpuError("vulkan: BBP window too large (positions * blocks >= 2^32)");
    reset_bbp_acc();
    prepared_ = true;
}

void VulkanDevice::reset_bbp_acc() { std::memset(bbp_acc_.ptr, 0, bbp_acc_.size); }

void VulkanDevice::read_bbp_acc(std::vector<uint64_t>& acc) {
    acc.assign(params_.bbp.positions, 0);
    const auto* w = static_cast<const uint32_t*>(bbp_acc_.ptr);
    for (uint32_t i = 0; i < params_.bbp.positions; ++i)
        acc[i] = static_cast<uint64_t>(w[2 * i]) | (static_cast<uint64_t>(w[2 * i + 1]) << 32);
}

RangeResult VulkanDevice::run_range(uint64_t begin, uint64_t count) {
    if (!prepared_) throw GpuError("vulkan: run_range before prepare");
    RangeResult res;
    if (count == 0) return res;

    // Геометрия: до 2^20 потоков, остальное — шагами внутри потока. Счётчики
    // workgroup'ы 32-битные, поэтому 256·steps < 2^32.
    // MR: автомату (MrLane) нужно >= 32 чисел на поток, иначе диспатч ждёт
    // отставших с простыми; на M4 Pro 2^16 потоков вместо 2^20 — ×2.6.
    const uint64_t min_steps = params_.kernel == KernelId::MillerRabin ? 32 : 1;
    const uint64_t threads = std::max<uint64_t>(1, std::min<uint64_t>((count + min_steps - 1) / min_steps, kThreadsTarget));
    const auto groups = static_cast<uint32_t>(std::min<uint64_t>((threads + kLocalSize - 1) / kLocalSize, kMaxGroups));
    const uint64_t total = static_cast<uint64_t>(groups) * kLocalSize;
    const uint64_t steps = (count + total - 1) / total;
    if (steps >= (1ull << 24)) throw GpuError("vulkan: dispatch too large");

    PushConstants pc{};
    pc.begin_lo = static_cast<uint32_t>(begin);
    pc.begin_hi = static_cast<uint32_t>(begin >> 32);
    pc.count_lo = static_cast<uint32_t>(count);
    pc.count_hi = static_cast<uint32_t>(count >> 32);
    pc.steps = static_cast<uint32_t>(steps);
    pc.seed = params_.seed;
    pc.n = params_.n;
    pc.len = params_.len();
    pc.mr_base_lo = static_cast<uint32_t>(params_.mr_base);
    pc.mr_base_hi = static_cast<uint32_t>(params_.mr_base >> 32);
    pc.mr_step_lo = static_cast<uint32_t>(params_.mr_step);
    pc.mr_step_hi = static_cast<uint32_t>(params_.mr_step >> 32);
    pc.bbp_pos0 = static_cast<uint32_t>(params_.bbp.pos0);
    pc.bbp_block_terms = params_.bbp.block_terms;
    pc.bbp_blocks = params_.bbp.blocks;

    check(vkResetCommandBuffer(cmd_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd_, &bi), "vkBeginCommandBuffer");
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_for(params_.kernel));
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &set_, 0, nullptr);
    vkCmdPushConstants(cmd_, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd_, groups, 1, 1);
    // Записи шейдера должны стать видимы хосту после fence.
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    check(vkEndCommandBuffer(cmd_), "vkEndCommandBuffer");

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    check(vkResetFences(dev_, 1, &fence_), "vkResetFences");
    check(vkQueueSubmit(queue_, 1, &si, fence_), "vkQueueSubmit");
    const auto timeout_ns = static_cast<uint64_t>(opt_.timeout_s * 1e9);
    const VkResult wr = vkWaitForFences(dev_, 1, &fence_, VK_TRUE, timeout_ns);
    if (wr == VK_TIMEOUT)
        throw GpuError("vulkan: dispatch did not finish in " + std::to_string(opt_.timeout_s) + " s");
    check(wr, "vkWaitForFences");

    const auto* slot = static_cast<const uint32_t*>(partials_.ptr);
    for (uint32_t g = 0; g < groups; ++g, slot += kPartialUints) {
        RangeResult part;
        part.checksum = static_cast<uint64_t>(slot[0]) | (static_cast<uint64_t>(slot[1]) << 32);
        part.hits = slot[2];
        part.count = slot[3];
        part.first_full = static_cast<uint64_t>(slot[4]) | (static_cast<uint64_t>(slot[5]) << 32);
        res.merge(part);
    }
    if (res.count != count)
        throw GpuError("vulkan: dispatch processed " + std::to_string(res.count) + " of " +
                       std::to_string(count) + " indices");
    return res;
}

void VulkanDevice::destroy() {
    if (dev_) {
        vkDeviceWaitIdle(dev_);
        free_buffer(target_);
        free_buffer(partials_);
        free_buffer(bbp_acc_);
        if (fence_) vkDestroyFence(dev_, fence_, nullptr);
        if (cmd_pool_) vkDestroyCommandPool(dev_, cmd_pool_, nullptr);
        for (VkPipeline& p : pipelines_)
            if (p) vkDestroyPipeline(dev_, p, nullptr);
        if (shader_) vkDestroyShaderModule(dev_, shader_, nullptr);
        if (pool_) vkDestroyDescriptorPool(dev_, pool_, nullptr);
        if (pipe_layout_) vkDestroyPipelineLayout(dev_, pipe_layout_, nullptr);
        if (set_layout_) vkDestroyDescriptorSetLayout(dev_, set_layout_, nullptr);
        vkDestroyDevice(dev_, nullptr);
        dev_ = VK_NULL_HANDLE;
    }
    if (inst_) {
        vkDestroyInstance(inst_, nullptr);
        inst_ = VK_NULL_HANDLE;
    }
}

} // namespace

bool vulkan_compiled() { return true; }

std::unique_ptr<GpuDevice> create_vulkan_device(const VulkanOptions& opt, std::string* err) {
    auto d = std::make_unique<VulkanDevice>(opt);
    try {
        d->init();
    } catch (const GpuError& e) {
        if (err) *err = e.what();
        return nullptr;
    }
    return d;
}

std::vector<std::string> list_vulkan_devices() {
    std::vector<std::string> out;
    VkInstance inst = create_instance();
    if (!inst) return out;
    for (VkPhysicalDevice d : physical_devices(inst)) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        VkPhysicalDeviceFeatures f{};
        vkGetPhysicalDeviceFeatures(d, &f);
        const char* type = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? "discrete"
                           : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated"
                           : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? "virtual"
                           : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU            ? "cpu"
                                                                                    : "other";
        out.push_back(std::string(p.deviceName) + " (" + type + ", " + vendor_name(p.vendorID) +
                      (f.shaderInt64 ? ", int64" : ", no int64") + ")");
    }
    vkDestroyInstance(inst, nullptr);
    return out;
}

} // namespace monkey
