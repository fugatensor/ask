#include "ask/Runtime/CUDA.h"
#include "Pass.h"
#include "CUBLASLt.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <set>
#include <string_view>
#include <utility>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ask::runtime {
namespace {
using Handle = void *;
using DevicePointer = std::uint64_t;
// CUDA_KERNEL_NODE_PARAMS v1. Resolve the corresponding unversioned driver
// entry points explicitly; newer *_v2 APIs use a different structure.
struct KernelNodeParams {
  Handle function = nullptr;
  unsigned gridX = 0, gridY = 0, gridZ = 0;
  unsigned blockX = 0, blockY = 0, blockZ = 0;
  unsigned sharedBytes = 0;
  void **parameters = nullptr, **extra = nullptr;
};
constexpr int notReady = 600;
void require(bool condition, std::string_view message) {
  if (!condition) throw CUDAError(std::string(message));
}
struct Driver {
  Handle library = nullptr;
#define CUDA_FUNCTION(name, signature) using name##Type = int (*) signature; name##Type name = nullptr
  CUDA_FUNCTION(cuInit, (unsigned));
  CUDA_FUNCTION(cuDeviceGet, (int *, int));
  CUDA_FUNCTION(cuDeviceGetName, (char *, int, int));
  CUDA_FUNCTION(cuDeviceGetAttribute, (int *, int, int));
  CUDA_FUNCTION(cuDevicePrimaryCtxRetain, (Handle *, int));
  CUDA_FUNCTION(cuDevicePrimaryCtxRelease, (int));
  CUDA_FUNCTION(cuCtxPushCurrent, (Handle));
  CUDA_FUNCTION(cuCtxPopCurrent, (Handle *));
  CUDA_FUNCTION(cuStreamCreate, (Handle *, unsigned));
  CUDA_FUNCTION(cuStreamDestroy, (Handle));
  CUDA_FUNCTION(cuStreamSynchronize, (Handle));
  CUDA_FUNCTION(cuStreamWaitEvent, (Handle, Handle, unsigned));
  CUDA_FUNCTION(cuModuleLoadDataEx, (Handle *, const void *, unsigned, int *, void **));
  CUDA_FUNCTION(cuModuleUnload, (Handle));
  CUDA_FUNCTION(cuModuleGetFunction, (Handle *, Handle, const char *));
  CUDA_FUNCTION(cuFuncGetAttribute, (int *, int, Handle));
  CUDA_FUNCTION(cuFuncGetParamInfo, (Handle, std::size_t, std::size_t *, std::size_t *));
  CUDA_FUNCTION(cuMemAlloc, (DevicePointer *, std::size_t));
  CUDA_FUNCTION(cuMemFree, (DevicePointer));
  CUDA_FUNCTION(cuMemAllocHost, (void **, std::size_t));
  CUDA_FUNCTION(cuMemFreeHost, (void *));
  CUDA_FUNCTION(cuMemcpyHtoDAsync, (DevicePointer, const void *, std::size_t, Handle));
  CUDA_FUNCTION(cuMemcpyDtoHAsync, (void *, DevicePointer, std::size_t, Handle));
  CUDA_FUNCTION(cuMemsetD8Async, (DevicePointer, unsigned char, std::size_t, Handle));
  CUDA_FUNCTION(cuPointerGetAttribute, (void *, int, DevicePointer));
  CUDA_FUNCTION(cuMemGetAddressRange, (DevicePointer *, std::size_t *, DevicePointer));
  CUDA_FUNCTION(cuEventCreate, (Handle *, unsigned));
  CUDA_FUNCTION(cuEventRecord, (Handle, Handle));
  CUDA_FUNCTION(cuEventQuery, (Handle));
  CUDA_FUNCTION(cuEventSynchronize, (Handle));
  CUDA_FUNCTION(cuEventDestroy, (Handle));
  CUDA_FUNCTION(cuLaunchKernel, (Handle, unsigned, unsigned, unsigned, unsigned, unsigned,
                                unsigned, unsigned, Handle, void **, void **));
  CUDA_FUNCTION(cuGraphCreate, (Handle *, unsigned));
  CUDA_FUNCTION(cuGraphAddKernelNode, (Handle *, Handle, const Handle *, std::size_t, const KernelNodeParams *));
  CUDA_FUNCTION(cuGraphInstantiate, (Handle *, Handle, Handle *, char *, std::size_t));
  CUDA_FUNCTION(cuGraphExecKernelNodeSetParams, (Handle, Handle, const KernelNodeParams *));
  CUDA_FUNCTION(cuGraphAddChildGraphNode, (Handle *, Handle, const Handle *, std::size_t, Handle));
  CUDA_FUNCTION(cuGraphExecChildGraphNodeSetParams, (Handle, Handle, Handle));
  CUDA_FUNCTION(cuStreamBeginCapture, (Handle, int));
  CUDA_FUNCTION(cuStreamEndCapture, (Handle, Handle *));
  CUDA_FUNCTION(cuGraphLaunch, (Handle, Handle));
  CUDA_FUNCTION(cuGraphExecDestroy, (Handle));
  CUDA_FUNCTION(cuGraphDestroy, (Handle));
  CUDA_FUNCTION(cuGetErrorName, (int, const char **));
  CUDA_FUNCTION(cuGetErrorString, (int, const char **));
#undef CUDA_FUNCTION
  void *symbol(const char *name) {
#ifdef _WIN32
    return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
  }
  template<class T> void bind(T &function, const char *name, const char *fallback = nullptr) {
    function = reinterpret_cast<T>(symbol(name));
    if (!function && fallback) function = reinterpret_cast<T>(symbol(fallback));
    require(function != nullptr, std::string("CUDA driver is missing ") + name);
  }
  Driver() {
#ifdef _WIN32
    library = LoadLibraryA("nvcuda.dll");
#else
    for (const char *name : {"libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1"}) {
      library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (library) break;
    }
#endif
    require(library != nullptr, "CUDA driver library is unavailable");
    try {
#define B(name) bind(name, #name)
#define V(name) bind(name, #name "_v2", #name)
      B(cuInit); B(cuDeviceGet); B(cuDeviceGetName); B(cuDeviceGetAttribute);
      B(cuDevicePrimaryCtxRetain); V(cuDevicePrimaryCtxRelease);
      V(cuCtxPushCurrent); V(cuCtxPopCurrent);
      B(cuStreamCreate); V(cuStreamDestroy); B(cuStreamSynchronize); B(cuStreamWaitEvent);
      B(cuModuleLoadDataEx); B(cuModuleUnload); B(cuModuleGetFunction);
      B(cuFuncGetAttribute); B(cuFuncGetParamInfo); V(cuMemAlloc); V(cuMemFree);
      V(cuMemAllocHost); B(cuMemFreeHost); V(cuMemcpyHtoDAsync); V(cuMemcpyDtoHAsync);
      B(cuMemsetD8Async);
      B(cuPointerGetAttribute); V(cuMemGetAddressRange);
      B(cuEventCreate); B(cuEventRecord); B(cuEventQuery); B(cuEventSynchronize); V(cuEventDestroy);
      B(cuLaunchKernel); B(cuGetErrorName); B(cuGetErrorString);
      // Graph support is required only when the caller explicitly enables it.
#define G(name) name = reinterpret_cast<name##Type>(symbol(#name))
      G(cuGraphCreate); G(cuGraphAddKernelNode); G(cuGraphExecKernelNodeSetParams);
      G(cuGraphLaunch); G(cuGraphExecDestroy); G(cuGraphDestroy);
      G(cuGraphAddChildGraphNode); G(cuGraphExecChildGraphNodeSetParams); G(cuStreamEndCapture);
      cuStreamBeginCapture = reinterpret_cast<cuStreamBeginCaptureType>(symbol("cuStreamBeginCapture_v2"));
      cuGraphInstantiate = reinterpret_cast<cuGraphInstantiateType>(symbol("cuGraphInstantiate_v2"));
#undef G
#undef B
#undef V
    } catch (...) { unload(); throw; }
  }
  bool graphs() const {
    return cuGraphCreate && cuGraphAddKernelNode && cuGraphInstantiate &&
           cuGraphExecKernelNodeSetParams && cuGraphLaunch && cuGraphExecDestroy && cuGraphDestroy;
  }
  bool libraryGraphs() const {
    return graphs() && cuGraphAddChildGraphNode && cuGraphExecChildGraphNodeSetParams &&
           cuStreamBeginCapture && cuStreamEndCapture;
  }
  void unload() noexcept {
    if (!library) return;
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(library));
#else
    dlclose(library);
#endif
    library = nullptr;
  }
  ~Driver() { unload(); }
  void check(int status, const char *operation) const {
    if (!status) return;
    const char *name = nullptr, *message = nullptr;
    cuGetErrorName(status, &name); cuGetErrorString(status, &message);
    throw CUDAError(std::string(operation) + ": " + (name ? name : "CUDA error") +
                    " (" + std::to_string(status) + ") " + (message ? message : ""));
  }
};
std::shared_ptr<Driver> getDriver() {
  static std::mutex mutex;
  static std::weak_ptr<Driver> cached;
  std::lock_guard<std::mutex> lock(mutex);
  auto driver = cached.lock();
  if (!driver) { driver = std::make_shared<Driver>(); cached = driver; }
  return driver;
}
struct ContextScope {
  Driver &driver;
  explicit ContextScope(Driver &driver, Handle context) : driver(driver) {
    driver.check(driver.cuCtxPushCurrent(context), "cuCtxPushCurrent");
  }
  ~ContextScope() { Handle previous = nullptr; driver.cuCtxPopCurrent(&previous); }
};
struct HostStorage {
  Driver &driver;
  void *pointer = nullptr;
  HostStorage(Driver &driver, std::size_t bytes) : driver(driver) {
    driver.check(driver.cuMemAllocHost(&pointer, bytes), "cuMemAllocHost");
  }
  ~HostStorage() { if (pointer) driver.cuMemFreeHost(pointer); }
};
} // namespace

struct SessionState {
  std::shared_ptr<Driver> driver;
  std::shared_ptr<detail::CUBLASLtContext> cublas;
  mutable std::recursive_mutex mutex;
  Handle context = nullptr, stream = nullptr;
  int device = 0, major = 0, minor = 0;
  std::string name;
  bool closed = false;
  std::array<unsigned,3> maxGrid{}, maxBlock{};
  unsigned maxThreads = 0;
  std::map<std::size_t, std::vector<DevicePointer>> freeBuffers;
  std::size_t pooledBytes = 0;
  static constexpr std::size_t poolLimit = 1024ULL * 1024 * 1024;
  std::vector<Handle> freeEvents;
  std::map<std::string,std::uint64_t> counters = {
      {"context_creations",0},{"primary_context_retains",0},{"stream_creations",0},
      {"module_loads",0},{"module_hits",0},{"allocations",0},{"launches",0},
      {"synchronizations",0},{"event_waits",0},{"external_stream_waits",0},
      {"host_to_device_copies",0},{"device_to_host_copies",0},
      {"allocation_reuses",0},{"event_creations",0},{"event_reuses",0},
      {"resident_device_bytes",0},{"peak_resident_device_bytes",0},{"matmul_preparations",0},
      {"program_submissions",0},{"workspace_reuses",0},
      {"graph_instantiations",0},{"graph_launches",0},{"graph_node_updates",0},
      {"graph_nodes",0},{"graph_edges",0},
      {"prepared_internal_bytes",0},{"prepared_internal_slots",0}};
  void checkOpen() const { require(!closed, "CUDA execution session is closed"); }
  void synchronize() {
    driver->check(driver->cuStreamSynchronize(stream), "cuStreamSynchronize");
    ++counters["synchronizations"];
  }
  void clearPools() {
    for (auto &[bytes, pointers] : freeBuffers)
      for (auto pointer : pointers)
        if (!driver->cuMemFree(pointer)) counters["resident_device_bytes"] -= std::max<std::size_t>(1,bytes);
    freeBuffers.clear();
    pooledBytes = 0;
    for (auto event : freeEvents) driver->cuEventDestroy(event);
    freeEvents.clear();
  }
  ~SessionState() {
    if (!context) return;
    try {
      ContextScope current(*driver, context);
      clearPools();
      cublas.reset();
      if (stream) driver->cuStreamDestroy(stream);
    } catch (...) {}
    driver->cuDevicePrimaryCtxRelease(device);
  }
};
struct DeviceBuffer::Impl {
  std::shared_ptr<SessionState> state;
  DevicePointer pointer = 0;
  std::size_t bytes = 0;
  bool owned = false, readonly = false;
};
struct Kernel::Impl {
  std::shared_ptr<SessionState> state;
  Handle module = nullptr, function = nullptr;
  std::string name;
  std::vector<unsigned> parameterWidths;
  unsigned maxThreads = 0, maxDynamicSharedBytes = 0;
};
struct Event::Impl {
  std::shared_ptr<SessionState> state;
  Handle event = nullptr;
  bool complete = false, recorded = false;
  std::vector<std::shared_ptr<DeviceBuffer>> buffers;
  std::shared_ptr<Kernel> kernel;
  std::shared_ptr<PreparedProgram> program;
  std::function<void()> releaseWorkspace;
  void finish() {
    complete = true;
    if (event) {
      if (state->closed) state->driver->cuEventDestroy(event);
      else state->freeEvents.push_back(event);
      event = nullptr;
    }
    if (releaseWorkspace) { releaseWorkspace(); releaseWorkspace = {}; }
    buffers.clear(); kernel.reset(); program.reset();
  }
};
struct Session::Impl {
  std::shared_ptr<SessionState> state;
  // Module identity uses complete bytes, so no collision-prone ad-hoc hash is
  // needed here. The Python package has already validated its SHA256 receipt.
  std::map<std::pair<std::string,std::string>, std::shared_ptr<Kernel>> kernels;
  std::vector<std::shared_ptr<Event>> pending;
};

struct PreparedProgram::Impl {
  struct Binding { std::size_t parameter, buffer; };
  struct Workspace;
  struct Step {
    std::shared_ptr<Kernel> kernel;
    std::shared_ptr<detail::CUBLASLtMatmul> matmul;
    KernelNodeParams launch;
    std::size_t outputBytes = 0;
    std::vector<DevicePointer> constants;
    std::vector<Binding> publicBindings, internalBindings;
    std::vector<std::size_t> dependencies;
    using Execute = void (*)(const Step &, Workspace &, std::size_t, SessionState &);
    using AddGraph = Handle (*)(const Step &, Workspace &, std::size_t, SessionState &, const Handle *, std::size_t);
    Execute execute = nullptr, updateGraph = nullptr;
    AddGraph addGraph = nullptr;
  };
  struct Workspace {
    std::shared_ptr<SessionState> state;
    std::vector<std::shared_ptr<DeviceBuffer>> buffers;
    std::vector<Handle> nodes;
    std::vector<std::vector<DevicePointer>> values, graphValues;
    std::vector<std::vector<void *>> parameters;
    std::vector<KernelNodeParams> launches;
    Handle graph = nullptr, executable = nullptr;
    bool busy = false;
    ~Workspace() {
      auto &s = *state;
      std::lock_guard<std::recursive_mutex> lock(s.mutex);
      try {
        ContextScope current(*s.driver, s.context);
        if (executable) s.driver->cuGraphExecDestroy(executable);
        if (graph) s.driver->cuGraphDestroy(graph);
      } catch (...) {}
    }
  };
  std::shared_ptr<SessionState> state;
  std::weak_ptr<Session> session;
  std::vector<std::size_t> sizes, inputs, outputs;
  std::vector<std::size_t> allocatedOutputs, writableOutputs;
  std::vector<std::pair<std::size_t, std::size_t>> disjointPorts;
  // Public ports use caller-owned storage; only internal buffers have slots.
  std::vector<std::size_t> slotCapacities;
  std::vector<Step> steps;
  std::vector<std::size_t> publicSteps;
  std::vector<std::pair<std::size_t,unsigned>> alignedPorts;
  std::vector<std::shared_ptr<Workspace>> workspaces;
  std::size_t graphEdges = 0;
  bool graph = false;
  static void launchKernel(const Step &, Workspace &, std::size_t, SessionState &);
  static void launchMatmul(const Step &, Workspace &, std::size_t, SessionState &);
  static void zeroMatmul(const Step &, Workspace &, std::size_t, SessionState &);
  static void emptyMatmul(const Step &, Workspace &, std::size_t, SessionState &) {}
  static Handle addKernel(const Step &, Workspace &, std::size_t, SessionState &, const Handle *, std::size_t);
  static void updateKernel(const Step &, Workspace &, std::size_t, SessionState &);
  static Handle capture(const Step &, Workspace &, std::size_t, SessionState &);
  static Handle addLibrary(const Step &, Workspace &, std::size_t, SessionState &, const Handle *, std::size_t);
  static void updateLibrary(const Step &, Workspace &, std::size_t, SessionState &);
};

void PreparedProgram::Impl::launchKernel(const Step &, Workspace &w, std::size_t i, SessionState &s) {
  const auto &n=w.launches[i];
  s.driver->check(s.driver->cuLaunchKernel(n.function,n.gridX,n.gridY,n.gridZ,n.blockX,n.blockY,n.blockZ,
                                         n.sharedBytes,s.stream,n.parameters,nullptr),"cuLaunchKernel");
}
void PreparedProgram::Impl::launchMatmul(const Step &step, Workspace &w, std::size_t i, SessionState &s) {
  step.matmul->launch(w.values[i].data(),s.stream);
}
void PreparedProgram::Impl::zeroMatmul(const Step &step, Workspace &w, std::size_t i, SessionState &s) {
  s.driver->check(s.driver->cuMemsetD8Async(w.values[i][2],0,step.outputBytes,s.stream),"cuMemsetD8Async(matmul)");
}
Handle PreparedProgram::Impl::addKernel(const Step &, Workspace &w, std::size_t i, SessionState &s,
                                       const Handle *previous, std::size_t count) {
  Handle node=nullptr;
  s.driver->check(s.driver->cuGraphAddKernelNode(&node,w.graph,previous,count,&w.launches[i]),"cuGraphAddKernelNode");
  return node;
}
void PreparedProgram::Impl::updateKernel(const Step &, Workspace &w, std::size_t i, SessionState &s) {
  s.driver->check(s.driver->cuGraphExecKernelNodeSetParams(w.executable,w.nodes[i],&w.launches[i]),"cuGraphExecKernelNodeSetParams");
}
Handle PreparedProgram::Impl::capture(const Step &step, Workspace &w, std::size_t i, SessionState &s) {
  auto &d=*s.driver;
  d.check(d.cuStreamBeginCapture(s.stream,1),"cuStreamBeginCapture"); // Thread-local capture.
  Handle child=nullptr;
  try { step.execute(step,w,i,s); }
  catch(...) {
    d.cuStreamEndCapture(s.stream,&child);
    if(child) d.cuGraphDestroy(child);
    throw;
  }
  d.check(d.cuStreamEndCapture(s.stream,&child),"cuStreamEndCapture");
  return child;
}
Handle PreparedProgram::Impl::addLibrary(const Step &step, Workspace &w, std::size_t i, SessionState &s,
                                        const Handle *previous, std::size_t count) {
  Handle child=capture(step,w,i,s),node=nullptr;
  auto &d=*s.driver;
  int status=d.cuGraphAddChildGraphNode(&node,w.graph,previous,count,child);
  d.cuGraphDestroy(child);
  d.check(status,"cuGraphAddChildGraphNode");
  return node;
}
void PreparedProgram::Impl::updateLibrary(const Step &step, Workspace &w, std::size_t i, SessionState &s) {
  Handle child=capture(step,w,i,s);
  auto &d=*s.driver;
  int status=d.cuGraphExecChildGraphNodeSetParams(w.executable,w.nodes[i],child);
  d.cuGraphDestroy(child);
  d.check(status,"cuGraphExecChildGraphNodeSetParams");
}

Session::Session(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {}
std::shared_ptr<Session> Session::create(int ordinal) {
  require(ordinal >= 0, "device ordinal must be nonnegative");
  auto state = std::make_shared<SessionState>();
  state->driver = getDriver(); auto &d = *state->driver;
  d.check(d.cuInit(0), "cuInit");
  d.check(d.cuDeviceGet(&state->device, ordinal), "cuDeviceGet");
  char name[256] = {};
  d.check(d.cuDeviceGetName(name,sizeof(name),state->device), "cuDeviceGetName");
  state->name = name;
  auto attribute = [&](int key) { int value = 0;
    d.check(d.cuDeviceGetAttribute(&value,key,state->device), "cuDeviceGetAttribute");
    return value;
  };
  state->major = attribute(75); state->minor = attribute(76);
  state->maxThreads = attribute(1);
  for (unsigned i=0;i<3;++i) { state->maxBlock[i]=attribute(2+i); state->maxGrid[i]=attribute(5+i); }
  d.check(d.cuDevicePrimaryCtxRetain(&state->context,state->device), "cuDevicePrimaryCtxRetain");
  ++state->counters["primary_context_retains"];
  { ContextScope current(d,state->context);
    d.check(d.cuStreamCreate(&state->stream,1), "cuStreamCreate"); }
  ++state->counters["stream_creations"];
  auto impl = std::make_unique<Impl>(); impl->state = state;
  return std::shared_ptr<Session>(new Session(std::move(impl)));
}
Session::~Session() { try { close(); } catch (...) {} }
void Session::checkTypes(const std::vector<std::string> &types) const {
  auto &s=*impl->state; std::lock_guard<std::recursive_mutex> lock(s.mutex); s.checkOpen();
  static const std::set<std::string> supported={"bool","i32","i64","f32","f16","bf16"};
  for (const auto &type:types) {
    require(supported.count(type)!=0,"unsupported runtime element type: "+type);
    if (type=="bf16") require(s.major>=8,"bf16 requires hardware support (CUDA compute capability >= 8.0); selected device "+s.name+" has "+std::to_string(s.major)+"."+std::to_string(s.minor));
  }
}
std::shared_ptr<DeviceBuffer> Session::allocate(std::size_t bytes) {
  auto state=impl->state; auto &s=*state; std::lock_guard<std::recursive_mutex> lock(s.mutex); s.checkOpen();
  ContextScope current(*s.driver,s.context);
  auto p=std::make_unique<DeviceBuffer::Impl>(); p->state=state; p->bytes=bytes; p->owned=true;
  // Empty tensors have no accessible elements but still own a stable allocation.
  auto available=s.freeBuffers.find(bytes);
  if(available!=s.freeBuffers.end() && !available->second.empty()) {
    p->pointer=available->second.back();available->second.pop_back();
    s.pooledBytes-=std::max<std::size_t>(1,bytes);++s.counters["allocation_reuses"];
    if(available->second.empty()) s.freeBuffers.erase(available);
  } else {
    int status=s.driver->cuMemAlloc(&p->pointer,std::max<std::size_t>(1,bytes));
    // Cached free blocks must not make an otherwise possible allocation fail.
    if(status==2 && s.pooledBytes) {
      s.clearPools();status=s.driver->cuMemAlloc(&p->pointer,std::max<std::size_t>(1,bytes));
    }
    s.driver->check(status,"cuMemAlloc");
    ++s.counters["allocations"];
    auto &resident=s.counters["resident_device_bytes"];
    resident+=std::max<std::size_t>(1,bytes);
    s.counters["peak_resident_device_bytes"]=std::max(s.counters["peak_resident_device_bytes"],resident);
  }
  return std::shared_ptr<DeviceBuffer>(new DeviceBuffer(std::move(p)));
}
std::shared_ptr<DeviceBuffer> Session::importBuffer(std::uint64_t pointer,std::size_t bytes,bool readonly) {
  auto state=impl->state; auto &s=*state; std::lock_guard<std::recursive_mutex> lock(s.mutex); s.checkOpen();
  require(pointer || !bytes,"nonempty external tensor has a null device pointer");
  require(bytes<=std::numeric_limits<std::uint64_t>::max()-pointer,"external tensor pointer range overflows");
  ContextScope current(*s.driver,s.context);
  if (pointer) {
    auto &d=*s.driver; int device=-1,memory=0; Handle context=nullptr;
    d.check(d.cuPointerGetAttribute(&device,9,pointer),"cuPointerGetAttribute(device)");
    d.check(d.cuPointerGetAttribute(&memory,2,pointer),"cuPointerGetAttribute(memory type)");
    d.check(d.cuPointerGetAttribute(&context,1,pointer),"cuPointerGetAttribute(context)");
    require(device==s.device,"external tensor belongs to a different CUDA device");
    require(memory==2 || memory==4,"external tensor must refer to device or managed memory");
    require(context==s.context,"external tensor must belong to this device's primary CUDA context");
    // Address ranges are available for ordinary linear allocations. CUDA memory
    // pools/VMM may not expose this query; their producer owns the size contract.
    DevicePointer base=0; std::size_t allocation=0;
    int status=d.cuMemGetAddressRange(&base,&allocation,pointer);
    if (!status) require(pointer>=base && pointer-base<=allocation && bytes<=allocation-(pointer-base),
                         "external tensor extends beyond its CUDA allocation");
    else if (status!=1 && status!=801) d.check(status,"cuMemGetAddressRange");
  }
  auto p=std::make_unique<DeviceBuffer::Impl>(); p->state=state;p->pointer=pointer;p->bytes=bytes;p->readonly=readonly;
  return std::shared_ptr<DeviceBuffer>(new DeviceBuffer(std::move(p)));
}
std::shared_ptr<Kernel> Session::load(const std::string &cubin,const std::string &entry,
    const std::vector<std::string> &requiredTypes,int arch,const std::vector<unsigned> &parameterWidths) {
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();checkTypes(requiredTypes);
  const bool ptx=cubin.compare(0,4,"\x7f" "ELF")!=0;
  require(arch>=10 && (ptx ? s.major*10+s.minor>=arch : s.major==arch/10 && s.minor>=arch%10),
          "CUDA code target is incompatible with the selected device");
  require(ptx ? cubin.find('\0')==std::string::npos && cubin.find(".version")!=std::string::npos
              : cubin.size()>=64,"runtime load requires a CUDA cubin ELF image or PTX text");
  require(!cubin.empty() && !entry.empty() && entry.find('\0')==std::string::npos,"kernel binary and entry must be nonempty");
  const auto parameters=parameterWidths.size();
  require(parameters>0 && parameters<=64,"unsupported kernel parameter count");
  for(auto width:parameterWidths)
    require(width==0 || width==1 || width==2 || width==4 || width==8,"unsupported kernel parameter width");
  auto key=std::make_pair(cubin,entry);
  auto existing=impl->kernels.find(key);
  if (existing!=impl->kernels.end()) {
    require(existing->second->impl->parameterWidths==parameterWidths,"cached kernel parameter contract differs");
    ++s.counters["module_hits"];return existing->second;
  }
  ContextScope current(*s.driver,s.context);auto &d=*s.driver;
  auto p=std::make_unique<Kernel::Impl>();p->state=impl->state;p->name=entry;p->parameterWidths=parameterWidths;
  auto kernel=std::shared_ptr<Kernel>(new Kernel(std::move(p)));
  // PTX is a NUL-terminated driver input. Keep the stored/hashable source exact.
  std::string image=cubin;
  if(ptx) image.push_back('\0');
  d.check(d.cuModuleLoadDataEx(&kernel->impl->module,image.data(),0,nullptr,nullptr),"cuModuleLoadDataEx");
  d.check(d.cuModuleGetFunction(&kernel->impl->function,kernel->impl->module,entry.c_str()),"cuModuleGetFunction");
  auto function=kernel->impl->function;
  std::size_t expectedOffset=0;
  for (unsigned i=0;i<parameters;++i) { std::size_t offset=0,size=0;
    const std::size_t width=parameterWidths[i] ? parameterWidths[i] : sizeof(DevicePointer);
    expectedOffset=(expectedOffset+width-1)/width*width;
    d.check(d.cuFuncGetParamInfo(function,i,&offset,&size),"cuFuncGetParamInfo");
    require(offset==expectedOffset && size==width,"kernel parameter layout differs from the declared ABI");
    expectedOffset+=width; }
  std::size_t offset=0,size=0;
  require(d.cuFuncGetParamInfo(function,parameters,&offset,&size)==1,"kernel parameter count differs from the declared ABI");
  int maxDynamicShared=0,maxThreads=0;
  // The driver accounts for statically allocated shared memory in this limit.
  // Attribute 8 is CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES. We do not
  // silently opt in to a larger per-block allocation than the module requests.
  d.check(d.cuFuncGetAttribute(&maxDynamicShared,8,function),"cuFuncGetAttribute(max dynamic shared)");
  d.check(d.cuFuncGetAttribute(&maxThreads,0,function),"cuFuncGetAttribute(max threads)");
  require(maxDynamicShared>=0 && maxThreads>0,"invalid CUDA kernel resource limits");
  kernel->impl->maxThreads=maxThreads;
  kernel->impl->maxDynamicSharedBytes=maxDynamicShared;
  impl->kernels.emplace(std::move(key),kernel);++s.counters["module_loads"];return kernel;
}
void Session::validateLaunch(const std::shared_ptr<Kernel> &kernel,
    const std::array<unsigned,3> &grid,const std::array<unsigned,3> &block,unsigned sharedBytes) const {
  auto &s=*impl->state;
  require(kernel && kernel->impl->state==impl->state,"kernel belongs to a different CUDA execution session");
  require(sharedBytes<=kernel->impl->maxDynamicSharedBytes,"dynamic shared memory exceeds the compiled kernel configured limit");
  std::uint64_t threads=1;
  for(unsigned i=0;i<3;++i) { require(grid[i] && grid[i]<=s.maxGrid[i],"launch grid exceeds device limits");
    require(block[i] && block[i]<=s.maxBlock[i],"launch block exceeds device limits");threads*=block[i]; }
  require(threads<=s.maxThreads && threads<=kernel->impl->maxThreads,"launch thread count exceeds kernel/device limits");
}
std::shared_ptr<Event> Session::completion() {
  auto &s=*impl->state;
  auto p=std::make_unique<Event::Impl>();p->state=impl->state;
  if(s.freeEvents.empty()) {
    s.driver->check(s.driver->cuEventCreate(&p->event,2),"cuEventCreate");
    ++s.counters["event_creations"];
  } else {
    p->event=s.freeEvents.back();s.freeEvents.pop_back();++s.counters["event_reuses"];
  }
  return std::shared_ptr<Event>(new Event(std::move(p)));
}
std::shared_ptr<Event> Session::submit(const std::shared_ptr<Kernel> &kernel,
    const std::vector<KernelArgument> &arguments,std::array<unsigned,3> grid,
    std::array<unsigned,3> block,unsigned sharedBytes) {
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();
  validateLaunch(kernel,grid,block,sharedBytes);
  require(arguments.size()==kernel->impl->parameterWidths.size(),"kernel argument count differs from its ABI");
  std::vector<DevicePointer> storage(arguments.size());
  std::vector<void *> parameters;
  std::vector<std::shared_ptr<DeviceBuffer>> buffers;
  for(std::size_t index=0;index<arguments.size();++index) {
    const auto &argument=arguments[index];
    const auto width=kernel->impl->parameterWidths[index];
    if(width) {
      require(!argument.buffer && argument.scalar.size()==width,"scalar argument byte width differs from its ABI");
      std::memcpy(&storage[index],argument.scalar.data(),width);
    } else {
      require(argument.scalar.empty(),"pointer argument requires a device buffer or null");
      const auto &buffer=argument.buffer;
      require(!buffer || buffer->impl->state==impl->state,"device tensor belongs to a different CUDA execution session");
      storage[index]=buffer ? buffer->impl->pointer : 0;
      if(buffer) buffers.push_back(buffer);
    }
    parameters.push_back(&storage[index]);
  }
  reap(); ContextScope current(*s.driver,s.context);auto &d=*s.driver;
  auto event=completion();event->impl->buffers=std::move(buffers);event->impl->kernel=kernel;
  d.check(d.cuLaunchKernel(kernel->impl->function,grid[0],grid[1],grid[2],block[0],block[1],block[2],
                         sharedBytes,s.stream,parameters.data(),nullptr),"cuLaunchKernel");
  ++s.counters["launches"];
  int status=d.cuEventRecord(event->impl->event,s.stream);
  if(status) {
    // Keep unrecorded completions alive if the stream cannot be drained.
    try { s.synchronize();event->impl->finish(); }
    catch(...) { impl->pending.push_back(event);throw; }
    d.check(status,"cuEventRecord");
  }
  event->impl->recorded=true;
  impl->pending.push_back(event);return event;
}

namespace {
void assignGraphSlots(pass::AllocationPlan &allocation, std::vector<std::size_t> &slotOfBuffer,
                      const std::set<std::size_t> &publicPorts,
                      const std::vector<std::size_t> &lastUse) {
  const auto &plan = allocation.plan;
  const auto count = plan.steps.size(), absent = std::numeric_limits<std::size_t>::max();
  allocation.graph = true;
  allocation.dependencies.resize(count);
  std::vector<std::size_t> producer(plan.buffer_count, absent), accessCount(plan.buffer_count);
  std::vector<std::vector<std::size_t>> accesses(count), released(count);
  std::vector<bool> hasSuccessor(count);
  for (std::size_t i = 0; i < count; ++i) {
    auto &parents = allocation.dependencies[i];
    for (auto buffer : plan.steps[i].inputs)
      if (producer[buffer] != absent) parents.push_back(producer[buffer]);
    std::sort(parents.begin(), parents.end());
    parents.erase(std::unique(parents.begin(), parents.end()), parents.end());
    for (auto predecessor : parents) hasSuccessor[predecessor] = true;
    for (auto buffer : plan.steps[i].outputs) producer[buffer] = i;
    auto &used = accesses[i];used = plan.steps[i].inputs;
    used.insert(used.end(), plan.steps[i].outputs.begin(), plan.steps[i].outputs.end());
    std::sort(used.begin(), used.end());used.erase(std::unique(used.begin(), used.end()), used.end());
    for (auto buffer : used) ++accessCount[buffer];
  }
  // A terminal last access cannot precede a later writer in this DAG.
  for (const auto &step : plan.steps)
    for (auto buffer : step.outputs)
      if (!publicPorts.count(buffer) && lastUse[buffer] + 1 < count && hasSuccessor[lastUse[buffer]])
        released[lastUse[buffer] + 1].push_back(buffer);
  std::vector<std::size_t> visited(count, count), owner, epoch, remaining, ancestors, reusable;
  std::set<std::pair<std::size_t, std::size_t>> candidates;
  for (std::size_t i = 0; i < count; ++i) {
    for (auto buffer : released[i]) candidates.emplace(producer[buffer], slotOfBuffer[buffer]);
    ancestors = allocation.dependencies[i];reusable.clear();
    for (auto buffer : plan.steps[i].outputs) {
      if (publicPorts.count(buffer)) continue;
      // Visit actual ancestors until every access of one old slot occupant
      // has been seen. This keeps independent branches independent and stops
      // close to the current node for ordinary chains.
      // No candidate can have an access older than its producer. This bound
      // prevents independent branches from repeatedly walking a deep prefix.
      const auto firstAccess = candidates.empty() ? count : candidates.begin()->first;
      while (reusable.empty() && !ancestors.empty()) {
        auto predecessor = ancestors.back();ancestors.pop_back();
        if (predecessor < firstAccess || visited[predecessor] == i) continue;
        visited[predecessor] = i;
        for (auto old : accesses[predecessor]) {
          auto slot = slotOfBuffer[old];
          if (slot == absent || owner[slot] != old || lastUse[old] >= i) continue;
          if (epoch[slot] != i) { epoch[slot] = i;remaining[slot] = accessCount[old]; }
          if (--remaining[slot] == 0) reusable.push_back(slot);
        }
        const auto &parents = allocation.dependencies[predecessor];
        ancestors.insert(ancestors.end(), parents.begin(), parents.end());
      }
      std::size_t slot;
      if (reusable.empty()) {
        slot = owner.size();owner.push_back(buffer);epoch.push_back(count);remaining.push_back(0);
        allocation.slot_capacity_bytes.push_back(allocation.buffer_sizes[buffer]);
      } else {
        slot = reusable.back();reusable.pop_back();
        candidates.erase({producer[owner[slot]], slot});owner[slot] = buffer;
        allocation.slot_capacity_bytes[slot] = std::max(allocation.slot_capacity_bytes[slot], allocation.buffer_sizes[buffer]);
      }
      slotOfBuffer[buffer] = slot;
      allocation.assignments.push_back({buffer, slot});
    }
  }
}
} // namespace

std::shared_ptr<PreparedProgram> Session::prepareProgram(
    std::vector<std::size_t> sizes,std::vector<std::size_t> inputs,
    std::vector<std::size_t> outputs,std::vector<PlanStep> steps,bool graph) {
  auto &s=*impl->state;
  pass::AllocationPlan allocation;
  const std::size_t noSlot=std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> slotOfBuffer, scratch(steps.size(),noSlot);
  std::vector<std::shared_ptr<detail::CUBLASLtMatmul>> matmuls(steps.size());
  std::vector<std::array<std::size_t,3>> matrixBytes(steps.size());
  {
    std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();
    require(!graph || s.driver->graphs(),"CUDA driver does not support graph replay");
    require(!outputs.empty(),"program must have at least one output");
    auto index=[&](std::size_t value) { require(value<sizes.size(),"program buffer index is out of range"); };
    std::set<std::size_t> available;
    for(auto value:inputs) { index(value);require(available.insert(value).second,"program inputs must be unique"); }
    for(auto value:outputs) index(value);
    allocation.plan.public_inputs=inputs;allocation.plan.public_outputs=outputs;
    allocation.buffer_sizes=sizes;
    for(std::size_t stepIndex=0;stepIndex<steps.size();++stepIndex) {
      const auto &step=steps[stepIndex];
      require(bool(step.kernel)!=bool(step.matmul),"program step requires one kernel or matmul description");
      std::set<std::size_t> ports;
      for(auto value:step.inputs) { index(value);ports.insert(value); }
      for(auto value:step.outputs) {
        index(value);require(!available.count(value),"program step overwrites an existing logical buffer");
        require(ports.insert(value).second,"program step repeats or aliases an output");
      }
      allocation.plan.steps.push_back({step.inputs,step.outputs});
      if(step.matmul) {
        require(step.inputs.size()==2 && step.outputs.size()==1 && step.arguments.empty(),
                "matmul step requires two inputs, one output and no kernel arguments");
        require(!graph || s.driver->libraryGraphs(),"CUDA driver does not support library graph capture");
        checkTypes({step.matmul->elementType});
        auto bytes=detail::matmulBufferBytes(*step.matmul);matrixBytes[stepIndex]=bytes;
        require(sizes[step.inputs[0]]==bytes[0] && sizes[step.inputs[1]]==bytes[1] &&
                sizes[step.outputs[0]]==bytes[2],"matmul buffer byte counts differ from its dimensions");
        if(bytes[2] && step.matmul->k) {
          ContextScope current(*s.driver,s.context);
          auto operation=detail::CUBLASLtMatmul::prepare(*step.matmul,s.cublas);
          ++s.counters["matmul_preparations"];
          if(auto workspace=operation->workspaceBytes()) {
            // Scratch is a private logical buffer with this step's lifetime.
            scratch[stepIndex]=allocation.buffer_sizes.size();
            allocation.plan.steps.back().outputs.push_back(scratch[stepIndex]);
            allocation.buffer_sizes.push_back(workspace);
          }
          matmuls[stepIndex]=std::move(operation);
        }
      } else {
        validateLaunch(step.kernel,step.grid,step.block,step.sharedBytes);
        require(step.arguments.size()==step.kernel->impl->parameterWidths.size(),"program kernel argument count differs from its ABI");
        std::set<std::size_t> bound;
        for(std::size_t i=0;i<step.arguments.size();++i) {
          const auto &argument=step.arguments[i];auto width=step.kernel->impl->parameterWidths[i];
          if(width) require(argument.buffer==-1 && argument.scalar.size()==width,"program scalar argument differs from its ABI");
          else {
            require(argument.scalar.empty() && argument.buffer>=-1,"program pointer argument differs from its ABI");
            if(argument.buffer>=0) {
              index(argument.buffer);require(ports.count(argument.buffer),"program argument is not a step port");
              bound.insert(argument.buffer);
            }
          }
        }
        require(bound==ports,"program step port is not bound to a kernel parameter");
      }
      available.insert(step.outputs.begin(),step.outputs.end());
    }
    require(available.size()==sizes.size(),"program contains an undefined buffer");
    allocation.plan.buffer_count=allocation.buffer_sizes.size();
    slotOfBuffer.assign(allocation.plan.buffer_count,noSlot);
    // Every producer is an access, including unconsumed tensors and scratch.
    std::vector<std::size_t> lastUse(allocation.plan.buffer_count,0);
    for(std::size_t i=0;i<steps.size();++i) {
      for(auto buffer:allocation.plan.steps[i].inputs) lastUse[buffer]=std::max(lastUse[buffer],i);
      for(auto buffer:allocation.plan.steps[i].outputs) lastUse[buffer]=std::max(lastUse[buffer],i);
    }
    std::set<std::size_t> publicPorts(inputs.begin(),inputs.end());
    publicPorts.insert(outputs.begin(),outputs.end());
    std::vector<std::size_t> slotLastUse;
    if(graph) assignGraphSlots(allocation,slotOfBuffer,publicPorts,lastUse);
    else for(std::size_t i=0;i<steps.size();++i)
      for(auto buffer:allocation.plan.steps[i].outputs) {
        if(publicPorts.count(buffer)) continue;
        std::size_t slot=0;
        while(slot<slotLastUse.size() && slotLastUse[slot]>=i) ++slot;
        if(slot==slotLastUse.size()) {
          slotLastUse.push_back(lastUse[buffer]);
          allocation.slot_capacity_bytes.push_back(allocation.buffer_sizes[buffer]);
        } else {
          slotLastUse[slot]=lastUse[buffer];
          allocation.slot_capacity_bytes[slot]=std::max(allocation.slot_capacity_bytes[slot],allocation.buffer_sizes[buffer]);
        }
        slotOfBuffer[buffer]=slot;
        allocation.assignments.push_back({buffer,slot});
      }
  }
  auto decision=pass::checkAllocationPlan(allocation);
  require(decision.verdict==pass::Verdict::Pass,"ASK execution plan validation failed: "+decision.explanation);
  std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();
  using Program=PreparedProgram::Impl;
  auto p=std::make_unique<Program>();p->session=shared_from_this();p->state=impl->state;
  p->sizes=std::move(sizes);p->inputs=std::move(inputs);p->outputs=std::move(outputs);
  std::set<std::size_t> publicPorts(p->inputs.begin(),p->inputs.end());
  publicPorts.insert(p->outputs.begin(),p->outputs.end());
  for(auto index:std::set<std::size_t>(p->outputs.begin(),p->outputs.end())) {
    if(std::find(p->inputs.begin(),p->inputs.end(),index)!=p->inputs.end()) continue;
    p->allocatedOutputs.push_back(index);
    for(auto other:publicPorts) if(other!=index) p->disjointPorts.emplace_back(index,other);
  }
  for(std::size_t port=0;port<p->outputs.size();++port)
    if(std::find(p->allocatedOutputs.begin(),p->allocatedOutputs.end(),p->outputs[port])!=p->allocatedOutputs.end())
      p->writableOutputs.push_back(port);
  p->slotCapacities=std::move(allocation.slot_capacity_bytes);
  s.counters["prepared_internal_slots"]+=p->slotCapacities.size();
  for(auto capacity:p->slotCapacities) s.counters["prepared_internal_bytes"]+=capacity;
  p->steps.reserve(steps.size());p->graph=graph && !steps.empty();
  std::map<std::size_t,unsigned> alignments;
  for(std::size_t i=0;i<steps.size();++i) {
    auto &source=steps[i];Program::Step step;
    if(graph) {
      step.dependencies=std::move(allocation.dependencies[i]);
      p->graphEdges+=step.dependencies.size();
    }
    const auto bind=[&](std::size_t parameter,std::size_t buffer,unsigned alignment=1) {
      auto slot=slotOfBuffer[buffer];
      if(slot==noSlot) {
        step.publicBindings.push_back({parameter,buffer});
        if(alignment>1) alignments[buffer]=std::max(alignments[buffer],alignment);
      } else step.internalBindings.push_back({parameter,slot});
    };
    if(source.matmul) {
      step.constants.resize(4);step.matmul=std::move(matmuls[i]);step.outputBytes=matrixBytes[i][2];
      step.execute=step.outputBytes ? (source.matmul->k ? &Program::launchMatmul : &Program::zeroMatmul) : &Program::emptyMatmul;
      step.addGraph=&Program::addLibrary;step.updateGraph=&Program::updateLibrary;
      if(step.outputBytes) {
        const auto alignment=detail::matmulElementBytes(source.matmul->elementType);
        bind(2,source.outputs[0],alignment);
        if(source.matmul->k) {
          bind(0,source.inputs[0],alignment);bind(1,source.inputs[1],alignment);
          if(scratch[i]!=noSlot) bind(3,scratch[i]);
        }
      }
    } else {
      step.kernel=std::move(source.kernel);
      step.execute=&Program::launchKernel;step.addGraph=&Program::addKernel;step.updateGraph=&Program::updateKernel;
      step.launch={step.kernel->impl->function,source.grid[0],source.grid[1],source.grid[2],
                   source.block[0],source.block[1],source.block[2],source.sharedBytes,nullptr,nullptr};
      step.constants.resize(source.arguments.size());
      for(std::size_t j=0;j<source.arguments.size();++j) {
        const auto &argument=source.arguments[j];
        if(!argument.scalar.empty()) std::memcpy(&step.constants[j],argument.scalar.data(),argument.scalar.size());
        else if(argument.buffer>=0) bind(j,argument.buffer);
      }
    }
    if(!step.publicBindings.empty()) p->publicSteps.push_back(p->steps.size());
    p->steps.push_back(std::move(step));
  }
  p->alignedPorts.assign(alignments.begin(),alignments.end());
  return std::shared_ptr<PreparedProgram>(new PreparedProgram(std::move(p)));
}

PreparedProgram::PreparedProgram(std::unique_ptr<Impl> impl):impl(std::move(impl)){}
PreparedProgram::~PreparedProgram()=default;
void PreparedProgram::bindPort(std::vector<std::shared_ptr<DeviceBuffer>> &bound,
    std::size_t index,const std::shared_ptr<DeviceBuffer> &buffer) const {
  require(buffer && buffer->impl->state==impl->state,"program tensor belongs to a different CUDA execution session");
  require(buffer->size()==impl->sizes[index],"program tensor byte count differs from its compiled type");
  require(!bound[index] || bound[index]->pointer()==buffer->pointer(),"repeated logical buffers must use the same device storage");
  bound[index]=buffer;
}
std::shared_ptr<Event> PreparedProgram::submit(
    const std::vector<std::shared_ptr<DeviceBuffer>> &inputs,
    const std::vector<std::shared_ptr<DeviceBuffer>> &outputs) {
  auto session=impl->session.lock();require(bool(session),"program execution session no longer exists");
  auto &s=*session->impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();
  require(inputs.size()==impl->inputs.size() && outputs.size()==impl->outputs.size(),"program input/output count differs from its contract");
  std::vector<std::shared_ptr<DeviceBuffer>> bound(impl->sizes.size());
  for(std::size_t i=0;i<inputs.size();++i) bindPort(bound,impl->inputs[i],inputs[i]);
  for(std::size_t i=0;i<outputs.size();++i) bindPort(bound,impl->outputs[i],outputs[i]);
  for(auto port:impl->writableOutputs)
    require(!outputs[port]->readonly(),"program output tensor is read-only");
  for(const auto &[index,other]:impl->disjointPorts) {
    const auto &output=bound[index];const auto &buffer=bound[other];
    require(!output->size() || !buffer->size() || output->pointer()+output->size()<=buffer->pointer() ||
            buffer->pointer()+buffer->size()<=output->pointer(),"distinct program output buffers must not overlap other ports");
  }
  return enqueue(session,std::move(bound));
}
std::shared_ptr<Event> PreparedProgram::enqueue(const std::shared_ptr<Session> &session,
    std::vector<std::shared_ptr<DeviceBuffer>> bound) {
  auto &s=*session->impl->state;
  for(const auto &[index,alignment]:impl->alignedPorts)
    require(bound[index]->pointer()%alignment==0,"matmul buffer address is not aligned to its element type");
  session->reap();
  std::shared_ptr<Impl::Workspace> workspace;
  for(const auto &candidate:impl->workspaces) if(!candidate->busy) { workspace=candidate;break; }
  if(workspace) ++s.counters["workspace_reuses"];
  else {
    workspace=std::make_shared<Impl::Workspace>();workspace->state=session->impl->state;
    workspace->buffers.reserve(impl->slotCapacities.size());
    for(auto capacity:impl->slotCapacities)
      workspace->buffers.push_back(session->allocate(capacity));
    workspace->values.resize(impl->steps.size());
    workspace->parameters.resize(impl->steps.size());
    workspace->launches.reserve(impl->steps.size());
    for(std::size_t i=0;i<impl->steps.size();++i) {
      const auto &step=impl->steps[i];
      auto &values=workspace->values[i];values=step.constants;
      for(const auto &binding:step.internalBindings)
        values[binding.parameter]=workspace->buffers[binding.buffer]->pointer();
      auto &parameters=workspace->parameters[i];parameters.reserve(values.size());
      for(auto &value:values) parameters.push_back(&value);
      auto launch=step.launch;launch.parameters=parameters.data();
      workspace->launches.push_back(launch);
    }
    impl->workspaces.push_back(workspace);
  }
  // The verified plan fixes ABI layout, scalar bytes and internal addresses.
  // Submissions patch only caller-owned ports in this idle workspace.
  for(auto i:impl->publicSteps) {
    for(const auto &binding:impl->steps[i].publicBindings)
      workspace->values[i][binding.parameter]=bound[binding.buffer]->pointer();
  }
  const auto &nodes=workspace->launches;
  ContextScope current(*s.driver,s.context);auto &d=*s.driver;
  if(impl->graph) {
    if(!workspace->executable) {
      // Retain a partially constructed graph only within this attempt.
      if(workspace->graph) { d.cuGraphDestroy(workspace->graph);workspace->graph=nullptr;workspace->nodes.clear(); }
      d.check(d.cuGraphCreate(&workspace->graph,0),"cuGraphCreate");
      std::vector<Handle> dependencies;
      for(std::size_t i=0;i<impl->steps.size();++i) {
        const auto &step=impl->steps[i];
        dependencies.clear();
        for(auto predecessor:step.dependencies) dependencies.push_back(workspace->nodes[predecessor]);
        auto node=step.addGraph(step,*workspace,i,s,dependencies.data(),dependencies.size());
        workspace->nodes.push_back(node);
      }
      d.check(d.cuGraphInstantiate(&workspace->executable,workspace->graph,nullptr,nullptr,0),"cuGraphInstantiate");
      workspace->graphValues=workspace->values;
      ++s.counters["graph_instantiations"];
      s.counters["graph_nodes"]+=impl->steps.size();s.counters["graph_edges"]+=impl->graphEdges;
    } else for(auto i:impl->publicSteps) if(workspace->graphValues[i]!=workspace->values[i]) {
      const auto &step=impl->steps[i];step.updateGraph(step,*workspace,i,s);
      workspace->graphValues[i]=workspace->values[i];++s.counters["graph_node_updates"];
    }
  }
  auto event=session->completion();event->impl->buffers=std::move(bound);event->impl->program=shared_from_this();
  workspace->busy=true;event->impl->releaseWorkspace=[workspace]{workspace->busy=false;};
  try {
    if(impl->graph) {
      d.check(d.cuGraphLaunch(workspace->executable,s.stream),"cuGraphLaunch");++s.counters["graph_launches"];
      s.counters["launches"]+=nodes.size();
    } else for(std::size_t i=0;i<impl->steps.size();++i) {
      const auto &step=impl->steps[i];step.execute(step,*workspace,i,s);
      ++s.counters["launches"];
    }
    d.check(d.cuEventRecord(event->impl->event,s.stream),"cuEventRecord");
    event->impl->recorded=true;
  } catch(...) {
    // A later launch can fail after preceding steps were queued. Retain every
    // argument and intermediate until that partial submission has completed.
    try { s.synchronize();event->impl->finish(); }
    catch(...) { session->impl->pending.push_back(event);throw; }
    throw;
  }
  ++s.counters["program_submissions"];session->impl->pending.push_back(event);return event;
}
std::pair<std::vector<std::shared_ptr<DeviceBuffer>>,std::shared_ptr<Event>> PreparedProgram::run(
    const std::vector<std::shared_ptr<DeviceBuffer>> &inputs) {
  auto session=impl->session.lock();require(bool(session),"program execution session no longer exists");
  auto &s=*session->impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();
  require(inputs.size()==impl->inputs.size(),"program input count differs from its contract");
  std::vector<std::shared_ptr<DeviceBuffer>> bound(impl->sizes.size()),outputs;
  for(std::size_t i=0;i<inputs.size();++i) bindPort(bound,impl->inputs[i],inputs[i]);
  for(auto index:impl->allocatedOutputs) bound[index]=session->allocate(impl->sizes[index]);
  outputs.reserve(impl->outputs.size());
  for(auto index:impl->outputs) outputs.push_back(bound[index]);
  auto event=enqueue(session,std::move(bound));return {std::move(outputs),std::move(event)};
}
void Session::waitForExternalStream(std::uint64_t stream) {
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();
  require(stream!=0,"CUDA Array Interface stream 0 is ambiguous");
  if(stream==reinterpret_cast<std::uintptr_t>(s.stream)) return;
  ContextScope current(*s.driver,s.context);auto &d=*s.driver;Handle event=nullptr;
  d.check(d.cuEventCreate(&event,2),"cuEventCreate(external stream)");
  try {
    d.check(d.cuEventRecord(event,reinterpret_cast<Handle>(static_cast<std::uintptr_t>(stream))),"cuEventRecord(external stream)");
    d.check(d.cuStreamWaitEvent(s.stream,event,0),"cuStreamWaitEvent");
    ++s.counters["external_stream_waits"];
  } catch (...) { d.cuEventDestroy(event);throw; }
  d.cuEventDestroy(event);
}
void Session::reap() {
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);
  impl->pending.erase(std::remove_if(impl->pending.begin(),impl->pending.end(),
      [](const auto &event){return event->query();}),impl->pending.end());
}
void Session::synchronize() {
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();
  ContextScope current(*s.driver,s.context);s.synchronize();reap();
}
void Session::close() {
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);if(s.closed)return;
  ContextScope current(*s.driver,s.context);std::exception_ptr failure;
  try { s.synchronize(); } catch (...) { failure=std::current_exception(); }
  // A failed drain is not proof of completion. Leave resources owned by the
  // session so an explicit caller may retry close rather than recycling them.
  if(failure)std::rethrow_exception(failure);
  reap();
  s.closed=true;impl->pending.clear();impl->kernels.clear();
  s.clearPools();
  if(s.stream) { s.driver->cuStreamDestroy(s.stream);s.stream=nullptr; }
}
bool Session::closed() const { auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);return s.closed; }
std::uint64_t Session::stream() const { auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();return reinterpret_cast<std::uintptr_t>(s.stream); }
int Session::device() const {return impl->state->device;}
std::array<int,2> Session::capability() const {return {impl->state->major,impl->state->minor};}
std::string Session::deviceName() const {return impl->state->name;}
std::map<std::string,std::uint64_t> Session::stats() const {
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);auto result=s.counters;
  result["pending_events"]=impl->pending.size();result["resident_modules"]=impl->kernels.size();
  result["pooled_bytes"]=s.pooledBytes;result["pooled_events"]=s.freeEvents.size();return result;
}
DeviceBuffer::DeviceBuffer(std::unique_ptr<Impl> impl):impl(std::move(impl)){}
DeviceBuffer::~DeviceBuffer(){
  if(!impl->owned || !impl->pointer)return;auto &s=*impl->state;
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  try{
    ContextScope current(*s.driver,s.context);
    auto bytes=std::max<std::size_t>(1,impl->bytes);
    if(!s.closed && bytes<=SessionState::poolLimit-s.pooledBytes) {
      s.freeBuffers[impl->bytes].push_back(impl->pointer);s.pooledBytes+=bytes;
    } else if(!s.driver->cuMemFree(impl->pointer)) s.counters["resident_device_bytes"]-=bytes;
  }catch(...){}
}
std::uint64_t DeviceBuffer::pointer()const{return impl->pointer;}
std::size_t DeviceBuffer::size()const{return impl->bytes;}
bool DeviceBuffer::readonly()const{return impl->readonly;}
void DeviceBuffer::copyFromHost(const void *data,std::size_t bytes){
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();
  require(!impl->readonly,"device tensor is read-only");require(bytes==impl->bytes,"host byte count must equal device buffer size");
  if(!bytes)return;ContextScope current(*s.driver,s.context);HostStorage staging(*s.driver,bytes);
  std::memcpy(staging.pointer,data,bytes);
  s.driver->check(s.driver->cuMemcpyHtoDAsync(impl->pointer,staging.pointer,bytes,s.stream),"cuMemcpyHtoDAsync");
  s.synchronize();++s.counters["host_to_device_copies"];
}
std::string DeviceBuffer::copyToHost()const{
  std::string output(impl->bytes,'\0');copyToHost(output.data(),output.size());return output;
}
void DeviceBuffer::copyToHost(void *data,std::size_t bytes)const{
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();
  require(bytes==impl->bytes,"host byte count must equal device buffer size");
  if(!impl->bytes)return;ContextScope current(*s.driver,s.context);HostStorage staging(*s.driver,impl->bytes);
  s.driver->check(s.driver->cuMemcpyDtoHAsync(staging.pointer,impl->pointer,impl->bytes,s.stream),"cuMemcpyDtoHAsync");
  s.synchronize();++s.counters["device_to_host_copies"];
  std::memcpy(data,staging.pointer,impl->bytes);
}
Kernel::Kernel(std::unique_ptr<Impl> impl):impl(std::move(impl)){}
Kernel::~Kernel(){if(!impl->module)return;auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);
  try{ContextScope current(*s.driver,s.context);s.driver->cuModuleUnload(impl->module);}catch(...){} }
std::string Kernel::entry()const{return impl->name;}
unsigned Kernel::maxThreads()const{return impl->maxThreads;}
Event::Event(std::unique_ptr<Impl> impl):impl(std::move(impl)){}
Event::~Event(){
  if(impl->complete)return;
  try { wait(); }
  catch(...) {
    // Destruction after an unrecoverable driver error cannot prove completion.
    // Quarantine the remaining native resources instead of freeing storage
    // still referenced by queued GPU work. Explicit wait/close reports errors.
    (void)impl.release();
  }
}
bool Event::query(){
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);if(impl->complete)return true;
  ContextScope current(*s.driver,s.context);
  if(!impl->recorded) { s.synchronize();impl->finish();return true; }
  int status=s.driver->cuEventQuery(impl->event);
  if(status==notReady)return false;s.driver->check(status,"cuEventQuery");impl->finish();return true;
}
void Event::wait(){
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);if(impl->complete)return;
  ContextScope current(*s.driver,s.context);
  if(impl->recorded) s.driver->check(s.driver->cuEventSynchronize(impl->event),"cuEventSynchronize");
  else s.synchronize();
  ++s.counters["event_waits"];impl->finish();
}
} // namespace ask::runtime
