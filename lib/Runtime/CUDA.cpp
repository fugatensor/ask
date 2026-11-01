#include "ask/Runtime/CUDA.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <set>
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
constexpr int notReady = 600;
void require(bool condition, const std::string &message) {
  if (!condition) throw CUDAError(message);
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
  CUDA_FUNCTION(cuPointerGetAttribute, (void *, int, DevicePointer));
  CUDA_FUNCTION(cuMemGetAddressRange, (DevicePointer *, std::size_t *, DevicePointer));
  CUDA_FUNCTION(cuEventCreate, (Handle *, unsigned));
  CUDA_FUNCTION(cuEventRecord, (Handle, Handle));
  CUDA_FUNCTION(cuEventQuery, (Handle));
  CUDA_FUNCTION(cuEventSynchronize, (Handle));
  CUDA_FUNCTION(cuEventDestroy, (Handle));
  CUDA_FUNCTION(cuLaunchKernel, (Handle, unsigned, unsigned, unsigned, unsigned, unsigned,
                                unsigned, unsigned, Handle, void **, void **));
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
      B(cuPointerGetAttribute); V(cuMemGetAddressRange);
      B(cuEventCreate); B(cuEventRecord); B(cuEventQuery); B(cuEventSynchronize); V(cuEventDestroy);
      B(cuLaunchKernel); B(cuGetErrorName); B(cuGetErrorString);
#undef B
#undef V
    } catch (...) { unload(); throw; }
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
  mutable std::recursive_mutex mutex;
  Handle context = nullptr, stream = nullptr;
  int device = 0, major = 0, minor = 0;
  std::string name;
  bool closed = false;
  std::array<unsigned,3> maxGrid{}, maxBlock{};
  unsigned maxThreads = 0;
  std::map<std::string,std::uint64_t> counters = {
      {"context_creations",0},{"primary_context_retains",0},{"stream_creations",0},
      {"module_loads",0},{"module_hits",0},{"allocations",0},{"launches",0},
      {"synchronizations",0},{"event_waits",0},{"external_stream_waits",0},
      {"host_to_device_copies",0},{"device_to_host_copies",0}};
  void checkOpen() const { require(!closed, "CUDA execution session is closed"); }
  void synchronize() {
    driver->check(driver->cuStreamSynchronize(stream), "cuStreamSynchronize");
    ++counters["synchronizations"];
  }
  ~SessionState() {
    if (!context) return;
    try {
      ContextScope current(*driver, context);
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
  unsigned parameters = 0, maxThreads = 0, maxDynamicSharedBytes = 0;
};
struct Event::Impl {
  std::shared_ptr<SessionState> state;
  Handle event = nullptr;
  bool complete = false;
  std::vector<std::shared_ptr<DeviceBuffer>> buffers;
  std::shared_ptr<Kernel> kernel;
  void finish() { complete = true; buffers.clear(); kernel.reset(); }
};
struct Session::Impl {
  std::shared_ptr<SessionState> state;
  // Module identity uses complete bytes, so no collision-prone ad-hoc hash is
  // needed here. The Python package has already validated its SHA256 receipt.
  std::map<std::pair<std::string,std::string>, std::shared_ptr<Kernel>> kernels;
  std::vector<std::shared_ptr<Event>> pending;
};

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
  s.driver->check(s.driver->cuMemAlloc(&p->pointer,std::max<std::size_t>(1,bytes)),"cuMemAlloc");
  ++s.counters["allocations"];
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
    const std::vector<std::string> &requiredTypes,int arch,unsigned parameters) {
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();checkTypes(requiredTypes);
  require(arch>=10 && s.major==arch/10 && s.minor>=arch%10,"CUDA cubin target is incompatible with the selected device");
  require(cubin.size()>=64 && cubin.compare(0,4,"\x7f" "ELF")==0,"runtime load requires a compiled CUDA cubin ELF image");
  require(!cubin.empty() && !entry.empty() && entry.find('\0')==std::string::npos,"kernel binary and entry must be nonempty");
  require(parameters>0 && parameters<=64,"unsupported kernel parameter count");
  auto key=std::make_pair(cubin,entry);
  auto existing=impl->kernels.find(key);
  if (existing!=impl->kernels.end()) {
    require(existing->second->impl->parameters==parameters,"cached kernel parameter contract differs");
    ++s.counters["module_hits"];return existing->second;
  }
  ContextScope current(*s.driver,s.context);auto &d=*s.driver;
  auto p=std::make_unique<Kernel::Impl>();p->state=impl->state;p->name=entry;p->parameters=parameters;
  auto kernel=std::shared_ptr<Kernel>(new Kernel(std::move(p)));
  d.check(d.cuModuleLoadDataEx(&kernel->impl->module,cubin.data(),0,nullptr,nullptr),"cuModuleLoadDataEx");
  d.check(d.cuModuleGetFunction(&kernel->impl->function,kernel->impl->module,entry.c_str()),"cuModuleGetFunction");
  auto function=kernel->impl->function;
  for (unsigned i=0;i<parameters;++i) { std::size_t offset=0,size=0;
    d.check(d.cuFuncGetParamInfo(function,i,&offset,&size),"cuFuncGetParamInfo");
    require(offset==i*sizeof(DevicePointer) && size==sizeof(DevicePointer),"kernel ABI requires consecutive 64-bit pointer parameters"); }
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
std::shared_ptr<Event> Session::submit(const std::shared_ptr<Kernel> &kernel,
    const std::vector<std::shared_ptr<DeviceBuffer>> &buffers,std::array<unsigned,3> grid,
    std::array<unsigned,3> block,unsigned sharedBytes) {
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();
  require(kernel && kernel->impl->state==impl->state,"kernel belongs to a different CUDA execution session");
  require(buffers.size()==kernel->impl->parameters,"kernel argument count differs from its ABI");
  require(sharedBytes<=kernel->impl->maxDynamicSharedBytes,"dynamic shared memory exceeds the compiled kernel configured limit");
  std::uint64_t threads=1;
  for(unsigned i=0;i<3;++i) { require(grid[i] && grid[i]<=s.maxGrid[i],"launch grid exceeds device limits");
    require(block[i] && block[i]<=s.maxBlock[i],"launch block exceeds device limits");threads*=block[i]; }
  require(threads<=s.maxThreads && threads<=kernel->impl->maxThreads,"launch thread count exceeds kernel/device limits");
  std::vector<DevicePointer> pointers; std::vector<void *> parameters;
  for(const auto &buffer:buffers) {
    require(!buffer || buffer->impl->state==impl->state,"device tensor belongs to a different CUDA execution session");
    pointers.push_back(buffer ? buffer->impl->pointer : 0);
  }
  for(auto &pointer:pointers) parameters.push_back(&pointer);
  reap(); ContextScope current(*s.driver,s.context);auto &d=*s.driver;
  auto p=std::make_unique<Event::Impl>();p->state=impl->state;p->buffers=buffers;p->kernel=kernel;
  auto event=std::shared_ptr<Event>(new Event(std::move(p)));
  d.check(d.cuEventCreate(&event->impl->event,2),"cuEventCreate");
  d.check(d.cuLaunchKernel(kernel->impl->function,grid[0],grid[1],grid[2],block[0],block[1],block[2],
                         sharedBytes,s.stream,parameters.data(),nullptr),"cuLaunchKernel");
  ++s.counters["launches"];
  int status=d.cuEventRecord(event->impl->event,s.stream);
  if(status) { s.synchronize();d.check(status,"cuEventRecord"); }
  impl->pending.push_back(event);return event;
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
  if(!failure) reap();
  s.closed=true;impl->pending.clear();impl->kernels.clear();
  if(s.stream) { s.driver->cuStreamDestroy(s.stream);s.stream=nullptr; }
  if(failure)std::rethrow_exception(failure);
}
bool Session::closed() const { auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);return s.closed; }
std::uint64_t Session::stream() const { auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();return reinterpret_cast<std::uintptr_t>(s.stream); }
int Session::device() const {return impl->state->device;}
std::array<int,2> Session::capability() const {return {impl->state->major,impl->state->minor};}
std::string Session::deviceName() const {return impl->state->name;}
std::map<std::string,std::uint64_t> Session::stats() const {
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);auto result=s.counters;
  result["pending_events"]=impl->pending.size();result["resident_modules"]=impl->kernels.size();return result;
}
DeviceBuffer::DeviceBuffer(std::unique_ptr<Impl> impl):impl(std::move(impl)){}
DeviceBuffer::~DeviceBuffer(){
  if(!impl->owned || !impl->pointer)return;auto &s=*impl->state;
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  try{ContextScope current(*s.driver,s.context);s.driver->cuMemFree(impl->pointer);}catch(...){}
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
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);s.checkOpen();
  if(!impl->bytes)return {};ContextScope current(*s.driver,s.context);HostStorage staging(*s.driver,impl->bytes);
  s.driver->check(s.driver->cuMemcpyDtoHAsync(staging.pointer,impl->pointer,impl->bytes,s.stream),"cuMemcpyDtoHAsync");
  s.synchronize();++s.counters["device_to_host_copies"];
  return std::string(static_cast<const char *>(staging.pointer),impl->bytes);
}
Kernel::Kernel(std::unique_ptr<Impl> impl):impl(std::move(impl)){}
Kernel::~Kernel(){if(!impl->module)return;auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);
  try{ContextScope current(*s.driver,s.context);s.driver->cuModuleUnload(impl->module);}catch(...){} }
std::string Kernel::entry()const{return impl->name;}
unsigned Kernel::maxThreads()const{return impl->maxThreads;}
Event::Event(std::unique_ptr<Impl> impl):impl(std::move(impl)){}
Event::~Event(){if(!impl->event)return;try{wait();}catch(...){}
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);
  try{ContextScope current(*s.driver,s.context);s.driver->cuEventDestroy(impl->event);}catch(...){} }
bool Event::query(){
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);if(impl->complete)return true;
  ContextScope current(*s.driver,s.context);int status=s.driver->cuEventQuery(impl->event);
  if(status==notReady)return false;s.driver->check(status,"cuEventQuery");impl->finish();return true;
}
void Event::wait(){
  auto &s=*impl->state;std::lock_guard<std::recursive_mutex> lock(s.mutex);if(impl->complete)return;
  ContextScope current(*s.driver,s.context);s.driver->check(s.driver->cuEventSynchronize(impl->event),"cuEventSynchronize");
  ++s.counters["event_waits"];impl->finish();
}
} // namespace ask::runtime
