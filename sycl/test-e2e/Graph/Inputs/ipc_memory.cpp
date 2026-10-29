// A USM device allocation exported by one process is imported by another, and
// both processes read and write it from graphs: nothing in the graph layers
// inspects pointer provenance, so an imported pointer is an ordinary kernel or
// memcpy argument. Each process verifies the values written by the other
// process's graph. The blocking std::system call orders the processes, so no
// IPC event is needed:
//
//   exporter: Data[i] = i
//   consumer: graph {Data[i] += 1; copy Data} replayed K times, checks i + K
//   exporter: graph {copy Data; Data[i] *= 2}, checks i + K
//   verifier: graph {copy Data}, checks 2 * (i + K)
//
// The imported pointer is baked into the executable graph, so it is closed
// only after the executable graph has been destroyed.
//
// With USE_ZE_IPC the IPC get/put/open/close calls are made directly through
// Level Zero instead of the SYCL IPC extension, mirroring what the UR Level
// Zero adapter does underneath.

#include "../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/properties/all_properties.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

#if defined(__linux__)
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif // defined(__linux__)

#ifdef USE_ZE_IPC
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <cstring>
#include <stdexcept>
#include <sys/syscall.h>
#endif // USE_ZE_IPC

namespace ipc = exp_ext::ipc;

#if defined(GRAPH_E2E_NATIVE_RECORDING) && defined(USE_ZE_IPC)
constexpr const char *CommsFile = "graph_nr_ze_ipc_memory_comms.bin";
#elif defined(GRAPH_E2E_NATIVE_RECORDING)
constexpr const char *CommsFile = "graph_nr_ipc_memory_comms.bin";
#else
constexpr const char *CommsFile = "graph_rr_ipc_memory_comms.bin";
#endif
constexpr size_t N = 1024;
constexpr int ConsumerReplays = 3;

#ifdef USE_ZE_IPC
// The Level Zero IPC handle only describes the whole allocation, while a SYCL
// USM pointer may be a sub-allocation of a pooled allocation, so the offset
// into the allocation is sent alongside the handle. The handle carries a file
// descriptor that is only valid in the exporting process, so its pid is sent
// too.
struct ZeIpcData {
  pid_t Pid;
  size_t Offset;
  ze_ipc_mem_handle_t Handle;
};

void checkZe(ze_result_t Result, const char *Call) {
  if (Result != ZE_RESULT_SUCCESS)
    throw std::runtime_error(std::string{Call} + " failed with " +
                             std::to_string(Result));
}

void *zeAllocBaseHelper(void *Ptr, ze_context_handle_t ZeCtx) {
  void *Base = nullptr;
  checkZe(zeMemGetAddressRange(ZeCtx, Ptr, &Base, nullptr),
          "zeMemGetAddressRange");
  return Base;
}

int duplicateFd(pid_t Pid, int RemoteFd) {
  int PidFd = syscall(SYS_pidfd_open, Pid, 0);
  if (PidFd == -1)
    throw std::runtime_error("pidfd_open failed");
  int LocalFd = syscall(SYS_pidfd_getfd, PidFd, RemoteFd, 0);
  close(PidFd);
  if (LocalFd == -1)
    throw std::runtime_error("pidfd_getfd failed");
  return LocalFd;
}

// Level Zero equivalents of the ipc::memory functions, exchanging a ZeIpcData
// as the handle data.
ipc::handle_data_t zeIpcGetHelper(void *Ptr, const context &Ctx) {
  auto ZeCtx = get_native<backend::ext_oneapi_level_zero>(Ctx);
  void *Base = zeAllocBaseHelper(Ptr, ZeCtx);
  ZeIpcData IpcData{
      getpid(),
      static_cast<size_t>(static_cast<char *>(Ptr) - static_cast<char *>(Base)),
      {}};
  checkZe(zeMemGetIpcHandle(ZeCtx, Base, &IpcData.Handle), "zeMemGetIpcHandle");
  auto *Bytes = reinterpret_cast<const std::byte *>(&IpcData);
  return {Bytes, Bytes + sizeof(IpcData)};
}

ZeIpcData toZeIpcData(const ipc::handle_data_t &HandleData) {
  if (HandleData.size() != sizeof(ZeIpcData))
    throw std::runtime_error("unexpected handle size " +
                             std::to_string(HandleData.size()));
  ZeIpcData IpcData;
  std::memcpy(&IpcData, HandleData.data(), sizeof(IpcData));
  return IpcData;
}

void zeIpcPutHelper(const ipc::handle_data_t &HandleData, const context &Ctx) {
  checkZe(zeMemPutIpcHandle(get_native<backend::ext_oneapi_level_zero>(Ctx),
                            toZeIpcData(HandleData).Handle),
          "zeMemPutIpcHandle");
}

void *zeIpcOpenHelper(const ipc::handle_data_t &HandleData, const context &Ctx,
                      const device &Dev) {
  ZeIpcData IpcData = toZeIpcData(HandleData);

  // Like the UR adapter, replace the exporter's file descriptor at the start
  // of the handle with a duplicate that is valid in this process.
  int RemoteFd = -1;
  std::memcpy(&RemoteFd, IpcData.Handle.data, sizeof(RemoteFd));
  int LocalFd = duplicateFd(IpcData.Pid, RemoteFd);
  std::memcpy(IpcData.Handle.data, &LocalFd, sizeof(LocalFd));

  void *Base = nullptr;
  ze_result_t Result =
      zeMemOpenIpcHandle(get_native<backend::ext_oneapi_level_zero>(Ctx),
                         get_native<backend::ext_oneapi_level_zero>(Dev),
                         IpcData.Handle, 0, &Base);
  close(LocalFd);
  checkZe(Result, "zeMemOpenIpcHandle");
  return static_cast<char *>(Base) + IpcData.Offset;
}

void zeIpcCloseHelper(void *Ptr, const context &Ctx) {
  auto ZeCtx = get_native<backend::ext_oneapi_level_zero>(Ctx);
  checkZe(zeMemCloseIpcHandle(ZeCtx, zeAllocBaseHelper(Ptr, ZeCtx)),
          "zeMemCloseIpcHandle");
}
#endif // USE_ZE_IPC

int consumerValue(size_t I) { return static_cast<int>(I) + ConsumerReplays; }
int exporterValue(size_t I) { return 2 * consumerValue(I); }

template <typename RecordFn>
void recordAndReplay(queue &Queue, int Replays, RecordFn Record) {
#ifdef GRAPH_E2E_NATIVE_RECORDING
  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};
#else
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
#endif

  Graph.begin_recording(Queue);
  Record();
  Graph.end_recording(Queue);

  auto ExecGraph = Graph.finalize();
  for (int I = 0; I < Replays; ++I)
    Queue.ext_oneapi_graph(ExecGraph);
  Queue.wait_and_throw();
}

template <typename ExpectedFn>
int countMismatches(const int *Got, ExpectedFn Expected,
                    const std::string &Name) {
  int Errors = 0;
  for (size_t I = 0; I < N; ++I)
    Errors += !check_value(I, Expected(I), Got[I], Name);
  return Errors;
}

int spawn(const char *Exe, const std::string &Mode) {
  std::string Cmd = std::string{Exe} + " " + Mode;
  std::cout << "Spawning: " << Cmd << "\n";
  int Rc = std::system(Cmd.c_str());
  if (!WIFEXITED(Rc) || WEXITSTATUS(Rc) != 0) {
    std::cerr << Mode << " process failed, raw status " << Rc << "\n";
    return 1;
  }
  return 0;
}

int spawner(char *argv[]) {
  queue Queue{property::queue::in_order{}};
  context Ctx = Queue.get_context();

#if defined(__linux__)
  // UMF currently requires ptrace permissions to be set for the spawner. As
  // such we need to set it until this limitation has been addressed.
  // https://github.com/oneapi-src/unified-memory-framework/tree/main?tab=readme-ov-file#level-zero-memory-provider
  if (Queue.get_backend() == backend::ext_oneapi_level_zero &&
      prctl(PR_SET_PTRACER, getppid()) == -1) {
    std::cout << "Failed to set ptracer permissions!" << std::endl;
    return 1;
  }
#endif // defined(__linux__)

  int *Data = malloc_device<int>(N, Queue);
  int *Snapshot = malloc_host<int>(N, Queue);
  Queue
      .parallel_for(range<1>{N},
                    [=](id<1> Idx) { Data[Idx] = static_cast<int>(Idx); })
      .wait();

  int Errors = 0;
  {
    // Handle layout: [size_t size][size bytes].
#ifdef USE_ZE_IPC
    ipc::handle_data_t Bytes = zeIpcGetHelper(Data, Ctx);
#else
    ipc::handle Handle = ipc::memory::get(Data, Ctx);
    ipc::handle_data_t Bytes = Handle.data();
#endif
    size_t HandleSize = Bytes.size();
    std::ofstream FS(CommsFile, std::ios::out | std::ios::binary);
    FS.write(reinterpret_cast<const char *>(&HandleSize), sizeof(size_t));
    FS.write(reinterpret_cast<const char *>(Bytes.data()), HandleSize);
    if (!FS) {
      std::cerr << "Failed to write handle to " << CommsFile << "\n";
      std::remove(CommsFile);
      return 1;
    }
    FS.close();

    Errors += spawn(argv[0], "consumer");
    if (Errors == 0) {
      recordAndReplay(Queue, 1, [&] {
        Queue.memcpy(Snapshot, Data, N * sizeof(int));
        Queue.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] *= 2; });
      });
      Errors += countMismatches(Snapshot, consumerValue, "ConsumerWrite");
      Errors += spawn(argv[0], "verifier");
    }

#ifdef USE_ZE_IPC
    zeIpcPutHelper(Bytes, Ctx);
#else
    ipc::memory::put(Handle, Ctx);
#endif
    std::remove(CommsFile);
  }

  free(Snapshot, Queue);
  free(Data, Queue);
  return Errors == 0 ? 0 : 2;
}

int consumer(const std::string &Mode) {
  queue Queue{property::queue::in_order{}};
  device Dev = Queue.get_device();
  context Ctx = Queue.get_context();

  std::ifstream FS(CommsFile, std::ios::in | std::ios::binary);
  if (!FS) {
    std::cerr << Mode << ": failed to open " << CommsFile << "\n";
    return 1;
  }
  size_t HandleSize = 0;
  FS.read(reinterpret_cast<char *>(&HandleSize), sizeof(size_t));
  if (HandleSize == 0) {
    std::cerr << Mode << ": zero-sized handle\n";
    return 2;
  }
  std::unique_ptr<std::byte[]> HandleBuffer{new std::byte[HandleSize]};
  FS.read(reinterpret_cast<char *>(HandleBuffer.get()), HandleSize);

  ipc::handle_data_t HandleVec(HandleBuffer.get(),
                               HandleBuffer.get() + HandleSize);
#ifdef USE_ZE_IPC
  int *Data = static_cast<int *>(zeIpcOpenHelper(HandleVec, Ctx, Dev));
#else
  int *Data = static_cast<int *>(ipc::memory::open(HandleVec, Ctx, Dev));
#endif
  int *Snapshot = malloc_host<int>(N, Queue);

  int Errors = 0;
  if (Mode == "consumer") {
    recordAndReplay(Queue, ConsumerReplays, [&] {
      Queue.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] += 1; });
      Queue.memcpy(Snapshot, Data, N * sizeof(int));
    });
    Errors = countMismatches(Snapshot, consumerValue, "ConsumerReplay");
  } else {
    recordAndReplay(Queue, 1,
                    [&] { Queue.memcpy(Snapshot, Data, N * sizeof(int)); });
    Errors = countMismatches(Snapshot, exporterValue, "ExporterWrite");
  }

  free(Snapshot, Queue);
#ifdef USE_ZE_IPC
  zeIpcCloseHelper(Data, Ctx);
#else
  ipc::memory::close(Data, Ctx);
#endif
  return Errors == 0 ? 0 : 3;
}

int main(int argc, char *argv[]) {
  return argc == 1 ? spawner(argv) : consumer(argv[1]);
}
