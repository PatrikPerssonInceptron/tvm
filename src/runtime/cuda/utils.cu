#include <cuda_runtime.h>

namespace {

int64_t SecondsToCycles(const double seconds) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return int64_t(seconds * double(prop.clockRate) * 1000.0);
}

__global__ void BusyWaitKernel(const int64_t num_cycles) {
  // Based on pytorch/aten/src/ATen/cuda/Sleep.cu::spin_kernel
  const auto start = clock64();
  for (int64_t clock_delta = 0; clock_delta < num_cycles; clock_delta = clock64() - start);
}

}  // namespace

namespace tvm {
namespace runtime {
namespace cuda {

void BusyWait(const double duration, CUstream_st* stream) {
  const auto num_cycles = SecondsToCycles(duration);
  BusyWaitKernel<<<1, 1, 0, stream>>>(num_cycles);
}

}  // namespace cuda
}  // namespace runtime
}  // namespace tvm
