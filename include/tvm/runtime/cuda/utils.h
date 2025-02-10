struct CUstream_st;

namespace tvm {
namespace runtime {
namespace cuda {

void BusyWait(const double duration, CUstream_st* stream = nullptr);

}  // namespace cuda
}  // namespace runtime
}  // namespace tvm
