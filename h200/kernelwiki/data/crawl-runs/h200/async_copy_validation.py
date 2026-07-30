import time
import torch
from torch.utils.cpp_extension import load_inline

src = r'''
#include <torch/extension.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>

constexpr int TILE = 16;
__global__ void async_copy_gemm(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c, int n) {
  __shared__ alignas(alignof(float4)) float as[TILE][TILE];
  __shared__ alignas(alignof(float4)) float bs[TILE][TILE];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
  float sum = 0.f;
  auto pipe = cuda::make_pipeline();
  for (int k = 0; k < n; k += TILE) {
    if (tx < TILE / 4) {
      pipe.producer_acquire();
      const auto bytes = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
      cuda::memcpy_async(&as[ty][tx * 4], &a[row * n + k + tx * 4], bytes, pipe);
      cuda::memcpy_async(&bs[ty][tx * 4], &b[(k + ty) * n + blockIdx.x * TILE + tx * 4], bytes, pipe);
      pipe.producer_commit();
      pipe.consumer_wait();
    }
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < TILE; ++kk) sum += as[ty][kk] * bs[kk][tx];
    pipe.consumer_release();
    __syncthreads();
  }
  c[row * n + col] = sum;
}

torch::Tensor run(torch::Tensor a, torch::Tensor b) {
  auto c = torch::empty_like(a);
  const int n = a.size(0);
  dim3 block(TILE, TILE), grid(n / TILE, n / TILE);
  async_copy_gemm<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "async_copy_gemm launch failed");
  return c;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
'''

mod = load_inline(name="kw_async_copy_gemm", cpp_sources="", cuda_sources=src,
                  functions=None, extra_cuda_cflags=["-O3", "-std=c++17"], verbose=False)
torch.manual_seed(0)
for n in (128, 512, 1024):
  a = torch.randn((n, n), device="cuda", dtype=torch.float32)
  b = torch.randn((n, n), device="cuda", dtype=torch.float32)
  got = mod.run(a, b)
  ref = a @ b
  err = (got - ref).abs().max().item()
  assert torch.allclose(got, ref, rtol=2e-4, atol=2e-4), (n, err)
  for _ in range(5): mod.run(a, b)
  torch.cuda.synchronize()
  begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  begin.record()
  for _ in range(20): mod.run(a, b)
  end.record(); end.synchronize()
  print("N={} max_abs_err={:.6g} latency_ms={:.4f}".format(n, err, begin.elapsed_time(end) / 20.0))
print("RESULT=PASS")
