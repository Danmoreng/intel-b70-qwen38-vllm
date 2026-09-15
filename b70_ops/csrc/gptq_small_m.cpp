#include <ATen/DeviceGuard.h>
#include <ATen/Functions.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include <sycl/sycl.hpp>

namespace {

template <int M>
void launch_small_m(sycl::queue& queue, const sycl::half* x,
                    const int32_t* packed_weight, const sycl::half* scales,
                    sycl::half* out, int64_t k, int64_t n,
                    int64_t group_size) {
  constexpr int local_size = 256;
  const int64_t global_size = ((n + local_size - 1) / local_size) * local_size;
  queue.parallel_for(
      sycl::nd_range<1>(global_size, local_size),
      [=](sycl::nd_item<1> item) {
        const int64_t column = item.get_global_linear_id();
        if (column >= n) return;
        float accumulators[M] = {};
        const int64_t packed_k = k / 8;
        for (int64_t packed_index = 0; packed_index < packed_k;
             ++packed_index) {
          const uint32_t word = static_cast<uint32_t>(
              packed_weight[column * packed_k + packed_index]);
#pragma unroll
          for (int nibble = 0; nibble < 8; ++nibble) {
            const int64_t inner = packed_index * 8 + nibble;
            const int quantized = static_cast<int>((word >> (nibble * 4)) & 15) - 8;
            const float weight = static_cast<float>(quantized) *
                                 static_cast<float>(scales[(inner / group_size) * n + column]);
#pragma unroll
            for (int row = 0; row < M; ++row) {
              accumulators[row] += static_cast<float>(x[row * k + inner]) * weight;
            }
          }
        }
#pragma unroll
        for (int row = 0; row < M; ++row) {
          out[row * n + column] = static_cast<sycl::half>(accumulators[row]);
        }
      });
}

at::Tensor gptq_small_m(const at::Tensor& x, const at::Tensor& packed_weight,
                        const at::Tensor& scales, const at::Tensor& zero_point,
                        int64_t group_size) {
  const at::DeviceGuard guard(x.device());
  TORCH_CHECK(x.is_xpu() && packed_weight.is_xpu() && scales.is_xpu() &&
                  zero_point.is_xpu(),
              "all tensors must be on XPU");
  TORCH_CHECK(x.scalar_type() == at::kHalf && scales.scalar_type() == at::kHalf,
              "x and scales must be FP16");
  TORCH_CHECK(packed_weight.scalar_type() == at::kInt,
              "packed_weight must be int32");
  TORCH_CHECK(zero_point.scalar_type() == at::kChar && zero_point.numel() == 1,
              "only a scalar int8 symmetric zero point is supported");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && x.size(0) >= 1 &&
                  x.size(0) <= 5,
              "x must be contiguous [M,K] with 1 <= M <= 5");
  const int64_t m = x.size(0), k = x.size(1), n = packed_weight.size(1);
  TORCH_CHECK(group_size == 128 && k % 128 == 0,
              "only group_size=128 is supported");
  TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == k / 8 &&
                  packed_weight.stride(0) == 1 &&
                  packed_weight.stride(1) == k / 8,
              "packed_weight must be the GPTQ [K/8,N] transposed view");
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == k / group_size &&
                  scales.size(1) == n && scales.is_contiguous(),
              "scales must be contiguous [K/group_size,N]");
  auto out = at::empty({m, n}, x.options());
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  const auto* x_ptr = reinterpret_cast<const sycl::half*>(x.data_ptr<at::Half>());
  const auto* w_ptr = packed_weight.data_ptr<int32_t>();
  const auto* s_ptr = reinterpret_cast<const sycl::half*>(scales.data_ptr<at::Half>());
  auto* out_ptr = reinterpret_cast<sycl::half*>(out.data_ptr<at::Half>());
  if (m == 1) launch_small_m<1>(queue,x_ptr,w_ptr,s_ptr,out_ptr,k,n,group_size);
  else if (m == 2) launch_small_m<2>(queue,x_ptr,w_ptr,s_ptr,out_ptr,k,n,group_size);
  else if (m == 3) launch_small_m<3>(queue,x_ptr,w_ptr,s_ptr,out_ptr,k,n,group_size);
  else if (m == 4) launch_small_m<4>(queue,x_ptr,w_ptr,s_ptr,out_ptr,k,n,group_size);
  else launch_small_m<5>(queue,x_ptr,w_ptr,s_ptr,out_ptr,k,n,group_size);
  return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(b70_ops, m) {
  m.def("gptq_small_m(Tensor x, Tensor packed_weight, Tensor scales, "
        "Tensor zero_point, int group_size) -> Tensor");
}
TORCH_LIBRARY_IMPL(b70_ops, XPU, m) {
  m.impl("gptq_small_m", &gptq_small_m);
}
