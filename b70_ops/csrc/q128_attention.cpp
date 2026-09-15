#include "chunk_prefill.hpp"
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>
using namespace cute;

namespace {
template <int Q, int SG> struct Policy {
  using ShapeQK = Shape<Int<Q>, _32, _32>;
  using ShapePV = Shape<Int<Q>, _32, _32>;
  using ShapeOut = Shape<Int<Q>, _256>;
  using SubgroupLayoutQK = Layout<Shape<Int<SG>, _1, _1>>;
};
template <int Q, int SG>
void launch(sycl::queue& queue, const chunk_prefill_args_t& args) {
  using P = Policy<Q, SG>;
  FMHAConfig<typename P::ShapeQK, typename P::ShapePV, typename P::ShapeOut,
             typename P::SubgroupLayoutQK, void, 2, true, true, false, false,
             false, half_t, float_e4m3_t, float_e4m3_t,
             half_t>::kernel_dispatch(queue, args);
}

void check_inputs(const at::Tensor& q, const at::Tensor& k,
                  const at::Tensor& v, const at::Tensor& block_table,
                  const at::Tensor& cu_seqlens_q,
                  const at::Tensor& seqused_k, const at::Tensor& k_scale,
                  const at::Tensor& v_scale, const at::Tensor& out,
                  int64_t max_seqlen_k, int64_t variant) {
  TORCH_CHECK(q.is_xpu() && k.is_xpu() && v.is_xpu() && out.is_xpu(),
              "q, k, v and out must be XPU tensors");
  TORCH_CHECK(block_table.is_xpu() && cu_seqlens_q.is_xpu() &&
                  seqused_k.is_xpu() && k_scale.is_xpu() && v_scale.is_xpu(),
              "metadata and scales must be XPU tensors");
  TORCH_CHECK(q.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
              "q and out must be FP16");
  TORCH_CHECK(k.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                  v.scalar_type() == k.scalar_type(), "k and v must be E4M3FN");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == 24 && q.size(2) == 256,
              "q must have shape [M,24,256]");
  TORCH_CHECK(k.dim() == 4 && k.size(1) == 1664 && k.size(2) == 4 &&
                  k.size(3) == 256 && v.sizes() == k.sizes() &&
                  v.strides() == k.strides(),
              "k/v must share shape [pages,1664,4,256] and strides");
  TORCH_CHECK(out.sizes() == q.sizes() && out.strides() == q.strides(),
              "out must have q's shape and strides");
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous(),
              "q and out must be contiguous");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == 1 &&
                  block_table.scalar_type() == at::kInt && block_table.is_contiguous(),
              "block_table must be contiguous int32 with batch size one");
  TORCH_CHECK(cu_seqlens_q.numel() == 2 && cu_seqlens_q.scalar_type() == at::kInt &&
                  cu_seqlens_q.is_contiguous(),
              "cu_seqlens_q must contain two contiguous int32 values");
  TORCH_CHECK(seqused_k.numel() == 1 && seqused_k.scalar_type() == at::kInt &&
                  seqused_k.is_contiguous(),
              "seqused_k must contain one contiguous int32 value");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat && v_scale.scalar_type() == at::kFloat &&
                  k_scale.numel() == 1 && v_scale.numel() == 1,
              "k/v scales must be scalar FP32 tensors");
  TORCH_CHECK(max_seqlen_k > 0, "max_seqlen_k must be positive");
  TORCH_CHECK(variant >= 0 && variant <= 2, "variant must be 0, 1 or 2");
  TORCH_CHECK(!out.is_alias_of(q) && !out.is_alias_of(k) && !out.is_alias_of(v),
              "out must not alias q, k or v");
}

at::Tensor& q128_forward_out(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& block_table, const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqused_k, const at::Tensor& k_scale,
    const at::Tensor& v_scale, at::Tensor& out, int64_t max_seqlen_k,
    int64_t variant) {
  check_inputs(q, k, v, block_table, cu_seqlens_q, seqused_k, k_scale,
               v_scale, out, max_seqlen_k, variant);
  chunk_prefill_args_t a{};
  a.query=q.data_ptr(); a.key=k.data_ptr(); a.value=v.data_ptr(); a.out=out.data_ptr();
  a.block_table=block_table.data_ptr(); a.cu_seqlens_q=cu_seqlens_q.data_ptr();
  a.cu_seqlens_k=seqused_k.data_ptr(); a.max_queries=q.size(0); a.max_keys=max_seqlen_k;
  a.total_seqlen_q=q.size(0); a.total_seqlen_k=get_paged_kv_cache_effective_total_seqlen(k);
  a.k_scale=k_scale.data_ptr(); a.v_scale=v_scale.data_ptr(); a.sm_scale=0.0625f;
  a.batch_size=1; a.num_heads_q=24; a.num_heads_k=4; a.head_size=256;
  a.max_blocks_per_seq=block_table.size(1); a.block_size=1664;
  a.is_varlen=true; a.is_paged=true; a.is_causal=true;
  a.q_stride_seq=q.stride(0); a.q_stride_heads=q.stride(1);
  a.o_stride_seq=out.stride(0); a.o_stride_heads=out.stride(1);
  a.k_stride_seq=k.stride(1); a.k_stride_heads=k.stride(2);
  a.v_stride_seq=v.stride(1); a.v_stride_heads=v.stride(2);
  a.page_stride_elements=get_paged_kv_cache_page_stride_elements(k);
  auto& queue=c10::xpu::getCurrentXPUStream().queue();
  if(variant==0) launch<256,32>(queue,a);
  else if(variant==1) launch<128,16>(queue,a);
  else launch<64,8>(queue,a);
  return out;
}

at::Tensor q128_forward(const at::Tensor& q, const at::Tensor& k,
                        const at::Tensor& v, const at::Tensor& block_table,
                        const at::Tensor& cu_seqlens_q, const at::Tensor& seqused_k,
                        const at::Tensor& k_scale, const at::Tensor& v_scale,
                        int64_t max_seqlen_k, int64_t variant) {
  auto out=at::empty_like(q);
  q128_forward_out(q,k,v,block_table,cu_seqlens_q,seqused_k,k_scale,v_scale,
                   out,max_seqlen_k,variant);
  return out;
}
}

TORCH_LIBRARY(b70_ops,m) {
  m.def("q128_forward(Tensor q, Tensor k, Tensor v, Tensor block_table, Tensor cu_seqlens_q, Tensor seqused_k, Tensor k_scale, Tensor v_scale, int max_seqlen_k, int variant) -> Tensor");
  m.def("q128_forward_out(Tensor q, Tensor k, Tensor v, Tensor block_table, Tensor cu_seqlens_q, Tensor seqused_k, Tensor k_scale, Tensor v_scale, Tensor(a!) out, int max_seqlen_k, int variant) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(b70_ops,XPU,m) {
  m.impl("q128_forward",&q128_forward);
  m.impl("q128_forward_out",&q128_forward_out);
}
