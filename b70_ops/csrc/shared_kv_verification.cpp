#define B70_VERIFY_MASK 1
#include "paged_decode.hpp"

#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

using namespace cute;

namespace {

using VerifyPolicy = decode_policy_qpacked_head<_16, _256, _64>;
using VerifyPolicyQ8 = decode_policy_qpacked_head<_8, _256, _64>;

void check_inputs(const at::Tensor& q, const at::Tensor& k,
                  const at::Tensor& v, const at::Tensor& block_table,
                  const at::Tensor& cu_seqlens_q,
                  const at::Tensor& seqused_k, const at::Tensor& k_scale,
                  const at::Tensor& v_scale, const at::Tensor& out,
                  const at::Tensor& temp_out, const at::Tensor& exp_sums,
                  const at::Tensor& max_logits, int64_t max_seqlen_k,
                  int64_t num_splits, int64_t q_tile) {
  TORCH_CHECK(q.is_xpu() && k.is_xpu() && v.is_xpu() && out.is_xpu(),
              "q, k, v and out must be XPU tensors");
  TORCH_CHECK(q.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
              "q and out must be FP16");
  TORCH_CHECK(k.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                  v.scalar_type() == k.scalar_type(),
              "k and v must be E4M3FN");
  TORCH_CHECK(q.dim() == 3 && q.size(0) == 1 && q.size(2) == 256 &&
                  q.size(1) >= 48 && q.size(1) <= 120 && q.size(1) % 24 == 0,
              "packed q must have shape [1,48|72|96|120,256]");
  TORCH_CHECK(k.dim() == 4 && k.size(1) == 1664 && k.size(2) == 4 &&
                  k.size(3) == 256 && v.sizes() == k.sizes() &&
                  v.strides() == k.strides(),
              "k/v must share production shape and strides");
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous() &&
                  out.sizes() == q.sizes(),
              "q/out must be contiguous and have equal shapes");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == 1 &&
                  block_table.scalar_type() == at::kInt &&
                  block_table.is_contiguous(),
              "block_table must be one contiguous int32 row");
  TORCH_CHECK(cu_seqlens_q.numel() == 2 &&
                  cu_seqlens_q.scalar_type() == at::kInt &&
                  seqused_k.numel() == 1 &&
                  seqused_k.scalar_type() == at::kInt,
              "invalid sequence metadata");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat &&
                  v_scale.scalar_type() == at::kFloat &&
                  k_scale.numel() == 1 && v_scale.numel() == 1,
              "k/v scales must be scalar FP32 tensors");
  TORCH_CHECK(num_splits >= 1 && num_splits <= 32,
              "num_splits must be in [1,32]");
  TORCH_CHECK(q_tile == 8 || q_tile == 16, "q_tile must be 8 or 16");
  TORCH_CHECK(temp_out.numel() == q.numel() * num_splits &&
                  temp_out.scalar_type() == at::kHalf,
              "invalid split output scratch");
  TORCH_CHECK(exp_sums.numel() == q.size(1) * num_splits &&
                  max_logits.numel() == exp_sums.numel() &&
                  exp_sums.scalar_type() == at::kFloat &&
                  max_logits.scalar_type() == at::kFloat,
              "invalid reduction scratch");
  TORCH_CHECK(max_seqlen_k > 0, "max_seqlen_k must be positive");
}

at::Tensor& shared_kv_verify_out(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& block_table, const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqused_k, const at::Tensor& k_scale,
    const at::Tensor& v_scale, at::Tensor& out, at::Tensor& temp_out,
    at::Tensor& exp_sums, at::Tensor& max_logits, int64_t max_seqlen_k,
    int64_t num_splits, int64_t q_tile) {
  check_inputs(q, k, v, block_table, cu_seqlens_q, seqused_k, k_scale,
               v_scale, out, temp_out, exp_sums, max_logits, max_seqlen_k,
               num_splits, q_tile);
  paged_decode_args_t args{};
  args.query = q.data_ptr();
  args.key = k.data_ptr();
  args.value = v.data_ptr();
  args.out = out.data_ptr();
  args.tem_out = num_splits == 1 ? out.data_ptr() : temp_out.data_ptr();
  args.exp_sums = exp_sums.data_ptr();
  args.max_logits = max_logits.data_ptr();
  args.block_table = block_table.data_ptr();
  args.cu_seqlens_q = cu_seqlens_q.data_ptr();
  args.cu_seqlens_k = seqused_k.data_ptr();
  args.max_queries = 1;
  args.max_keys = max_seqlen_k;
  args.total_seqlen_q = 1;
  args.total_seqlen_k = get_paged_kv_cache_effective_total_seqlen(k);
  args.k_scale = k_scale.data_ptr();
  args.v_scale = v_scale.data_ptr();
  args.sm_scale = 0.0625f;
  args.batch_size = 1;
  args.num_heads_q = q.size(1);
  args.num_heads_k = 4;
  args.head_size = 256;
  args.v_head_size = 256;
  args.max_blocks_per_seq = block_table.size(1);
  args.block_size = 1664;
  args.is_varlen = true;
  args.is_paged = true;
  args.is_causal = true;
  args.num_kv_splits = num_splits;
  args.q_stride_seq = q.stride(0);
  args.q_stride_heads = q.stride(1);
  args.k_stride_page = k.stride(0);
  args.k_stride_seq = k.stride(1);
  args.k_stride_heads = k.stride(2);
  args.v_stride_page = v.stride(0);
  args.v_stride_seq = v.stride(1);
  args.v_stride_heads = v.stride(2);
  args.page_stride_elements = get_paged_kv_cache_page_stride_elements(k);
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  if (q_tile == 8) {
    PagedDecodeConfig<typename VerifyPolicyQ8::ShapeQK,
                      typename VerifyPolicyQ8::ShapePV,
                      typename VerifyPolicyQ8::ShapeOut,
                      typename VerifyPolicyQ8::SubgroupLayoutQK, void, 1,
                      true, false, false, half_t, float_e4m3_t, float_e4m3_t,
                      half_t>::kernel_dispatch(queue, args);
  } else {
    PagedDecodeConfig<typename VerifyPolicy::ShapeQK,
                      typename VerifyPolicy::ShapePV,
                      typename VerifyPolicy::ShapeOut,
                      typename VerifyPolicy::SubgroupLayoutQK, void, 1, true,
                      false, false, half_t, float_e4m3_t, float_e4m3_t,
                      half_t>::kernel_dispatch(queue, args);
  }
  return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(b70_ops, m) {
  m.def("shared_kv_verify_out(Tensor q, Tensor k, Tensor v, Tensor block_table, Tensor cu_seqlens_q, Tensor seqused_k, Tensor k_scale, Tensor v_scale, Tensor(a!) out, Tensor(b!) temp_out, Tensor(c!) exp_sums, Tensor(d!) max_logits, int max_seqlen_k, int num_splits, int q_tile) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(b70_ops, XPU, m) {
  m.impl("shared_kv_verify_out", &shared_kv_verify_out);
}
