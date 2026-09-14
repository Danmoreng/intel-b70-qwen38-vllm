#include "chunk_prefill.hpp"
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>
using namespace cute;

template<int Q, int SG> struct Policy {
 using ShapeQK=Shape<Int<Q>,_32,_32>;
 using ShapePV=Shape<Int<Q>,_32,_32>;
 using ShapeOut=Shape<Int<Q>,_256>;
 using SubgroupLayoutQK=Layout<Shape<Int<SG>,_1,_1>>;
};
template<int Q,int SG> void launch(sycl::queue& queue,const chunk_prefill_args_t& a) {
 using P=Policy<Q,SG>;
 FMHAConfig<typename P::ShapeQK,typename P::ShapePV,typename P::ShapeOut,typename P::SubgroupLayoutQK,void,2,true,true,false,false,false,half_t,float_e4m3_t,float_e4m3_t,half_t>::kernel_dispatch(queue,a);
}
at::Tensor tiles(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,const at::Tensor& bt,const at::Tensor& cq,const at::Tensor& used,const at::Tensor& ks,const at::Tensor& vs,int64_t maxk,int64_t variant) {
 TORCH_CHECK(q.is_xpu() && q.scalar_type()==at::kHalf && k.scalar_type()==at::ScalarType::Float8_e4m3fn && v.scalar_type()==k.scalar_type());
 TORCH_CHECK(q.dim()==3 && q.size(1)==24 && q.size(2)==256 && k.dim()==4 && k.size(2)==4 && k.size(3)==256);
 TORCH_CHECK(bt.size(0)==1 && cq.numel()==2 && used.numel()==1 && k.size(1)==1664 && q.is_contiguous());
 TORCH_CHECK(ks.numel()==1 && vs.numel()==1 && variant>=0 && variant<=2);
 auto o=at::empty_like(q);
 chunk_prefill_args_t a{};
 a.query=q.data_ptr();a.key=k.data_ptr();a.value=v.data_ptr();a.out=o.data_ptr();
 a.block_table=bt.data_ptr();a.cu_seqlens_q=cq.data_ptr();a.cu_seqlens_k=used.data_ptr();
 a.max_queries=q.size(0);a.max_keys=maxk;a.total_seqlen_q=q.size(0);
 a.total_seqlen_k=get_paged_kv_cache_effective_total_seqlen(k);
 a.k_scale=ks.data_ptr();a.v_scale=vs.data_ptr();a.sm_scale=0.0625f;
 a.batch_size=1;a.num_heads_q=24;a.num_heads_k=4;a.head_size=256;
 a.max_blocks_per_seq=bt.size(1);a.block_size=1664;a.is_varlen=true;a.is_paged=true;a.is_causal=true;
 a.q_stride_seq=q.stride(0);a.q_stride_heads=q.stride(1);
 a.o_stride_seq=o.stride(0);a.o_stride_heads=o.stride(1);
 a.k_stride_seq=k.stride(1);a.k_stride_heads=k.stride(2);
 a.v_stride_seq=v.stride(1);a.v_stride_heads=v.stride(2);
 a.page_stride_elements=get_paged_kv_cache_page_stride_elements(k);
 auto& queue=c10::xpu::getCurrentXPUStream().queue();
 if(variant==0)launch<256,32>(queue,a);
 else if(variant==1)launch<128,16>(queue,a);
 else launch<64,8>(queue,a);
 return o;
}
TORCH_LIBRARY(b70_tiles,m){m.def("forward(Tensor q, Tensor k, Tensor v, Tensor bt, Tensor cq, Tensor used, Tensor ks, Tensor vs, int maxk, int variant) -> Tensor");}
TORCH_LIBRARY_IMPL(b70_tiles,XPU,m){m.impl("forward",&tiles);}
