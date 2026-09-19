#include <ATen/DeviceGuard.h>
#include <ATen/Functions.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>
#include <sycl/sycl.hpp>

// TP1, symmetric GPTQ G128 QKVZ plus unchanged FP16 B/A weights.
// One subgroup reduces one output column; packed K loads are coalesced.
template<int M>
void launch(sycl::queue& q, const sycl::half* x, const int32_t* w,
            const sycl::half* s, const sycl::half* ba,
            sycl::half* y, sycl::half* z) {
  constexpr int K=5120, N=16384, B=96, SG=16;
  q.parallel_for(sycl::nd_range<1>((N+B)*SG,128),
    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
      const int col=item.get_global_linear_id()/SG;
      const int lane=item.get_sub_group().get_local_linear_id();
      float sum[M]={};
      if(col<N) {
        for(int pk=lane;pk<K/8;pk+=SG) {
          const uint32_t bits=static_cast<uint32_t>(w[col*(K/8)+pk]);
          const float scale=static_cast<float>(s[(pk/16)*N+col]);
          #pragma unroll
          for(int nib=0;nib<8;++nib) {
            const int k=pk*8+nib;
            const float v=static_cast<float>(static_cast<sycl::half>((int((bits>>(4*nib))&15)-8)*scale));
            #pragma unroll
            for(int m=0;m<M;++m) sum[m]+=static_cast<float>(x[m*K+k])*v;
          }
        }
      } else {
        for(int k=lane;k<K;k+=SG) {
          const float v=static_cast<float>(ba[(col-N)*K+k]);
          #pragma unroll
          for(int m=0;m<M;++m) sum[m]+=static_cast<float>(x[m*K+k])*v;
        }
      }
      #pragma unroll
      for(int m=0;m<M;++m) {
        const float value=sycl::reduce_over_group(item.get_sub_group(),sum[m],sycl::plus<float>());
        if(lane==0) {
          if(col<N) y[m*N+col]=static_cast<sycl::half>(value);
          else z[m*B+col-N]=static_cast<sycl::half>(value);
        }
      }
    });
}

std::tuple<at::Tensor,at::Tensor> project(const at::Tensor& x,const at::Tensor& w,
                                       const at::Tensor& s,const at::Tensor& ba) {
  const at::DeviceGuard guard(x.device());
  for(const auto& t:{x,w,s,ba}) TORCH_CHECK(t.is_xpu() && t.device()==x.device() && t.is_contiguous(),"XPU contiguous tensors on one device required");
  TORCH_CHECK(x.scalar_type()==at::kHalf && s.scalar_type()==at::kHalf && ba.scalar_type()==at::kHalf && w.scalar_type()==at::kInt,"FP16 activations/scales/BA and INT32 packed QKVZ required");
  TORCH_CHECK(x.dim()==2 && x.size(1)==5120 && x.size(0)>=1 && x.size(0)<=5,"only M1..5 K5120");
  TORCH_CHECK(w.sizes()==at::IntArrayRef({16384,640}) && s.sizes()==at::IntArrayRef({40,16384}) && ba.sizes()==at::IntArrayRef({96,5120}),"unsupported projection layout");
  auto y=at::empty({x.size(0),16384},x.options());auto z=at::empty({x.size(0),96},x.options());
  auto& queue=c10::xpu::getCurrentXPUStream().queue();
  auto xp=reinterpret_cast<const sycl::half*>(x.data_ptr<at::Half>());
  auto sp=reinterpret_cast<const sycl::half*>(s.data_ptr<at::Half>());
  auto bp=reinterpret_cast<const sycl::half*>(ba.data_ptr<at::Half>());
  auto yp=reinterpret_cast<sycl::half*>(y.data_ptr<at::Half>());
  auto zp=reinterpret_cast<sycl::half*>(z.data_ptr<at::Half>());
  #define DISPATCH(M) case M: launch<M>(queue,xp,w.data_ptr<int32_t>(),sp,bp,yp,zp);break
  switch(x.size(0)){DISPATCH(1);DISPATCH(2);DISPATCH(3);DISPATCH(4);DISPATCH(5);}
  return {y,z};
}
TORCH_LIBRARY(b70_gdn,m){m.def("project(Tensor x, Tensor w, Tensor s, Tensor ba) -> (Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(b70_gdn,XPU,m){m.impl("project",&project);}
