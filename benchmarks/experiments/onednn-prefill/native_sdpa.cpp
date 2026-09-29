// Isolated fused-SDPA feasibility operator for the pinned vLLM 0.30 XPU stack.
// Input: q [6,Q,256], k/v [1,L,256], output [6,Q,256], all contiguous FP16.
// The graph uses bottom-right causal masking and oneDNN Graph fusion.
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <vector>
#include <torch/all.h>
#include <torch/library.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>

namespace {
namespace dg = dnnl::graph;
using lt = dg::logical_tensor;

uint16_t fp8e4m3_to_half_bits(uint8_t x) {
    uint16_t sign = static_cast<uint16_t>(x & 0x80) << 8;
    uint16_t exponent = (x >> 3) & 0x0f;
    uint16_t fraction = x & 0x07;
    if (exponent != 0) {
        if (exponent == 15 && fraction == 7) return sign | 0x7e00; // NaN
        return sign | static_cast<uint16_t>((exponent + 8) << 10) |
               static_cast<uint16_t>(fraction << 7);
    }
    if (fraction == 0) return sign;
    uint16_t lead = fraction >= 4 ? 2 : (fraction >= 2 ? 1 : 0);
    return sign | static_cast<uint16_t>((lead + 6) << 10) |
           static_cast<uint16_t>((fraction - (1 << lead)) << (10 - lead));
}

void gather_dequant(at::Tensor cache, at::Tensor pages, at::Tensor scale,
                    at::Tensor out, int64_t head) {
    TORCH_CHECK(cache.is_xpu() && pages.is_xpu() && scale.is_xpu() && out.is_xpu());
    TORCH_CHECK(cache.device() == pages.device() && cache.device() == scale.device() &&
                cache.device() == out.device(), "gather tensors must share an XPU device");
    TORCH_CHECK(cache.scalar_type() == at::ScalarType::Float8_e4m3fn);
    TORCH_CHECK(cache.dim() == 4 && cache.size(1) == 1664 && cache.size(2) == 4 &&
                cache.size(3) == 256 && cache.stride(3) == 1);
    TORCH_CHECK(pages.scalar_type() == at::kInt && pages.is_contiguous() && pages.dim() == 1);
    TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.numel() == 1);
    TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous() &&
                out.dim() == 2 && out.size(1) == 256);
    TORCH_CHECK(head >= 0 && head < 4 && pages.numel() >= (out.size(0) + 1663) / 1664);
    auto& queue = c10::xpu::getCurrentXPUStream(cache.device().index()).queue();
    const auto* src = reinterpret_cast<const uint8_t*>(cache.data_ptr());
    const auto* page_ids = pages.data_ptr<int32_t>();
    const auto* factor = scale.data_ptr<float>();
    auto* dst = reinterpret_cast<sycl::half*>(out.data_ptr());
    int64_t n = out.numel();
    int64_t stride0 = cache.stride(0), stride1 = cache.stride(1), stride2 = cache.stride(2);
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::nd_range<1>(((n + 255) / 256) * 256, 256),
                             [=](sycl::nd_item<1> item) {
            int64_t i = item.get_global_linear_id();
            if (i >= n) return;
            int64_t token = i / 256;
            int64_t dimension = i % 256;
            int64_t logical_page = token / 1664;
            int64_t physical_page = page_ids[logical_page];
            int64_t offset = physical_page * stride0 + (token % 1664) * stride1 +
                             head * stride2 + dimension;
            sycl::half value = sycl::bit_cast<sycl::half>(fp8e4m3_to_half_bits(src[offset]));
            dst[i] = static_cast<sycl::half>(static_cast<float>(value) * factor[0]);
        });
    });
}

struct Entry {
    dg::compiled_partition compiled;
    std::vector<lt> inputs;
    std::vector<lt> outputs;
    int32_t length_k;
    int32_t length_q;
    uint64_t last_used = 0;
    std::optional<sycl::event> last_event;
};

void* graph_alloc(size_t bytes, size_t, const void*, const void*) {
    return c10::xpu::XPUCachingAllocator::raw_alloc(bytes);
}

void graph_free(void* pointer, const void*, const void*, void* event) {
    if (event) static_cast<sycl::event*>(event)->wait();
    c10::xpu::XPUCachingAllocator::raw_delete(pointer);
}

struct EngineState {
    sycl::device device;
    sycl::context context;
    dnnl::engine engine;
    std::map<std::pair<int64_t, int64_t>, Entry> cache;
    uint64_t clock = 0;
    uint64_t last_used = 0;
};

constexpr size_t kMaxCompiledShapes = 32;
constexpr size_t kMaxContexts = 4;
std::mutex cache_mutex;
std::list<EngineState> engines;
uint64_t context_clock = 0;

Entry build(dnnl::engine& engine, int64_t Q, int64_t L) {
    using dt = lt::data_type;
    int64_t id = 0;
    lt query(id++, dt::f16, {1, 1, 6, Q, 256}, {6*Q*256, 6*Q*256, Q*256, 256, 1});
    lt key(id++, dt::f16, {1, 1, 1, L, 256}, {L*256, L*256, L*256, 256, 1});
    lt score(id++, dt::f32, {1, 1, 6, Q, L}, lt::layout_type::strided);
    dg::op bmm1(id++, dg::op::kind::MatMul, "qk");
    bmm1.set_attr<bool>(dg::op::attr::transpose_b, true);
    bmm1.add_inputs({query, key}); bmm1.add_outputs({score});

    lt divisor(id++, dt::f16, lt::dims{1}, lt::layout_type::strided);
    lt scaled(id++, dt::f32, {1, 1, 6, Q, L}, lt::layout_type::strided);
    dg::op divide(id++, dg::op::kind::Divide, "scale");
    divide.add_inputs({score, divisor}); divide.add_outputs({scaled});

    lt row(id++, dt::s32, {1, 1, 6, Q, L}, lt::layout_type::strided);
    dg::op row_index(id++, dg::op::kind::GenIndex, "row");
    row_index.set_attr<int64_t>(dg::op::attr::axis, -2);
    row_index.add_inputs({scaled}); row_index.add_outputs({row});
    lt len_k(id++, dt::s32, 0, lt::layout_type::strided, lt::property_type::host_scalar);
    lt row_plus_l(id++, dt::s32, {1, 1, 6, Q, L}, lt::layout_type::strided);
    dg::op add(id++, dg::op::kind::Add, "row_plus_l");
    add.add_inputs({row, len_k}); add.add_outputs({row_plus_l});
    lt len_q(id++, dt::s32, 0, lt::layout_type::strided, lt::property_type::host_scalar);
    lt allowed(id++, dt::s32, {1, 1, 6, Q, L}, lt::layout_type::strided);
    dg::op subtract(id++, dg::op::kind::Subtract, "minus_q");
    subtract.add_inputs({row_plus_l, len_q}); subtract.add_outputs({allowed});
    lt col(id++, dt::s32, {1, 1, 6, Q, L}, lt::layout_type::strided);
    dg::op col_index(id++, dg::op::kind::GenIndex, "column");
    col_index.set_attr<int64_t>(dg::op::attr::axis, -1);
    col_index.add_inputs({scaled}); col_index.add_outputs({col});
    lt visible(id++, dt::boolean, {1, 1, 6, Q, L}, lt::layout_type::strided);
    dg::op compare(id++, dg::op::kind::GreaterEqual, "visible");
    compare.add_inputs({allowed, col}); compare.add_outputs({visible});
    lt negative_inf(id++, dt::f32, lt::dims{1}, lt::layout_type::strided);
    lt masked(id++, dt::f32, {1, 1, 6, Q, L}, lt::layout_type::strided);
    dg::op select(id++, dg::op::kind::Select, "causal_mask");
    select.add_inputs({visible, scaled, negative_inf}); select.add_outputs({masked});

    lt probability(id++, dt::f16, {1, 1, 6, Q, L}, lt::layout_type::strided);
    dg::op softmax(id++, dg::op::kind::SoftMax, "softmax");
    softmax.set_attr<int64_t>(dg::op::attr::axis, -1);
    softmax.set_attr<std::string>(dg::op::attr::mode, "inf_as_zero");
    softmax.add_inputs({masked}); softmax.add_outputs({probability});
    lt value(id++, dt::f16, {1, 1, 1, L, 256}, {L*256, L*256, L*256, 256, 1});
    lt output(id++, dt::f16, {1, 1, 6, Q, 256}, {6*Q*256, 6*Q*256, Q*256, 256, 1});
    dg::op bmm2(id++, dg::op::kind::MatMul, "pv");
    bmm2.add_inputs({probability, value}); bmm2.add_outputs({output});

    dg::graph graph(dnnl::engine::kind::gpu);
    for (auto& op : std::vector<dg::op>{bmm1, divide, row_index, add, subtract,
                                        col_index, compare, select, softmax, bmm2}) {
        graph.add_op(op);
    }
    graph.finalize();
    auto partitions = graph.get_partitions();
    TORCH_CHECK(partitions.size() == 1, "SDPA graph did not fuse into one partition");
    std::vector<lt> inputs{query, key, divisor, len_k, len_q, negative_inf, value};
    std::vector<lt> outputs{output};
    auto compiled = partitions[0].compile(inputs, outputs, engine);
    TORCH_CHECK(compiled.get_scratchpad_logical_tensor().get_mem_size() == 0,
                "oneDNN graph unexpectedly requires scratchpad");
    return {compiled, inputs, outputs, static_cast<int32_t>(L), static_cast<int32_t>(Q)};
}

void sdpa(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out,
          at::Tensor divisor, at::Tensor negative_inf) {
    TORCH_CHECK(q.is_xpu() && k.is_xpu() && v.is_xpu() && out.is_xpu());
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                q.device() == out.device() && q.device() == divisor.device() &&
                q.device() == negative_inf.device(),
                "SDPA tensors must share an XPU device");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
                v.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf);
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && out.is_contiguous());
    TORCH_CHECK(q.dim() == 3 && q.size(0) == 6 && q.size(2) == 256 &&
                k.dim() == 3 && k.size(0) == 1 && k.size(2) == 256 &&
                v.sizes() == k.sizes() && out.sizes() == q.sizes());
    TORCH_CHECK(divisor.is_xpu() && divisor.scalar_type() == at::kHalf && divisor.numel() == 1);
    TORCH_CHECK(negative_inf.is_xpu() && negative_inf.scalar_type() == at::kFloat && negative_inf.numel() == 1);
    auto& queue = c10::xpu::getCurrentXPUStream(q.device().index()).queue();
    TORCH_CHECK(queue.has_property<sycl::property::queue::in_order>(),
                "oneDNN prefill requires an in-order caller queue");
    int64_t Q = q.size(1), L = k.size(1);
    TORCH_CHECK(Q > 0 && Q <= L && L <= INT32_MAX);
    std::lock_guard<std::mutex> guard(cache_mutex);
    auto state_it = std::find_if(engines.begin(), engines.end(), [&](const EngineState& state) {
        return state.device == queue.get_device() && state.context == queue.get_context();
    });
    if (state_it == engines.end()) {
        if (engines.size() >= kMaxContexts) {
            auto oldest = std::min_element(engines.begin(), engines.end(),
                [](const EngineState& a, const EngineState& b) {
                    return a.last_used < b.last_used;
                });
            for (auto& item : oldest->cache) {
                if (item.second.last_event) item.second.last_event->wait();
            }
            engines.erase(oldest);
        }
        auto allocator = dg::sycl_interop::make_allocator(graph_alloc, graph_free);
        auto engine = dg::sycl_interop::make_engine_with_allocator(
            queue.get_device(), queue.get_context(), allocator);
        engines.push_back({queue.get_device(), queue.get_context(), engine});
        state_it = std::prev(engines.end());
    }
    auto& state = *state_it;
    state.last_used = ++context_clock;
    auto key = std::make_pair(Q, L);
    if (!state.cache.count(key)) {
        const char* diagnostics = std::getenv("B70_ONEDNN_DIAGNOSTICS");
        bool log_cache = diagnostics && diagnostics[0] == '1';
        if (state.cache.size() >= kMaxCompiledShapes) {
            auto oldest = std::min_element(state.cache.begin(), state.cache.end(),
                [](const auto& a, const auto& b) {
                    return a.second.last_used < b.second.last_used;
                });
            if (oldest->second.last_event) oldest->second.last_event->wait();
            if (log_cache) std::cerr << "B70_ONEDNN_CACHE_EVICT Q="
                                     << oldest->first.first << " L="
                                     << oldest->first.second << std::endl;
            state.cache.erase(oldest);
        }
        state.cache.emplace(key, build(state.engine, Q, L));
        if (log_cache) std::cerr << "B70_ONEDNN_CACHE_MISS Q=" << Q
                                 << " L=" << L << " entries="
                                 << state.cache.size() << std::endl;
    }
    auto& entry = state.cache.at(key);
    entry.last_used = ++state.clock;
    auto stream = dnnl::sycl_interop::make_stream(state.engine, queue);
    using tensor = dg::tensor;
    std::vector<tensor> inputs{
        tensor(entry.inputs[0], state.engine, q.data_ptr()),
        tensor(entry.inputs[1], state.engine, k.data_ptr()),
        tensor(entry.inputs[2], state.engine, divisor.data_ptr()),
        tensor::make_scalar_tensor(entry.inputs[3], &entry.length_k),
        tensor::make_scalar_tensor(entry.inputs[4], &entry.length_q),
        tensor(entry.inputs[5], state.engine, negative_inf.data_ptr()),
        tensor(entry.inputs[6], state.engine, v.data_ptr())};
    std::vector<tensor> outputs{tensor(entry.outputs[0], state.engine, out.data_ptr())};
    std::vector<sycl::event> deps;
    if (entry.last_event) deps.push_back(*entry.last_event);
    entry.last_event = dg::sycl_interop::execute(entry.compiled, stream, inputs, outputs, deps);
}
} // namespace

TORCH_LIBRARY(b70_sdpa_probe, module) {
    module.def("forward(Tensor q, Tensor k, Tensor v, Tensor(a!) out, Tensor divisor, Tensor negative_inf) -> ()");
    module.impl("forward", torch::kXPU, &sdpa);
    module.def("gather_dequant(Tensor cache, Tensor pages, Tensor scale, Tensor(a!) out, int head) -> ()");
    module.impl("gather_dequant", torch::kXPU, &gather_dequant);
}
