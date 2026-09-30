"""Keep GPTQ row dispatch outside vLLM's shape-specialized compiled graph."""
import torch


_quant_compiled = None


def _quant_reference(x: torch.Tensor):
    from vllm._xpu_ops import xpu_ops
    # Keep projection width static while row counts remain dynamic. This
    # preserves reduction fusion without freezing the row dispatch.
    torch._dynamo.mark_static(x, 1)
    return xpu_ops.dynamic_per_token_int8_quant_ref(x, True, 8)


def _dispatch(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
              zero: torch.Tensor, group_size: int, min_rows: int,
              bias: torch.Tensor | None) -> torch.Tensor:
    # This code executes with concrete tensor sizes, including during graph capture.
    # The compiler sees one opaque operation, so a large-M warmup cannot freeze
    # activation quantization into small-M decode graphs.
    if x.shape[0] < min_rows:
        return torch.ops._xpu_C.int4_gemm_w4a16(
            x, packed.t(), bias, scales, zero, group_size, None)
    global _quant_compiled
    if _quant_compiled is None:
        _quant_compiled = torch.compile(_quant_reference, fullgraph=True, dynamic=True)
    quant, factor, origin = _quant_compiled(x)
    return torch.ops._xpu_C.int4_gemm_w4a8(
        quant, factor, origin, packed.t(), scales, zero,
        group_size, None, bias).to(x.dtype)


@torch.library.custom_op('b70_linear::gptq_by_rows', mutates_args=(), device_types='xpu')
def gptq_by_rows(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                 zero: torch.Tensor, group_size: int, min_rows: int,
                 bias: torch.Tensor | None) -> torch.Tensor:
    return _dispatch(x, packed, scales, zero, group_size, min_rows, bias)


@gptq_by_rows.register_fake
def _fake(x, packed, scales, zero, group_size, min_rows, bias):
    return x.new_empty((x.shape[0], packed.shape[0]))
