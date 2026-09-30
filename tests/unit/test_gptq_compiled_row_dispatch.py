"""A large tracing input must not select INT8 for every later decode shape."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
    from vllm_xpu_kernels import _xpu_C
except ImportError as error:
    raise unittest.SkipTest('Requires the pinned B70 image') from error

sys.path.insert(0, str(Path(__file__).parents[2] / 'docker/w4a8-current'))
import b70_gptq_row_dispatch as dispatch


class CompiledDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # CPU implementations make the selected native ABI visible without a GPU.
        cls.library = torch.library.Library('_xpu_C', 'IMPL', 'CPU')
        cls.calls = []

        def w4a16(x, weight, bias, scales, zero, group, retained):
            cls.calls.append(('w4a16', x.shape[0]))
            return x.new_full((x.shape[0], weight.shape[1]), 16)

        def w4a8(x, factor, origin, weight, scales, zero, group, retained, bias):
            cls.calls.append(('w4a8', x.shape[0]))
            return torch.full((x.shape[0], weight.shape[1]), 8, dtype=torch.float16)

        cls.library.impl('int4_gemm_w4a16', w4a16)
        cls.library.impl('int4_gemm_w4a8', w4a8)
        dispatch.gptq_by_rows.register_kernel('cpu', dispatch._dispatch)

    def test_large_first_trace_then_small_and_boundary_shapes(self):
        def quant(x, signed, bits):
            return x.to(torch.int8), x.new_ones((x.shape[0], 1)), torch.zeros((x.shape[0], 1), dtype=torch.int32)

        ops_module = ModuleType('vllm._xpu_ops')
        ops_module.xpu_ops = SimpleNamespace(dynamic_per_token_int8_quant_ref=quant)
        modules = {'vllm._xpu_ops': ops_module}
        packed = torch.ones((256, 16), dtype=torch.int32)
        scales = torch.ones((1, 256), dtype=torch.float16)
        zero = torch.tensor([8], dtype=torch.int8)
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        def forward(x):
            return dispatch.gptq_by_rows(x, packed, scales, zero, 128, 512, None)

        compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=True)
        with patch.dict(sys.modules, modules):
            for rows in (6656, 1, 4, 5, 20, 511, 512, 513, 20):
                result = compiled(torch.ones((rows, 128), dtype=torch.float16))
                expected = 16 if rows < 512 else 8
                self.assertTrue(torch.equal(result, torch.full((rows, 256), expected, dtype=torch.float16)))
                self.assertEqual(self.calls[-1], ('w4a16' if rows < 512 else 'w4a8', rows))
        for graph in graphs:
            calls = [str(node.target) for node in graph.graph.nodes if node.op == 'call_function']
            self.assertTrue(any('b70_linear.gptq_by_rows' in call for call in calls))
            self.assertFalse(any('int4_gemm_w4a8' in call or 'int4_gemm_w4a16' in call for call in calls))


if __name__ == '__main__':
    unittest.main()
