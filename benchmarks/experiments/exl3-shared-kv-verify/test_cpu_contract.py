"""CPU guard/packing checks; no device initialization or inference."""
import torch
import verify_attention as V


class DeviceFacade:
    def __init__(self, tensor, device='xpu:0'):
        self.tensor, self.device = tensor, torch.device(device)

    def __getattr__(self, name):
        return getattr(self.tensor, name)


def inputs(batch=1, rows=4, page=1600):
    base = torch.empty((2, page, 4, 2, 256), dtype=torch.float8_e4m3fn)
    f = DeviceFacade
    return dict(q=f(torch.empty((batch * rows, 24, 256), dtype=torch.float16)),
                k=f(base[:, :, :, 0]), v=f(base[:, :, :, 1]),
                cu_seqlens_q=f(torch.arange(batch + 1, dtype=torch.int32) * rows),
                seqused_k=f(torch.full((batch,), page + 1, dtype=torch.int32)),
                block_table=f(torch.zeros((batch, 2), dtype=torch.int32)),
                max_seqlen_q=rows, max_seqlen_k=page + 1,
                softmax_scale=0.0625, causal=True,
                k_descale=f(torch.tensor(0.75).expand(batch, 4)),
                v_descale=f(torch.tensor(1.25).expand(batch, 4)))


checks = 0
for batch in (1, 4):
    for rows in (2, 3, 4, 5):
        for page in (64, 1600, 1664):
            assert V.eligible(inputs(batch, rows, page)); checks += 1
        q = torch.arange(batch * rows * 24 * 256).reshape(batch * rows, 24, 256)
        packed = V.packed_queries(q, batch, rows)
        # Independent position/head indexing, rather than only inverse identity.
        for b in range(batch):
            for r in range(rows):
                for h in range(24):
                    index = (h // 6) * rows * 6 + r * 6 + h % 6
                    assert torch.equal(packed[b, index], q[b * rows + r, h])
        assert torch.equal(V.unpacked_output(packed, batch, rows), q)

bad = [
    {'max_seqlen_q': 1}, {'max_seqlen_q': 6}, {'causal': False},
    {'causal': torch.tensor(True)}, {'softmax_scale': 0.125},
    {'window_size': (63, 0)}, {'s_aux': object()}, {'q_descale': object()},
    {'dynamic_causal': object()}, {'mask_mod': object()}, {'aux_tensors': []},
    {'scheduler_metadata': object()}, {'fa_version': 4}, {'num_splits': 4},
    {'softcap': 1.0}, {'return_softmax_lse': True}, {'dropout_p': 0.1},
    {'softcap': torch.tensor(0.0)}, {'max_seqlen_k': 3201},
    {'k_descale': DeviceFacade(torch.ones(1, 4))},
    {'v': DeviceFacade(torch.empty((2, 1600, 4, 256), dtype=torch.float16))},
    {'seqused_k': DeviceFacade(torch.ones(1, dtype=torch.int32), 'xpu:1')},
    {'block_table': DeviceFacade(torch.ones(1, 2, dtype=torch.int64))},
    {'out': DeviceFacade(torch.empty((4, 24, 256), dtype=torch.bfloat16))},
    {'new_attention_feature': False},
]
marker = object()
for change in bad:
    d = {**inputs(), **change}
    assert not V.eligible(d), change
    calls = []
    def native(**kwargs):
        calls.append(kwargs); return marker
    assert V.dispatch(native, **d) is marker and calls[0] == d
    checks += 1
assert not V.eligible(inputs(page=65))
assert not V.eligible(inputs(batch=17))
print(f'PASS {checks + 2} supported/fallback cases and independent packed-head indexing')
