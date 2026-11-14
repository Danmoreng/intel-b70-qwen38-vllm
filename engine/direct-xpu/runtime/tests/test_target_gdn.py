"""Compare actual 0.33 build semantics and device row selection against native."""
import dataclasses
import inspect
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exl3xpu.gdn_rows import HostRows
from exl3xpu.gdn_target_patch import transformed_build, EXPECTED_SITES
from vllm.v1.attention.backends import gdn_attn

ORIGINAL = gdn_attn.GDNAttentionMetadataBuilder.build
SOURCE, SITES = transformed_build(textwrap.dedent(inspect.getsource(ORIGINAL)))
NS = dict(vars(gdn_attn), _Exl3HostRows=HostRows)
exec(compile(SOURCE, '<test GDN semantic port>', 'exec'), NS)
PORTED = NS['build']


class TargetGDNTests(unittest.TestCase):
    def test_accounted_device_sites_and_unknown_source(self):
        self.assertEqual(SITES, EXPECTED_SITES)
        source = textwrap.dedent(inspect.getsource(ORIGINAL))
        with self.assertRaises(ValueError):
            transformed_build(source.replace('query_lens[non_spec_sequence_masks_cpu]', 'query_lens[:1]'))

    def test_row_selection_c1_c4_and_reused_indices(self):
        for device in ['cpu'] + (['xpu'] if torch.xpu.is_available() else []):
            for mask in ([True], [False], [True]*4, [False]*4,
                         [True, True, False, False], [False, True, False, True]):
                m = torch.tensor(mask, dtype=torch.bool)
                t = torch.arange(len(mask)*5, device=device).reshape(len(mask), 5)
                rows = HostRows(m)
                expected = t[m.to(device)]
                self.assertTrue(torch.equal(rows.select(t), expected))
                prepared = rows.indices
                self.assertTrue(torch.equal(rows.select(t+10), expected+10))
                self.assertIs(rows.indices, prepared)
                if device == 'xpu' and rows.prefix_length is None:
                    pointer = rows.device_indices[str(t.device)].data_ptr()
                    rows.select(t)
                    self.assertEqual(pointer, rows.device_indices[str(t.device)].data_ptr())
        with self.assertRaises(ValueError): HostRows(torch.ones(3)).select(torch.ones(3))
        with self.assertRaises(ValueError): HostRows(torch.ones(3, dtype=torch.bool)).select(torch.ones(4))

    @unittest.skipUnless(torch.xpu.is_available(), 'XPU graph replay requires the device')
    def test_row_selection_graph_replay(self):
        for mask in ([True], [True]*4, [True, True, False, False], [False, True, False, True]):
            m = torch.tensor(mask, dtype=torch.bool)
            t = torch.arange(len(mask)*5, device='xpu').reshape(len(mask), 5)
            rows = HostRows(m)
            rows.select(t)  # prepare CPU indices and the one pinned transfer outside capture
            torch.xpu.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph): out = rows.select(t).clone()
            t.add_(7)
            graph.replay()
            self.assertTrue(torch.equal(out, t[m.to('xpu')]))

    def builder(self, device, graphs):
        b = gdn_attn.GDNAttentionMetadataBuilder.__new__(gdn_attn.GDNAttentionMetadataBuilder)
        b.use_spec_decode = True; b.num_spec = 3
        b.use_full_cuda_graph = graphs; b.decode_cudagraph_max_bs = 16
        b.kv_cache_spec = None
        b.gdn_prefill_backend = 'sycl'
        b.checkpoint_builder = SimpleNamespace(build=lambda *args: None)
        b.plan_flashinfer_checkpoints = False
        b.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode='align'))
        b._build_chunk_metadata = lambda *args: (None, None)
        for name, shape, dtype in (
            ('spec_state_indices_tensor', (16, 4), torch.int32),
            ('non_spec_state_indices_tensor', (16,), torch.int32),
            ('spec_sequence_masks', (16,), torch.bool),
            ('spec_token_indx', (64,), torch.int32), ('non_spec_token_indx', (64,), torch.int32),
            ('spec_query_start_loc', (17,), torch.int32), ('non_spec_query_start_loc', (17,), torch.int32),
            ('num_accepted_tokens', (16,), torch.int32)):
            setattr(b, name, torch.empty(shape, device=device, dtype=dtype))
        return b

    def test_actual_builder_fields_all_spec_non_spec_mixed_partial_c1_c4(self):
        # Only block selection / chunk-convolution helper primitives are stubbed.
        # Execute the real reviewed build body, including mixed classification,
        # six remaining row sites, cumulative query lengths, and graph metadata padding.
        for device in ['cpu'] + (['xpu'] if torch.xpu.is_available() else []):
            for graphs in (False, True):
                for lens, drafts in (([4], [3]), ([1], [0]), ([1]*4, [0]*4), ([1], [-1]), ([4]*4, [3]*4),
                                     ([1, 1, 8, 4], [-1]*4),
                                     ([4, 17, 4, 11], [3, -1, 3, -1]),
                                     ([4, 4, 0, 0], [3, 3, -1, -1]), ([4, 2], [3, 1])):
                    n = len(lens)
                    cpu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
                    m = SimpleNamespace(query_start_loc=cpu.to(device), query_start_loc_cpu=cpu,
                                        block_table_tensor=torch.arange(n*4, dtype=torch.int32, device=device).reshape(n,4),
                                        seq_lens=torch.tensor(lens, dtype=torch.int32, device=device)+100,
                                        seq_lens_cpu_upper_bound=torch.tensor(lens,dtype=torch.int32)+100,
                                        is_prefilling=torch.tensor([x>1 for x in lens],dtype=torch.bool),
                                        num_reqs=n, num_actual_tokens=sum(lens),
                                        compute_num_computed_tokens=lambda: torch.full((n,),100,device=device))
                    m.replace=lambda **updates: SimpleNamespace(**(vars(m)|updates))
                    accepted = torch.arange(1, n+1, dtype=torch.int32, device=device)
                    def split(*args, **kwargs):
                        nd = sum(x == 1 for x in lens)
                        return nd, sum(x > 1 for x in lens), nd, sum(x for x in lens if x > 1)
                    with patch('vllm.utils.torch_utils.PIN_MEMORY', device != 'cpu'), patch.object(gdn_attn, 'mamba_get_block_table_tensor', side_effect=lambda t,*args:t), \
                         patch.object(gdn_attn, 'split_decodes_and_prefills', side_effect=split), \
                         patch.object(gdn_attn, 'compute_causal_conv1d_metadata', return_value=(None,None,None)):
                        # The exec namespace contains direct references; install equivalent helpers there.
                        old = {k:NS[k] for k in ('mamba_get_block_table_tensor','split_decodes_and_prefills','compute_causal_conv1d_metadata')}
                        NS.update(mamba_get_block_table_tensor=gdn_attn.mamba_get_block_table_tensor,
                                  split_decodes_and_prefills=gdn_attn.split_decodes_and_prefills,
                                  compute_causal_conv1d_metadata=gdn_attn.compute_causal_conv1d_metadata)
                        try:
                            ref = ORIGINAL(self.builder(device,graphs),0,m,accepted,torch.tensor(drafts))
                            got = PORTED(self.builder(device,graphs),0,m,accepted,torch.tensor(drafts))
                        finally: NS.update(old)
                    for field in dataclasses.fields(ref):
                        a, b = getattr(ref,field.name), getattr(got,field.name)
                        if isinstance(a,torch.Tensor): self.assertTrue(torch.equal(a,b), (device,graphs,lens,field.name,a,b))
                        else: self.assertEqual(a,b,(device,graphs,lens,field.name))


if __name__ == '__main__': unittest.main()
