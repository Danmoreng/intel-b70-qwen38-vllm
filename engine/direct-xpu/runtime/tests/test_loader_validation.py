"""CPU integration with real Torch/vLLM: fused transforms, completeness, head width and TP guards."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import tempfile
import json
import gc
import weakref
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exl3xpu.vllm_plugin import Exl3Config,Exl3LinearMethod,_build_draft_head
from exl3xpu.loader_audit import audit_runner

class LoaderTests(unittest.TestCase):
    def layer(self,prefix='layers.0.self_attn.qkv_proj',bits=4,sizes=(128,128,128)):
        config=Exl3Config(4,6,'mul1')
        method=Exl3LinearMethod(config,bits,prefix)
        layer=torch.nn.Module()
        with patch('exl3xpu.vllm_plugin.get_tensor_model_parallel_world_size',return_value=1):
            method.create_weights(layer,128,list(sizes),128,sum(sizes),torch.float16)
        return method,layer
    def fill(self,method,layer,groups=None):
        groups=groups or [(i,) for i in range(len(layer.exl3_sizes))]
        for ordinal,g in enumerate(groups):
            sid=g if len(g)>1 else g[0];width=sum(layer.exl3_sizes[i] for i in g)
            layer.trellis.weight_loader(layer.trellis,torch.zeros(8,width//16,16*method.K,dtype=torch.int16),sid)
            layer.suh.weight_loader(layer.suh,torch.full((128,),ordinal+1,dtype=torch.float16),sid)
            layer.svh.weight_loader(layer.svh,torch.ones(width,dtype=torch.float16),sid)
            layer.mul1.weight_loader(layer.mul1,torch.tensor(1,dtype=torch.int32),sid)
    def test_qkv_keeps_independent_hadamard_shards(self):
        m,l=self.layer();self.fill(m,l);m.process_weights_after_loading(l)
        self.assertEqual(l.exl3_bounds,[0,128,256,384])
        self.assertTrue(torch.equal(l.suh[:,0],torch.tensor([1,2,3],dtype=torch.float16)))
        self.assertEqual(l.exl3_shard_of_nb.tolist(),[0,1,2])
    def test_gdn_combined_qkv_and_separate_z(self):
        m,l=self.layer(prefix='layers.0.linear_attn.in_proj_qkvz',sizes=(128,128,256,256))
        self.fill(m,l,[(0,1,2),(3,)]);m.process_weights_after_loading(l)
        self.assertEqual(l.exl3_bounds,[0,512,768]);self.assertEqual(l.exl3_shard_of_nb.tolist(),[0,0,0,0,1,1])
        self.assertTrue(torch.equal(l.suh[:,0],torch.tensor([1,2],dtype=torch.float16)))
    def test_head_stays_six_bits(self):
        m,l=self.layer(prefix='lm_head',bits=6,sizes=(128,));self.fill(m,l);m.process_weights_after_loading(l)
        self.assertEqual(l.exl3_K,6);self.assertEqual(l.trellis.shape[-1],96)
    def test_mtp_fc_is_four_bits(self):
        m,l=self.layer(prefix='mtp.fc',sizes=(128,));self.fill(m,l);m.process_weights_after_loading(l)
        self.assertEqual(l.exl3_loader_report['members'],['mtp.fc'])
    def test_draft_head_preserves_target_six_bit_trellis(self):
        m,l=self.layer(prefix='lm_head',bits=6,sizes=(256,));self.fill(m,l);m.process_weights_after_loading(l)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'draft.json';p.write_text(json.dumps({'blocks':[1]}));_build_draft_head(l,p)
            self.assertEqual(l.exl3_K,6)
            self.assertEqual(l.exl3_draft['trellis'].shape[-1],96)
            self.assertTrue(torch.equal(l.exl3_draft['trellis'],l.trellis[:,8:16]))
            self.assertEqual(l.exl3_draft['idx'].tolist(),list(range(128,256)))
            for blocks in ([],[-1],[2],[1,1]):
                p.write_text(json.dumps({'blocks':blocks}))
                with self.assertRaises(ValueError): _build_draft_head(l,p)
    def test_missing_trellis_or_transform_rejected(self):
        for component in ('trellis','suh','svh'):
            m,l=self.layer();self.fill(m,l);l.exl3_loaded_shards[component].remove(1)
            with self.assertRaisesRegex(ValueError,'incomplete'): m.process_weights_after_loading(l)
    def test_invalid_shape_and_duplicate_rejected(self):
        m,l=self.layer()
        with self.assertRaisesRegex(ValueError,'shape/bits'): l.trellis.weight_loader(l.trellis,torch.zeros(8,8,96,dtype=torch.int16),0)
        self.fill(m,l)
        with self.assertRaisesRegex(ValueError,'duplicate'): l.suh.weight_loader(l.suh,torch.ones(128),0)
    def test_invalid_fused_group_and_tp_rejected(self):
        m,l=self.layer()
        with self.assertRaises(ValueError): l.suh.weight_loader(l.suh,torch.ones(128),[0,2])
        with patch('exl3xpu.vllm_plugin.get_tensor_model_parallel_world_size',return_value=2):
            with self.assertRaises(NotImplementedError): m.create_weights(torch.nn.Module(),128,[128],128,128,torch.float16)
    def test_whole_model_missing_mtp_rejected(self):
        cfg=Exl3Config(4,6,'mul1');cfg.expected_inventory=[{'canonical':'lm_head','bits':6},{'canonical':'mtp.fc','bits':4}]
        m,l=self.layer(prefix='lm_head',bits=6,sizes=(128,));self.fill(m,l);m.process_weights_after_loading(l)
        target=torch.nn.Module();target.add_module('lm_head',l)
        runner=SimpleNamespace(vllm_config=SimpleNamespace(quant_config=cfg),model=target,drafter=SimpleNamespace(model=torch.nn.Module()))
        with self.assertRaisesRegex(ValueError,'mtp.fc'): audit_runner(runner)
    def test_discarded_head_not_retained_by_loader_callbacks(self):
        m,l=self.layer(prefix='lm_head',bits=6,sizes=(256,));self.fill(m,l);m.process_weights_after_loading(l)
        # Parameter/callback handles can outlive an MTP head replaced by the shared
        # target head. They must not retain its entire quantized/pruned storage.
        param=l.trellis
        callbacks=[getattr(l,name).weight_loader for name in ('trellis','svh','mul1')]
        reference=weakref.ref(l)
        del l
        gc.collect()
        self.assertIsNone(reference())
        with self.assertRaisesRegex(RuntimeError,'owner was released'):
            callbacks[0](param,torch.zeros_like(param))

if __name__=='__main__': unittest.main()
