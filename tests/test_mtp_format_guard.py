"""CPU gate for format selection and incompatible startup environments."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
PATCHES=ROOT/'docker/patches'
sys.path.insert(0,str(PATCHES))
from b70_mtp_policy import resolve_mtp_policy, FLAGS
from patch_mtp_format_guard import guarded_source

class Quant:
    def __init__(self,name): self.name=name
    def get_name(self): return self.name

class PolicyTests(unittest.TestCase):
    def test_production_gptq(self):
        p=resolve_mtp_policy(Quant('auto_gptq'),environment={f:'1' for f in FLAGS})
        self.assertTrue(p.is_gptq and p.bf16_draft and p.lmhead_int4 and p.mtp_int4)
    def test_exl3_native(self):
        p=resolve_mtp_policy(Quant('exl3'),environment={})
        self.assertFalse(p.is_gptq or p.bf16_draft or p.lmhead_int4 or p.mtp_int4)
    def test_exl3_rejects_each_override(self):
        for flag in FLAGS:
            with self.subTest(flag=flag),self.assertRaisesRegex(ValueError,'incompatible'):
                resolve_mtp_policy(Quant('exl3'),environment={flag:'1'})
    def test_unquantized_modelopt_and_other_formats_reject(self):
        for q in (None,Quant('modelopt_fp4'),Quant('awq')):
            with self.assertRaises(ValueError): resolve_mtp_policy(q,environment={FLAGS[1]:'1'})
    def test_mtp_int4_requires_unquantized_source(self):
        with self.assertRaisesRegex(ValueError,'source MTP'):
            resolve_mtp_policy(Quant('gptq'),environment={FLAGS[2]:'1'})
        hf=SimpleNamespace(quantization_config={'dynamic':{'-:mtp.*':{}}})
        self.assertTrue(resolve_mtp_policy(Quant('gptq'),hf,{FLAGS[2]:'1'}).mtp_int4)
    def test_invalid_flag(self):
        with self.assertRaisesRegex(ValueError,'must be 0 or 1'):
            resolve_mtp_policy(Quant('exl3'),environment={FLAGS[0]:'true'})
    def test_unknown_patch_source_rejected(self):
        with self.assertRaises(ValueError): guarded_source('import torch\n')

if __name__=='__main__': unittest.main()
