"""CPU coverage gate: loader and GPU oracle see exactly the same modules."""
import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exl3xpu.checkpoint import inventory,normalize_key,validate_exl3_environment,GPTQ_DRAFT_FLAGS
from exl3xpu.validation import int8_reference_pass,INT8_REFERENCE_TOLERANCE

def fixture():
    names=('model.language_model.layers.0.mlp.gate_proj','lm_head','mtp.fc')
    index={k+s:'weights.safetensors' for k in names for s in ('.trellis','.suh','.svh','.mul1')}
    qc={'tensor_storage':{names[0]:{'quant_format':'exl3','bits_per_weight':4},
                          'lm_head':{'quant_format':'exl3','bits_per_weight':6}}}
    return index,qc

class InventoryTests(unittest.TestCase):
    def test_missing_qc_mtp_is_discovered(self):
        idx,qc=fixture();inv=inventory(idx,qc)
        self.assertEqual(len(inv),3)
        self.assertEqual({v['canonical']:v['bits'] for v in inv},
                         {'layers.0.mlp.gate_proj':4,'lm_head':6,'mtp.fc':4})
        self.assertEqual(sum(v['mtp'] for v in inv),1)
    def test_missing_component_rejected(self):
        idx,qc=fixture();del idx['mtp.fc.suh']
        with self.assertRaisesRegex(ValueError,'Incomplete'): inventory(idx,qc)
    def test_missing_trellis_rejected(self):
        idx,qc=fixture();del idx['lm_head.trellis']
        with self.assertRaisesRegex(ValueError,'missing from weight index'): inventory(idx,qc)
    def test_duplicate_canonical_rejected(self):
        idx,qc=fixture();idx['model.mtp.fc.trellis']='weights.safetensors'
        with self.assertRaises(ValueError): inventory(idx,qc)
    def test_conflicting_codebook_rejected(self):
        idx,qc=fixture();idx['mtp.fc.mcg']='weights.safetensors'
        with self.assertRaisesRegex(ValueError,'Codebook mismatch'): inventory(idx,qc)
    def test_other_quant_format_rejected(self):
        idx,qc=fixture();qc['tensor_storage']['lm_head']['quant_format']='gptq'
        with self.assertRaisesRegex(ValueError,'format conflict'): inventory(idx,qc)
    def test_empty_inventory_rejected(self):
        with self.assertRaisesRegex(ValueError,'No EXL3'): inventory({}, {})
    def test_normalization(self):
        self.assertEqual(normalize_key('language_model.model.layers.3.self_attn.q_proj'),'layers.3.self_attn.q_proj')
        self.assertEqual(normalize_key('model.mtp.layers.0.self_attn.q_proj'),'mtp.layers.0.self_attn.q_proj')
    def test_exl3_rejects_gptq_environment(self):
        validate_exl3_environment({})
        for flag in GPTQ_DRAFT_FLAGS:
            with self.assertRaises(ValueError): validate_exl3_environment({flag:'1'})
    def test_int8_gate_preserves_threshold_and_rejects_nonfinite(self):
        self.assertEqual(INT8_REFERENCE_TOLERANCE,2e-3)
        self.assertTrue(int8_reference_pass([0.0,1.999e-3]))
        for value in (2e-3,3e-3,float('nan'),float('inf')):
            self.assertFalse(int8_reference_pass([value]))
        self.assertFalse(int8_reference_pass([]))

if __name__=='__main__': unittest.main()
