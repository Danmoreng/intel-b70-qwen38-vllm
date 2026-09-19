"""Run inside the pinned vLLM image; exercise its real loader on tiny CPU weights."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
    from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
    from vllm.model_executor.models import b70_draft_lmhead_int4 as quant
except ImportError as error:
    raise unittest.SkipTest("Requires the pinned B70 vLLM image") from error


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "target_patch", ROOT / "benchmarks/experiments/m06-target-head-int4/candidate/patch_target_head.py")
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)
SOURCE = PATCH.path.read_text()
if "_b70_finalize_target_head" not in SOURCE:
    SOURCE = PATCH.patch_source(SOURCE)
DRAFT_SOURCE = PATCH.path.with_name("b70_draft_lmhead_int4.py").read_text()
if "B70_DRAFT_LMHEAD_INT4_REUSED_TARGET" not in DRAFT_SOURCE:
    DRAFT_SOURCE = PATCH.patch_draft_source(DRAFT_SOURCE)
draft_node = next(n for n in ast.parse(DRAFT_SOURCE).body
                  if isinstance(n, ast.FunctionDef) and n.name == "build_draft_lmhead_int4")
exec(compile(ast.fix_missing_locations(ast.Module(body=[draft_node], type_ignores=[])),
             "patched-draft-builder", "exec"), quant.__dict__)


def methods(class_name):
    node = next(n for n in ast.parse(SOURCE).body
                if isinstance(n, ast.ClassDef) and n.name == class_name)
    selected = [n for n in node.body if isinstance(n, ast.FunctionDef)
                and (n.name.startswith("_b70_") or n.name in
                     ("load_weights", "compute_logits", "compute_logits_local"))]
    scope = dict(torch=torch, os=os, AutoWeightsLoader=AutoWeightsLoader)
    tree = ast.Module(body=[ast.ImportFrom(module="__future__",
                      names=[ast.alias(name="annotations")], level=0), *selected],
                      type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "actual-patched-methods", "exec"), scope)
    return {n.name: scope[n.name] for n in selected}


class Language(torch.nn.Module):
    hf_to_vllm_mapper = WeightsMapper()

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(128, 4, bias=False)
        self.lm_head = torch.nn.Linear(128, 8, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=False, vocab_size=8, hidden_size=128)
        self.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1))
        self.logits_processor = lambda head, hidden, **kw: head(hidden)


for name, method in methods("Qwen3_5ForCausalLMBase").items():
    setattr(Language, name, method)


class Outer(torch.nn.Module):
    hf_to_vllm_mapper = WeightsMapper()

    def __init__(self):
        super().__init__()
        self.language_model = Language()
        self.visual = torch.nn.Linear(128, 4, bias=False)


for name, method in methods("Qwen3_5ForConditionalGeneration").items():
    setattr(Outer, name, method)


class LoadingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = Outer()
        self.head = torch.randn(8, 128).half()
        # Repeated language_model groups mirror the streamed VL checkpoint.
        self.weights = [
            ("language_model.model.weight", torch.randn(4, 128)),
            ("visual.weight", torch.randn(4, 128)),
            ("language_model.lm_head.weight", self.head),
        ]
        self.model.half()
        self.enterContext(patch.dict(os.environ, B70_TARGET_LMHEAD_INT4="1",
                                     B70_TARGET_LMHEAD_FREE_FP16="1"))
        self.enterContext(patch.object(torch.xpu, "synchronize"))

    def test_interleaved_groups_pack_final_loaded_head_once(self):
        with patch.object(quant, "quantize_lmhead_to_int4", wraps=quant.quantize_lmhead_to_int4) as pack:
            loaded = self.model.load_weights(iter(self.weights))
        self.assertEqual(loaded, {name for name, _ in self.weights})
        self.assertEqual(pack.call_count, 1)
        torch.testing.assert_close(pack.call_args.args[0], self.head)
        expected = quant.quantize_lmhead_to_int4(self.head)
        for actual, reference in zip(self.model.language_model._b70_target_lmhead_int4[:3], expected[:3]):
            torch.testing.assert_close(actual, reference)
        self.assertEqual(self.model.language_model.lm_head.weight.numel(), 0)
        with self.assertRaisesRegex(RuntimeError, "weight reload"):
            self.model.load_weights(iter(self.weights))

    def test_disabled_leaves_loaded_fp16_head(self):
        with patch.dict(os.environ, B70_TARGET_LMHEAD_INT4="0"):
            self.model.load_weights(iter(self.weights))
        torch.testing.assert_close(self.model.language_model.lm_head.weight, self.head)
        self.assertFalse(hasattr(self.model.language_model, "_b70_target_lmhead_int4"))

    def test_missing_head_cannot_quantize_uninitialized_weights(self):
        with self.assertRaisesRegex(RuntimeError, "loaded output head"):
            self.model.load_weights(iter(self.weights[:-1]))

    def test_tied_head_refused(self):
        self.model.language_model.config.tie_word_embeddings = True
        with self.assertRaisesRegex(RuntimeError, "independent output head"):
            self.model.load_weights(iter(self.weights))

    def test_both_logits_routes_use_packed_weights(self):
        self.model.load_weights(iter(self.weights))
        hidden = torch.randn(2, 128).half()
        sentinel = torch.randn(2, 8)
        with patch.object(quant, "int4_lmhead_logits", return_value=sentinel) as project:
            self.assertIs(self.model.language_model.compute_logits(hidden), sentinel)
            self.assertIs(self.model.compute_logits_local(hidden), sentinel)
            self.assertEqual(project.call_count, 2)

    def test_v2_shared_draft_adopts_pack_after_fp16_release(self):
        self.model.load_weights(iter(self.weights))
        draft = Language()
        # The pinned V2 load_eagle_model replaces draft.lm_head with target.lm_head.
        draft.lm_head = self.model.language_model.lm_head
        self.assertEqual(draft.lm_head.weight.numel(), 0)
        with patch.dict(os.environ, B70_DRAFT_LMHEAD_INT4="1"), patch.object(
                quant, "quantize_lmhead_to_int4", side_effect=AssertionError("must not repack empty source")):
            quant.build_draft_lmhead_int4(draft)
        self.assertIs(draft._b70_lmhead_int4, self.model.language_model._b70_target_lmhead_int4)


if __name__ == "__main__":
    unittest.main()
