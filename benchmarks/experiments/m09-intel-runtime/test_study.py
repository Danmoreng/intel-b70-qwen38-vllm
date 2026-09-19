import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('study', ROOT/'study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class StudyTests(unittest.TestCase):
    def test_gate_rejects_unqualified_and_changed_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run/'manifest.json').write_text('{"dtype":"bfloat16"}')
            (run/'preflight.json').write_text('{"ready":false}')
            with self.assertRaises(RuntimeError):
                study.require_gate(run)
            (run/'preflight.json').write_text(json.dumps({'ready': True,
                'manifest_sha256': study.prepare.digest(run/'manifest.json')}))
            study.require_gate(run)
            (run/'manifest.json').write_text('{"dtype":"float16"}')
            with self.assertRaises(RuntimeError):
                study.require_gate(run)

    def test_candidate_preserves_other_serving_arguments(self):
        prod = {'Config': {'Cmd': ['serve', 'model', '--dtype', 'float16',
            '--speculative-config', '{"method":"mtp","num_speculative_tokens":4}',
            '--max-model-len', '200704', '--enable-prefix-caching',
            '--mamba-cache-mode', 'align', '--max-num-batched-tokens', '4096']}}
        original = prod['Config']['Cmd'].copy()
        manifest = {'candidate': 'sha256:pinned'}
        def engine(*args, **kwargs):
            return ['docker', 'run', kwargs['image'], *kwargs['arguments']]
        with patch.object(study.diag, 'engine_command', side_effect=engine):
            cmd = study.command(prod, manifest, Path('/tmp/evidence'))
        self.assertEqual(prod['Config']['Cmd'], original)
        self.assertEqual(cmd[cmd.index('--max-model-len')+1], '200704')
        self.assertEqual(cmd[cmd.index('--mamba-cache-mode')+1], 'align')
        self.assertEqual(cmd[cmd.index('--dtype')+1], 'float16')
        self.assertEqual(cmd[cmd.index(manifest['candidate'])+1:], original)
        self.assertIn('VLLM_SERVER_DEV_MODE=1', cmd)

    def test_recovery_does_not_touch_production_without_ownership(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(study.subprocess, 'run') as run:
            study.recover(Path(temp))
            run.assert_not_called()

    def test_failed_restoration_retains_ownership_for_systemd_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/'production-owned').write_text('owned')
            with patch.object(study.diag, 'stop_container'), patch.object(
                    study.subprocess, 'run', side_effect=RuntimeError('restore failed')):
                with self.assertRaises(RuntimeError):
                    study.recover(path)
            self.assertTrue((path/'production-owned').exists())


if __name__ == '__main__':
    unittest.main()
