import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from ram_kv_release import load_overlay, serving_arguments


class ProductionOverlay(unittest.TestCase):
    def setUp(self):
        self.release = json.loads((ROOT / 'config/production_image.json').read_text())

    def test_budget_and_effective_args(self):
        for budget in (10, 12):
            overlay = load_overlay(ROOT, self.release, {'B70_KV_RAM_GIB': str(budget)})
            self.assertEqual(overlay['transfer']['kv_connector_extra_config']['cpu_bytes_to_use'], budget * 1024**3)
            args = serving_arguments(['serve', 'model', '--watermark', '0.0'], overlay)
            self.assertEqual(args.count('--watermark'), 1)
            self.assertEqual(args[args.index('--watermark') + 1], '0.05')
            self.assertIn('kv_admission.CompletionReservedScheduler', args)
        for budget in ('0', '13', '1.5'):
            with self.assertRaises(ValueError):
                load_overlay(ROOT, self.release, {'B70_KV_RAM_GIB': budget})

    def test_modified_manifest_or_runtime_refuses_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / 'config', root / 'config')
            shutil.copytree(ROOT / 'runtime', root / 'runtime')
            module = root / 'runtime/kv_admission.py'
            module.write_text(module.read_text() + '\n# changed\n')
            with self.assertRaisesRegex(RuntimeError, 'module changed'):
                load_overlay(root, self.release, {})
            manifest = root / 'config/production_ram_kv.json'
            manifest.write_text(manifest.read_text() + '\n')
            with self.assertRaisesRegex(RuntimeError, 'manifest changed'):
                load_overlay(root, self.release, {})

    def test_wrong_base_image_and_baseline_rollback(self):
        release = copy.deepcopy(self.release)
        release['image_id'] = 'wrong'
        with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
            load_overlay(ROOT, release, {})
        release.pop('ram_kv_runtime')
        self.assertIsNone(load_overlay(ROOT, release, {}))
        self.assertEqual(serving_arguments(['serve'], None), ['serve'])
