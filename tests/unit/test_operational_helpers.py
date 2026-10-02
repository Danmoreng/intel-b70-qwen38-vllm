"""Protect immutable aliases and fail preflight before starting a service."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
from release_integrity import load_release, SERVICE

spec = importlib.util.spec_from_file_location('restore_production', REPO / 'scripts/restore-production.py')
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


class BuildSafety(unittest.TestCase):
    def run_build(self, tag=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / 'repo'
            (repo / 'scripts').mkdir(parents=True)
            shutil.copytree(REPO / 'config', repo / 'config')
            for name in ['build-image.sh', 'release_integrity.py']:
                shutil.copyfile(REPO / 'scripts' / name, repo / 'scripts' / name)
            (repo / '.env').write_text('VLLM_IMAGE=local/b70-qwen38-vllm:production-exl3-v1\n')
            (repo / 'docker/wheel').mkdir(parents=True)
            (repo / 'docker/wheel/vllm_xpu_kernels-0.1.15.4-cp38-abi3-manylinux_2_28_x86_64.whl').touch()
            tools = root / 'bin'
            tools.mkdir()
            log = root / 'calls'
            for name, body in {'docker': 'printf "%s\\n" "$*" >> "$MOCK_CALL_LOG"\n',
                               'curl': 'echo NETWORK >> "$MOCK_CALL_LOG"; exit 77\n',
                               'sha256sum': 'exit 0\n'}.items():
                path = tools / name
                path.write_text('#!/bin/sh\n' + body)
                path.chmod(0o755)
            env = {**os.environ, 'PATH': str(tools) + os.pathsep + os.environ['PATH'],
                   'MOCK_CALL_LOG': str(log), 'VLLM_IMAGE': 'local/b70-qwen38-vllm:production-exl3-v1'}
            env.pop('B70_BASE_BUILD_IMAGE', None)
            if tag is not None:
                env['B70_BASE_BUILD_IMAGE'] = tag
            run = subprocess.run(['bash', str(repo / 'scripts/build-image.sh')],
                                 capture_output=True, text=True, env=env, timeout=10)
            return run, log.read_text() if log.exists() else ''

    def test_protected_tags_have_no_download_or_docker_side_effects(self):
        for tag in ['local/b70-qwen38-vllm:production-exl3-v1',
                    'local/b70-qwen38-vllm:production-onednn-v2',
                    'another/image:production-future', 'sha256:' + '0' * 64]:
            with self.subTest(tag=tag):
                run, calls = self.run_build(tag)
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(calls, '')

    def test_default_and_candidate_tags_do_not_follow_serving_env(self):
        for tag in [None, 'local/b70-qwen38-vllm:base-test', 'local/b70-qwen38-vllm:candidate-test']:
            with self.subTest(tag=tag):
                run, calls = self.run_build(tag)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn('-t ' + (tag or 'local/b70-qwen38-vllm:base-vllm-0.30.0-xpu-kernels-0.1.15.4'), calls)
                self.assertNotIn('production-', calls)

    def test_current_release_receipts_validate(self):
        self.assertEqual(load_release()['status'], 'QUALIFIED_RELEASE')


class RestoreSafety(unittest.TestCase):
    def test_preflight_failure_starts_nothing(self):
        with patch.object(restore, 'preflight', side_effect=RuntimeError('bad release')), \
                patch.object(restore.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'bad release'):
                restore.main()
            run.assert_not_called()

    def test_different_service_checkout_is_rejected_before_launcher(self):
        with patch.object(restore, 'load_release', return_value={}), \
                patch.object(restore.subprocess, 'check_output', return_value='/some/other/checkout\n'), \
                patch.object(restore.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'WorkingDirectory'):
                restore.preflight(REPO)
            run.assert_not_called()

    def test_current_launcher_preflight_failure_cannot_start_service(self):
        start = '{ path='+str(REPO/'scripts/run-server.sh')+' ; argv[]='+str(REPO/'scripts/run-server.sh')+' ; ignore_errors=no ; }'
        with patch.object(restore, 'load_release', return_value={}), \
                patch.object(restore.subprocess, 'check_output', side_effect=[str(REPO),start]), \
                patch.object(restore.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'preflight')) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                restore.preflight(REPO)
            self.assertEqual(run.call_count, 1)
            self.assertIn(str(REPO / 'scripts/run-server-exl3.py'), run.call_args.args[0])
            self.assertNotIn(SERVICE, run.call_args.args[0])

    def test_same_checkout_wrong_service_launcher_is_rejected_before_start(self):
        start = '{ path=/old/gptq-launcher ; argv[]=/old/gptq-launcher ; ignore_errors=no ; }'
        with patch.object(restore,'load_release',return_value={}), \
                patch.object(restore.subprocess,'check_output',side_effect=[str(REPO),start]), \
                patch.object(restore.subprocess,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'ExecStart'):
                restore.preflight(REPO)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
