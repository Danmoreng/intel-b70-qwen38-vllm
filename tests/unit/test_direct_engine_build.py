"""Protect production tags and reject unreviewed direct-build inputs."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import hashlib

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
spec = importlib.util.spec_from_file_location('build_engine', REPO / 'scripts/build-engine.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def archive_bytes(name='source/file.txt'):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w:gz') as archive:
        info = tarfile.TarInfo(name)
        info.size = 5
        archive.addfile(info, io.BytesIO(b'hello'))
    return out.getvalue()


class DirectBuild(unittest.TestCase):
    def test_release_tag_may_have_an_explicitly_different_library_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / 'CMakeLists.txt').write_text('set(PROJECT_VERSION "3.14.1")\n')
            build.check_onednn_source_version(source, {'version': '3.14.2', 'runtime_version': '3.14.1'})

    def test_wrong_embedded_version_is_rejected_before_compilation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / 'CMakeLists.txt').write_text('set(PROJECT_VERSION "3.14.1")\n')
            with self.assertRaisesRegex(RuntimeError, 'oneDNN source version differs'):
                build.check_onednn_source_version(source, {'version': '3.14.2', 'runtime_version': '3.14.2'})

    def test_failure_cleanup_only_uses_this_build_log_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'build.log'
            log.write_text('noise b70-qwen38-vllm\n ---> Running in abcdef123456\n'
                           ' ---> Running in 111111111111\n')
            with patch.object(build.subprocess, 'run', side_effect=[
                    SimpleNamespace(returncode=0), SimpleNamespace(returncode=0),
                    SimpleNamespace(returncode=1)]) as docker:
                self.assertEqual(build.cleanup_build_containers(log), ['abcdef123456'])
                self.assertEqual(docker.call_args_list[1].args[0], ['docker', 'rm', '-f', 'abcdef123456'])
                self.assertEqual(docker.call_count, 3)

    def test_production_tag_is_rejected_before_docker_or_download(self):
        tag = json.loads((REPO / 'config/production_image.json').read_text())['image_tag']
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(sys, 'argv', ['build-engine.py', '--tag', tag, '--output', tmp + '/out']), \
                patch.object(build.subprocess, 'run') as docker, \
                patch.object(build.urllib.request, 'urlopen') as network:
            with self.assertRaises(ValueError):
                build.main()
            docker.assert_not_called()
            network.assert_not_called()
            self.assertFalse((Path(tmp) / 'out').exists())

    def prepare(self, archive, expected_digest=None, tamper=None, artifact_digest=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'direct'
            shutil.copytree(build.SOURCE, source, ignore=shutil.ignore_patterns('__pycache__'))
            lock_file = source / 'build-lock.json'
            lock = json.loads(lock_file.read_text())
            lock['artifacts'] = []
            if artifact_digest:
                lock['artifacts'] = [{'filename': 'test.whl', 'url': 'https://example.invalid/test.whl',
                                      'sha256': artifact_digest}]
            lock['sources'] = {'test': {'url': 'https://example.invalid/source.tar.gz',
                'sha256': expected_digest or hashlib.sha256(archive).hexdigest()}}
            lock_file.write_text(json.dumps(lock))
            if tamper:
                path = source / tamper
                path.write_text(path.read_text() + '\n# unreviewed\n' if path.exists() else '# unreviewed\n')
            out = root / 'out'
            out.mkdir()
            with patch.object(build, 'SOURCE', source), \
                    patch.object(build.urllib.request, 'urlopen', return_value=io.BytesIO(archive)) as network:
                context, _ = build.prepare_context(out)
                self.assertEqual((context / 'sources/test/file.txt').read_text(), 'hello')
                self.assertTrue((out / 'context-sha256.json').is_file())
                network.assert_called_once()

    def test_pinned_context_can_be_prepared_without_docker(self):
        self.prepare(archive_bytes())

    def test_changed_snapshot_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'manifest mismatch'):
            self.prepare(archive_bytes(), tamper='runtime/exl3xpu/checkpoint.py')

    def test_unlisted_source_file_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'inventory differs'):
            self.prepare(archive_bytes(), tamper='runtime/exl3xpu/unreviewed.py')

    def test_wrong_archive_digest_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
            self.prepare(archive_bytes(), expected_digest='0' * 64)

    def test_wrong_wheel_digest_is_rejected_before_docker(self):
        with self.assertRaisesRegex(RuntimeError, 'Artifact checksum mismatch'):
            self.prepare(archive_bytes(), artifact_digest='0' * 64)

    def test_archive_cannot_escape_output_directory(self):
        with self.assertRaises(tarfile.FilterError):
            self.prepare(archive_bytes('../../escape'))


if __name__ == '__main__':
    unittest.main()
