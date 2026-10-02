"""Both qualified EXL3 versions must use the pinned local-directory downloader."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]


class ModelDownloadDispatch(unittest.TestCase):
    def invoke(self, policy, entry='download-model.sh', extra=()):
        with tempfile.TemporaryDirectory(prefix='b70-download-dispatch-') as tmp:
            root = Path(tmp)
            (root/'scripts').mkdir()
            (root/'config').mkdir()
            (root/'bin').mkdir()
            for name in ['download-model.sh', 'download-exl3-model.sh']:
                shutil.copyfile(REPO/'scripts'/name, root/'scripts'/name)
            (root/'config/production_policy.json').write_text(json.dumps(dict(policy_id=policy,
                model=dict(id='test/pinned-exl3', revision='frozen-revision'))))
            (root/'config/production_image.json').write_text(json.dumps(dict(image_id='sha256:pinned-image')))
            (root/'.env').write_text('VLLM_IMAGE=wrong-serving-alias\nMODEL_ID=wrong/model\n')
            log=root/'calls.json'
            docker=root/'bin/docker'
            docker.write_text('#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ["B70_DOWNLOAD_TEST_LOG"],"w").write(json.dumps(sys.argv[1:]))\n')
            docker.chmod(0o755)
            env={**os.environ,'PATH':str(root/'bin')+os.pathsep+os.environ['PATH'],'B70_DOWNLOAD_TEST_LOG':str(log)}
            result=subprocess.run(['bash',str(root/'scripts'/entry),*extra],capture_output=True,text=True,env=env,timeout=10)
            return result,json.loads(log.read_text()) if log.exists() else None

    def test_v1_and_v2_download_to_pinned_exl3_cache(self):
        for version in (1,2):
            with self.subTest(version=version):
                result,argv=self.invoke(f'b70-qwen38-exl3-production-v{version}')
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn(str(Path.home()/'.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw')+':/models/checkpoint',argv)
                self.assertEqual(argv[argv.index('--entrypoint')+2],'sha256:pinned-image')
                self.assertEqual(argv[-6:],['download','test/pinned-exl3','--revision','frozen-revision','--local-dir','/models/checkpoint'])
                self.assertNotIn('wrong-serving-alias',argv)

    def test_unknown_policy_or_override_starts_no_download(self):
        for policy,extra in [('unexpected',()),('b70-qwen38-exl3-production-v2',('--override',))]:
            with self.subTest(policy=policy,extra=extra):
                result,argv=self.invoke(policy,'download-exl3-model.sh',extra)
                self.assertNotEqual(result.returncode,0)
                self.assertIsNone(argv)
