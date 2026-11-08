"""A failed deployment must restore modules before restarting the old image."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('promote_with_runtime', ROOT/'scripts/promote-exl3-review-release.py')
promote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promote)


class PromotionRuntimeRecovery(unittest.TestCase):
    def test_failed_api_gate_restores_config_runtime_and_private_env(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'repo'
            for name in ('config', 'runtime', 'candidate', 'candidate/runtime', 'scripts', 'saved', 'results'):
                (root/name).mkdir(parents=True, exist_ok=True)
            def write(name, value):
                path=root/name
                path.write_text(json.dumps(value) if isinstance(value, dict) else value)
                return path
            def digest(name):
                return hashlib.sha256((root/name).read_bytes()).hexdigest()
            modules=('request_defaults.py','kv_admission.py','xpu_cpu_offload.py')
            for name in modules:
                write('runtime/'+name, 'old '+name)
                write('candidate/runtime/'+name, 'new '+name)
            write('config/production_image.json', {'image_id':'old-image'})
            write('config/production_policy.json', 'old policy')
            write('config/production_policy.sha256', 'old digest')
            write('config/production_ram_kv.json', 'old RAM manifest')
            write('.env', 'VLLM_IMAGE=old-tag\nEXL3_TARGET_RUNTIME=vllm031\nPRIVATE_KEY=preserve-me\n')
            write('saved/production_image.json', {'image_id':'old-image','image_tag':'old-tag'})
            write('saved/snapshot-inventory.json', {'files_sha256':{}})
            write('candidate/production_policy.json', 'new policy')
            policy=digest('candidate/production_policy.json')
            write('candidate/production_policy.sha256', policy)
            write('candidate/ram.json', {
                'status':'PRODUCTION','base_image_id':'new-image','default_cpu_gib':12,
                'kv_cache_layout':'LBNHC','runtime_directory':'candidate/runtime',
                'runtime_sha256':{name:digest('candidate/runtime/'+name) for name in modules[1:]}})
            write('candidate/production_image.json', {
                'image_id':'new-image','policy_sha256':policy,
                'request_defaults_sha256':digest('candidate/runtime/request_defaults.py'),
                'ram_kv_runtime':{'path':'candidate/ram.json','sha256':digest('candidate/ram.json')}})
            write('results/gates.json', {'status':'PASS_EXACT_IMAGE_RELEASE_GATES','image_id':'new-image'})
            write('results/benchmark.json', {})
            write('results/decision.json', {
                'status':'QUALIFIED_FOR_PRODUCTION_PROMOTION','image_id':'new-image',
                'policy_sha256':policy,'immediate_rollback_image_id':'old-image','rationale':'test',
                'gate_assessment':{'path':'results/gates.json','sha256':digest('results/gates.json')},
                'benchmark_summary':{'path':'results/benchmark.json','sha256':digest('results/benchmark.json')}})
            write('scripts/run-exl3-review-micro.py', 'def wait_for_qualified_service(image): return {"image":image}\n')
            protected=[root/'config'/name for name in ('production_image.json','production_policy.json',
                       'production_policy.sha256','production_ram_kv.json')]
            protected += [root/'runtime'/name for name in modules]+[root/'.env']
            before={path:path.read_bytes() for path in protected}
            starts=[]
            def check_output(command, **kwargs):
                if '--property=WorkingDirectory' in command:return str(root)+'\n'
                return f'path={root}/scripts/run-server.sh ; argv[]={root}/scripts/run-server.sh ;'
            def run(command, **kwargs):
                if command[:4]==['systemctl','--user','start',promote.SERVICE]:
                    starts.append(True)
                    if len(starts)==1:
                        self.assertEqual((root/'runtime/xpu_cpu_offload.py').read_text(),'new xpu_cpu_offload.py')
                    else:
                        for path, value in before.items():self.assertEqual(path.read_bytes(),value)
                return subprocess.CompletedProcess(command, 1 if command[0]=='docker' and command[1] in ('inspect','image') else 0,
                                                   stdout='',stderr='')
            argv=['promote','--release-dir',str(root/'candidate'),'--decision',str(root/'results/decision.json'),
                  '--out',str(root/'results/deploy'),'--rollback-release',str(root/'saved'),
                  '--production-tag','local/b70-qwen38-vllm:production-exl3-test',
                  '--runtime-dir',str(root/'candidate/runtime')]
            with patch.object(promote,'REPO',root), patch.object(promote.Path,'home',return_value=Path(temporary)), \
                 patch.object(sys,'argv',argv), patch.object(promote.subprocess,'check_output',side_effect=check_output), \
                 patch.object(promote.subprocess,'run',side_effect=run), \
                 patch.object(promote,'verify',side_effect=RuntimeError('injected API failure')), \
                 patch('exl3_candidate_worker.profile',return_value={'env':{'EXL3_TARGET_RUNTIME':'vllm033'}}):
                with self.assertRaisesRegex(RuntimeError,'injected API failure'):promote.main()
            self.assertEqual(len(starts),2)
            for path,value in before.items():self.assertEqual(path.read_bytes(),value)
            receipt=json.loads((root/'results/deploy/promotion.json').read_text())
            self.assertEqual(receipt['status'],'FAILED_SAVED_RELEASE_RECOVERED')


if __name__=='__main__':unittest.main()
