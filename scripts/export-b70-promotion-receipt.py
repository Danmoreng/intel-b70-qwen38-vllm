#!/usr/bin/env python3
"""Export a successful service receipt without environment or generated text."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw.resolve()
    source = json.loads(raw.read_text())
    assert source['status'] == 'PASS_STRICT_PRODUCTION_DEPLOYMENT_AND_RESTART'
    first = source['stages']['first_start_api']
    restart = source['stages']['restart_api']
    assert first['container_id'] != restart['container_id']
    public = {key: source[key] for key in ['status', 'started_unix', 'finished_unix',
                                          'image_id', 'policy_sha256',
                                          'rollback_image_id', 'decision_sha256']}
    public['stages'] = {}
    for name in ['first_start_readiness', 'restart_readiness']:
        stage = source['stages'][name]
        assert stage['image_id'] == source['image_id'] and stage['models_endpoint_healthy']
        public['stages'][name] = {key: stage[key] for key in
                                 ['image_id', 'policy_sha256', 'models_endpoint_healthy', 'verified_unix']}
    for name in ['first_start_api', 'restart_api']:
        stage = source['stages'][name]
        assert stage['image_id'] == source['image_id']
        assert stage['policy_sha256'] == source['policy_sha256']
        assert stage['models_endpoint'] and stage['streaming_done'] and stage['streaming_usage']
        assert 'paris' in stage['real_generated_reply'].lower()
        message = stage['defaults_response']['choices'][0]['message']
        assert '377' in (message.get('content') or '')
        assert message.get('reasoning') or message.get('reasoning_content')
        env = stage['environment']
        assert env['EXL3_SDPA_CACHE_CAPACITY'] == '64'
        assert 'EXL3_MIGRATION_TRACE' not in env and 'B70_REVIEW_CACHE_DIAGNOSTIC' not in env
        assert env.get('VLLM_SERVER_DEV_MODE', '0') != '1'
        public['stages'][name] = {key: stage[key] for key in
                                 ['image_id', 'policy_sha256', 'container_id', 'models_endpoint',
                                  'streaming_done', 'streaming_usage', 'default_contract', 'verified_unix']}
        public['stages'][name]['environment_contract'] = dict(
            partition_cache_capacity=64, migration_trace=False,
            review_cache_diagnostic=False, server_dev_mode=False)
        public['stages'][name]['verified_requests'] = dict(
            real_generated_reply=True, stream_done_with_usage=True,
            default_reasoning_present=True, default_math_answer_correct=True)
    public['private_source_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
    public['scope'] = ('Sanitized export of the actual permanent-service first start and restart. '
                       'Original verifier checked real completions, streaming and default reasoning. '
                       'Environment values, mount paths and generated responses remain private; '
                       'this does not add a new quality/performance benchmark.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assert not args.out.exists(), 'Refuse to overwrite a deployment export'
    args.out.write_text(json.dumps(public, indent=2) + '\n')
    print('PASS_SANITIZED_ACTUAL_PRODUCTION_RECEIPT')


if __name__ == '__main__':
    main()
