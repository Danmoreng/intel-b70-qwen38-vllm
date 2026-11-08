#!/usr/bin/env python3
"""Compare actual rendered token IDs for every frozen serving request."""
import argparse
from array import array
import copy
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).parent / 'runtime'))
from request_defaults import apply_defaults

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base', required=True)
p.add_argument('--fixtures', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
p.add_argument('--compare', type=Path)
a = p.parse_args()
assert not a.out.exists()
rows = []
waves = 0
for summary in sorted(a.fixtures.glob('*/summary.json')):
    data = json.loads(summary.read_text())
    if data.get('repeat', 0) < 1 or summary.parent.name.startswith('warmup'):
        continue
    waves += 1
    for index, request in enumerate(data['requests'], 1):
        body = json.loads((summary.parent / f'request-{index}.json').read_text())
        # Benchmark receipts redact message bodies and save their exact bytes
        # separately, just as current-profile-benchmark's frozen replay does.
        prompt = (summary.parent / f'prompt-{index}.txt').read_text()
        assert hashlib.sha256(prompt.encode()).hexdigest() == body['prompt_sha256']
        body['messages'] = [{'role': 'user', 'content': prompt}]
        normalized = apply_defaults(copy.deepcopy(body))
        payload = {k: normalized[k] for k in ('model', 'messages', 'chat_template_kwargs', 'tools', 'documents')
                   if k in normalized}
        payload['add_generation_prompt'] = True
        raw = json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode()
        http = urllib.request.Request(a.base + '/tokenize', raw, {'Content-Type': 'application/json'})
        with urllib.request.urlopen(http, timeout=60) as response:
            result = json.load(response)
        ids = array('I', result['tokens'])
        assert ids.itemsize == 4
        if sys.byteorder != 'little':
            ids.byteswap()
        assert len(ids) == result['count'] == request['usage']['prompt_tokens'], summary.parent.name
        rows.append({'wave': summary.parent.name, 'request': index, 'count': len(ids),
                     'token_ids_sha256_u32le': hashlib.sha256(ids.tobytes()).hexdigest(),
                     'rendering_request_sha256': hashlib.sha256(raw).hexdigest()})
assert waves == 70 and len(rows) == 124
report = {'status': 'PASS_TOKENIZER_BASELINE_COUNTS', 'waves': waves, 'requests': len(rows),
          'rows': rows, 'base': a.base,
          'scope': 'Actual /tokenize IDs after the identical qualified chat-default normalization. Hash encoding is unsigned32bit little endian.'}
if a.compare:
    baseline = json.loads(a.compare.read_text())
    mismatches = [{'baseline': x, 'candidate': y} for x,y in zip(baseline['rows'],rows) if x != y]
    if len(baseline['rows']) != len(rows):
        mismatches.append({'baseline_count':len(baseline['rows']),'candidate_count':len(rows)})
    report.update(status='FAILED_TOKENIZER_PAIR' if mismatches else 'PASS_ALL124_RENDERED_TOKEN_IDS_IDENTICAL',
                  mismatches=mismatches, baseline_receipt_sha256=hashlib.sha256(a.compare.read_bytes()).hexdigest())
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(report, indent=2) + '\n')
print(report['status'], waves, len(rows))

assert report['status'] != 'FAILED_TOKENIZER_PAIR', 'Rendered token IDs or tokenizer input differ across runtimes; receipt saved'
