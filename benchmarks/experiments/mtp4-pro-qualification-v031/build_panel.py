"""CPU-only frozen, preweighted bounded panel. Run in the pinned image."""
import gzip
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer

sha = lambda data: hashlib.sha256(data).hexdigest()
tok = AutoTokenizer.from_pretrained('/model', local_files_only=True)
old_bytes = Path('/panel/mtp-panel.json.gz').read_bytes()
old_manifest = json.loads(Path('/panel/mtp-panel-manifest.json').read_text())
assert sha(old_bytes) == old_manifest['panel_gzip_sha256']
assert sha(gzip.decompress(old_bytes)) == old_manifest['panel_raw_sha256']
old = json.loads(gzip.decompress(old_bytes))
rows = []
for size in ['32k', '64k']:
    prompt = Path(f'/fixture/phase-{size}-c1-r1/prompt-1.txt').read_text()
    ids = tok.apply_chat_template([dict(role='user', content=prompt)], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    if isinstance(ids, dict) or hasattr(ids, 'input_ids'):
        ids = ids['input_ids']
    assert isinstance(ids, list) and isinstance(ids[0], int)
    rows.append(dict(name=f'source-{size}', domain='code', ids=ids,
        concurrency=1, weight=.35, context_tokens=len(ids), source_prompt_sha256=sha(prompt.encode())))
prose = next(w for w in old['windows'] if w['name']=='prose-32768')
rows.append({**prose, 'concurrency':4, 'weight':.30})
panel = dict(windows=rows, sampling=dict(temperature=1., top_p=.95, top_k=20,
    seed=770000, max_tokens=512, ignore_eos=True),
    weights_declared_before_measurement=True, cache_modes=['cold', 'warm'])
raw = json.dumps(panel, separators=(',', ':')).encode()
encoded = gzip.compress(raw, mtime=0)
Path('/out/panel.json.gz').write_bytes(encoded)
Path('/out/panel-manifest.json').write_text(json.dumps(dict(panel_raw_sha256=sha(raw),
    panel_gzip_sha256=sha(encoded), source_prose_panel_sha256=sha(old_bytes),
    windows=[{k:v for k,v in row.items() if k!='ids'} for row in rows]), indent=2)+'\n')
print([(r['name'], len(r['ids']), r['concurrency']) for r in rows])
