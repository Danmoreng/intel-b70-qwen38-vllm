#!/usr/bin/env python3
"""Source-guarded widened target verification; all other dispatch guards stay."""
import ast
import hashlib
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
source = path.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == 'efb06f59250cbf4efb16f3eb6eb119bbd4c24c9ff086459b4afb85cc21adacb2'
old = 'not 2 <= rows <= 5'
assert source.count(old) == 1
source = source.replace(old, 'not 2 <= rows <= 9')
old = '{2: 32, 3: 8, 4: 16, 5: 16}[rows]'
assert source.count(old) == 1
source = source.replace(old, '{2: 32, 3: 8, 4: 16, 5: 16, 6: 32, 7: 32, 8: 32, 9: 32}[rows]')
old = '    splits = splits if splits is not None else'
assert source.count(old) == 1
source = source.replace(old, '    adaptive_splits = rows >= 6 and splits is None and tile == 8\n' + old)
old = "y, temp, sums, maxima, d['max_seqlen_k'], splits, tile)"
assert source.count(old) == 1
source = source.replace(old, "y, temp, sums, maxima, d['max_seqlen_k'], splits, tile, adaptive_splits)")
ast.parse(source)
path.write_text(source)
print('WIDE_WRAPPER_SHA256', hashlib.sha256(source.encode()).hexdigest())
profile = Path(sys.argv[2])
config = yaml.safe_load(profile.read_text())
assert config['env']['EXL3_M04_LIBRARY_SHA256'] == '03ed092ce98faa5e1da18ed7f07b5b6862949bee2be0e742a0c28ef42ea1b6b3'
library = Path(config['env']['EXL3_M04_LIBRARY'])
config['env']['EXL3_M04_LIBRARY_SHA256'] = hashlib.sha256(library.read_bytes()).hexdigest()
profile.write_text(yaml.safe_dump(config, sort_keys=False))
print('WIDE_LIBRARY_SHA256', config['env']['EXL3_M04_LIBRARY_SHA256'])
