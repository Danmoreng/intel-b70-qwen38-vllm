#!/usr/bin/env python3
import ast
import hashlib
from pathlib import Path
import sys

path = Path(sys.argv[1]); source = path.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == '5d47acc849c678092acc5231527028f842239db42ab55ab7ecfb5e6f89d6311c'
old = '    from . import loader_audit\n'
new = "    if os.environ.get('EXL3_DFLASH_W4') == '1':\n        from . import draft_w4\n"+old
assert source.count(old)==1
source = source.replace(old,new); ast.parse(source); path.write_text(source)
