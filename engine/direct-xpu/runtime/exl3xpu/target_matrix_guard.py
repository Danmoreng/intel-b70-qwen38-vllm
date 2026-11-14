"""Experimental native-artifact guard, in addition to all original source guards."""
import hashlib
import os
from pathlib import Path
EXPECTED = {'_C.so': 'd91199a8dcf983240a82fb7acd886677cde7e1cd9d9491880a155cab3f3f4e17', '_C.so.manifest.json': '00820bca43fabfe019afc23b31edbfadc322efc43709673f18cecc55101e4a15', 'vllm_plugin.py': '000238efdd4fd3621437763ab156dc6502eb11d865e4994d194385899cf014a6'}

def check():
    root=Path(__file__).parent
    override=os.environ.get('EXL3_LIB')
    if override and Path(override).resolve() != (root/'_C.so').resolve():
        raise RuntimeError('Unqualified EXL3_LIB override for target matrix candidate')
    actual={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in EXPECTED}
    if actual != EXPECTED:raise RuntimeError('Unknown target matrix artifacts: '+str(actual))
