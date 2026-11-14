"""Build the pinned upstream PTI graph-lifetime fix; preserve other runtimes."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(source, expected):
    for name, digest in expected.items():
        if sha(source / name) != digest:
            raise RuntimeError('Unsupported PTI source: ' + name)


def main():
    source = Path('/sources/pti')
    build = Path('/build-pti')
    prefix = Path('/opt/b70-pti')
    lock = json.loads((HERE / 'upstream.json').read_text())
    assert (source / 'sdk/VERSION').read_text().strip() == lock['version']
    patch = HERE / 'native-graph-lifetime.patch'
    assert sha(patch) == lock['patch_sha256']
    verify(source, lock['files_before_sha256'])
    subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-d', str(source),
                    '-i', str(patch)], check=True)
    verify(source, lock['files_after_sha256'])
    argv = ['cmake', '-S', str(source / 'sdk'), '-B', str(build), '-G', 'Ninja',
            '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=icx', '-DCMAKE_CXX_COMPILER=icpx',
            '-DCMAKE_INSTALL_PREFIX=' + str(prefix), '-DCMAKE_INSTALL_LIBDIR=lib',
            '-DBUILD_TESTING=OFF', '-DPTI_BUILD_TESTING=OFF', '-DPTI_BUILD_SAMPLES=OFF',
            '-DPTI_BUILD_URGEMM=OFF',
            '-DPTI_COMMIT_HASH=' + lock['base_commit'] + '+' + lock['fix_commit'][:7]]
    subprocess.run(argv, check=True)
    dependencies = {
        path.name: subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        for path in (build / '_deps').glob('*-src') if (path / '.git').exists()}
    assert dependencies == lock['build_dependencies'], dependencies
    subprocess.run(['cmake', '--build', str(build), '--parallel', '2'], check=True)
    subprocess.run(['cmake', '--install', str(build)], check=True)
    # Install only these two libraries in the final image. In particular, do
    # not replace the selected Level Zero loader or other Intel userspace.
    libraries = {'/opt/venv/lib/libpti_view.so.1.2.1': prefix / 'lib/libpti_view.so.1.2.1',
                 '/opt/venv/lib/pti/libpti.so': prefix / 'lib/pti/libpti.so'}
    receipt = {**lock, 'status': 'BUILT_GPU_UNTESTED', 'cmake_argv': argv,
               'library_sha256': {dest: sha(path) for dest, path in libraries.items()},
               'torch_source_modified': False, 'native_recording_disabled': False}
    Path('/native/pti-patch.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
