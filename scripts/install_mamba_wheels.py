#!/usr/bin/env python3
"""Install matching Mamba-1 CUDA extensions from official GitHub release assets.

Pinned to PyTorch 2.5.x + CUDA 12 + Python 3.10. No silent source build.
"""
from pathlib import Path
import subprocess
import sys
import urllib.request
import torch

if sys.platform!='linux' or sys.version_info[:2]!=(3,10):
    raise SystemExit('This installation recipe targets Linux Python 3.10 only')
if not torch.cuda.is_available() or not torch.__version__.startswith('2.5.') or not str(torch.version.cuda).startswith('12.'):
    raise SystemExit('Install PyTorch 2.5.1 CUDA 12.1 before running this script')
abi=str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
base=Path('wheels');base.mkdir(parents=True,exist_ok=True)
for package,version,repo in (
    ('causal_conv1d','1.5.0.post8','Dao-AILab/causal-conv1d'),
    ('mamba_ssm','2.2.4','state-spaces/mamba'),
):
    fname=f'{package}-{version}+cu12torch2.5cxx11abi{abi}-cp310-cp310-linux_x86_64.whl'
    url=f'https://github.com/{repo}/releases/download/v{version}/{fname}'
    dest=base/fname
    if not dest.is_file():
        print(f'Downloading official CUDA wheel: {url}',flush=True)
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PhaseMamba-Conda-Setup'}),timeout=80) as src, dest.open('wb') as f:
                import shutil
                shutil.copyfileobj(src,f)
        except Exception:
            dest.unlink(missing_ok=True)
            print('Cannot download matching precompiled wheel; do NOT install an ABI-mismatched wheel.',file=sys.stderr)
            print('Try the releases page manually and ensure torch2.5/cu12/cp310/cxx11abi match.',file=sys.stderr)
            raise
    print(f'Installing {dest}',flush=True)
    subprocess.run([sys.executable,'-m','pip','install','--no-deps',str(dest)],check=True)
print('Mamba-1 and causal-conv1d wheel installation complete')
