"""Hash-pinned AutoDL stop script bridge; never runs an inline shell command."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
from experiments.qwen3_bonus.common import sha256

AUTODL_SHUTDOWN_SHA256='0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027'


def verify_platform_script(path: Path,expected: str) -> None:
    if not path.is_file() or path.is_symlink() or len(expected)!=64 or sha256(path)!=expected:
        raise ValueError('Platform shutdown script hash mismatch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--script',type=Path,default=Path('/usr/bin/shutdown'))
    parser.add_argument('--sha256',default=AUTODL_SHUTDOWN_SHA256)
    args=parser.parse_args()
    if args.script!=Path('/usr/bin/shutdown'):raise ValueError('Only the inspected AutoDL script is supported')
    verify_platform_script(args.script,args.sha256)
    if args.execute:
        # This provider file has no shebang; a fixed script argument is required.
        # Its exact inspected bytes are verified immediately before execution.
        raise SystemExit(subprocess.run(['/bin/bash',str(args.script)],check=False,timeout=30).returncode)
    print('platform_script_verified; stop not executed')


if __name__=='__main__':main()
