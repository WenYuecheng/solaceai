"""Small artifact helpers shared by the optional research package."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def output_path(path: str | Path) -> Path:
    result=Path(path).expanduser().resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError(f'Output must remain inside {ROOT}')
    return result


def sha256(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value) -> None:
    target=output_path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=target.name+'.',dir=target.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
        os.replace(name,target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]
