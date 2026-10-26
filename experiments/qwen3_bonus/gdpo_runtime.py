"""Finite rental accounting and atomic, integrity-checked GDPO checkpoints."""
from __future__ import annotations

import json
import math
import os
import shutil
import stat
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from experiments.qwen3_bonus.common import output_path, sha256, write_json


@dataclass(frozen=True)
class GDPOBudget:
    rate: float
    started_at: float
    quantum_seconds: float = 60
    cap_yuan: float = 20
    reserve_yuan: float = 2

    def __post_init__(self):
        values = (self.rate, self.started_at, self.quantum_seconds, self.cap_yuan, self.reserve_yuan)
        if any(not math.isfinite(value) for value in values):
            raise ValueError('Budget values must be finite')
        if self.rate <= 0 or self.started_at < 0 or self.quantum_seconds <= 0:
            raise ValueError('Actual rate, billing timestamp and quantum are required')
        if not 2 <= self.reserve_yuan < self.cap_yuan <= 20:
            raise ValueError('Cap must be at most 20 yuan, with at least 2 yuan reserved')
        if not math.isfinite(self.started_at+self.cap_yuan*3600/self.rate):
            raise ValueError('Budget deadline exceeds finite clock range')

    def cost(self, now: float) -> float:
        if not math.isfinite(now) or now < self.started_at:
            raise ValueError('Billing timestamp cannot be in the future')
        try:
            amount = math.ceil((now-self.started_at)/self.quantum_seconds)*self.quantum_seconds*(self.rate/3600)
        except OverflowError as error:
            raise ValueError('Billing cost exceeds finite numeric range') from error
        if not math.isfinite(amount):
            raise ValueError('Billing cost exceeds finite numeric range')
        return amount

    def _deadline(self, amount: float) -> float:
        # One quantum of headroom avoids crossing a rounded billing boundary.
        quanta = math.floor(amount*3600/self.rate/self.quantum_seconds)
        return self.started_at + max(0, quanta-1)*self.quantum_seconds

    def release_deadline(self) -> float:
        return self._deadline(self.cap_yuan)

    def setup_deadline(self) -> float:
        return self._deadline(2)

    def training_deadline(self, save_seconds: float = 120) -> float:
        if not math.isfinite(save_seconds) or save_seconds < 120:
            raise ValueError('Saving requires at least 120 seconds')
        return min(self._deadline(self.cap_yuan-self.reserve_yuan), self.release_deadline()-save_seconds)

    def status(self, now: float, completed_step: int, longest_step: float, save_seconds: float = 120) -> dict:
        cost = self.cost(now)
        deadline = self.training_deadline(save_seconds)
        if not isinstance(completed_step, int) or completed_step < 0:
            raise ValueError('A completed nonnegative optimizer step is required')
        if not math.isfinite(longest_step) or longest_step < 0:
            raise ValueError('Longest completed step must be finite and nonnegative')
        predicted = longest_step*1.25
        if not math.isfinite(predicted) or not math.isfinite(predicted*(self.rate/3600)):
            raise ValueError('Step estimate exceeds finite numeric range')
        available = max(0., deadline-now)
        setup_exceeded = completed_step <= 2 and (now >= self.setup_deadline() or cost > 2)
        if completed_step < 2:
            available = min(available, max(0., self.setup_deadline()-now))
        return {'accumulated_cost':cost, 'remaining':max(0.,self.cap_yuan-cost),
                'estimated_step_cost':predicted*(self.rate/3600),
                'remaining_steps':math.floor(available/predicted) if predicted else 0,
                'completed_step':completed_step, 'longest_step_seconds':longest_step,
                'setup_exceeded':setup_exceeded,
                'should_stop':setup_exceeded or now+predicted >= deadline or (completed_step < 2 and now+predicted >= self.setup_deadline()),
                'training_deadline':deadline, 'release_deadline':self.release_deadline(),
                'setup_deadline':self.setup_deadline(), 'save_seconds':save_seconds}


def should_save(step: int, last_saved: int, stopping: bool = False) -> bool:
    return step > last_saved and (step == 2 or step % 8 == 0 or stopping)


def _relative(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Checkpoint file paths must be relative and contained')
    path = output_path(root/name)
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Checkpoint file escaped its directory')
    return path


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size == 0:
        raise ValueError(f'Checkpoint needs a nonzero regular file: {path.name}')


def _validate_adapter(config_path: Path) -> None:
    try:
        config = json.loads(config_path.read_text())
        if not isinstance(config, dict) or any(not math.isfinite(float(config[key])) or float(config[key]) <= 0 for key in ('r','lora_alpha')):
            raise ValueError('Invalid adapter configuration')
        weights = config_path.with_name('adapter_model.safetensors')
        _regular(weights)
        with weights.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('Invalid safetensors header')
            length = struct.unpack('<Q', prefix)[0]
            if length < 2 or length > min(weights.stat().st_size-8, 16*1024*1024):
                raise ValueError('Invalid safetensors header size')
            header = json.loads(stream.read(length))
        sizes = {'F64':8,'F32':4,'F16':2,'BF16':2,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}
        ranges = []
        for key, tensor in header.items():
            if key == '__metadata__':
                continue
            shape, (start,end) = tensor['shape'], tensor['data_offsets']
            if not shape or any(not isinstance(n,int) or n <= 0 for n in shape):
                raise ValueError('Invalid adapter tensor shape')
            if not isinstance(start,int) or not isinstance(end,int) or start < 0 or end-start != math.prod(shape)*sizes[tensor['dtype']]:
                raise ValueError('Invalid adapter tensor offsets')
            ranges.append((start,end))
        ranges.sort()
        if not ranges or ranges[0][0] != 0 or ranges[-1][1] != weights.stat().st_size-8-length:
            raise ValueError('Invalid adapter safetensors payload')
        if any(left[1] != right[0] for left,right in zip(ranges,ranges[1:])):
            raise ValueError('Invalid adapter safetensors overlap')
    except (KeyError,TypeError,ValueError,OSError,struct.error) as error:
        raise ValueError(f'Invalid adapter/safetensors: {error}') from error


class AtomicCheckpointStore:
    """Single-writer store; save_callback must write all resumable state into staging."""
    def __init__(self, run: Path):
        self.run = output_path(run)
        self.run.mkdir(parents=True, exist_ok=True)

    def _required(self, root: Path, required_files: list[str]) -> None:
        if not required_files:
            raise ValueError('Explicit optimizer, dataloader and RNG required files are required')
        for name in required_files:
            if name in ('COMPLETE.json','manifest.json','metadata.json'):
                raise ValueError('Required state cannot be a store control file')
            _regular(_relative(root, name))

    def _files(self, root: Path) -> dict[str,str]:
        files = {}
        for path in root.rglob('*'):
            if path.is_symlink():
                raise ValueError('Checkpoint symlinks are forbidden')
            if path.is_dir():
                continue
            _regular(path)
            if path.name == 'adapter_config.json':
                _validate_adapter(path)
            files[str(path.relative_to(root))] = sha256(path)
        return files

    def commit(self, step: int, save_callback: Callable[[Path],None], metadata: dict, required_files: list[str]) -> Path:
        if not isinstance(step,int) or step < 1:
            raise ValueError('Only completed positive optimizer steps can be saved')
        pointer = self.run/'latest.json'
        if pointer.exists() and step <= json.loads(pointer.read_text())['step']:
            raise ValueError('Checkpoint steps must increase beyond the newer completed step')
        destination = output_path(self.run/f'global_step_{step}')
        if destination.exists():
            raise ValueError('Checkpoint step already exists')
        # Check paths before invoking potentially expensive framework saves.
        for name in required_files:
            _relative(self.run, name)
        if not required_files:
            raise ValueError('Explicit required files are required')
        json.dumps(metadata,allow_nan=False)
        staging = Path(tempfile.mkdtemp(prefix=f'.step-{step}-',dir=self.run))
        try:
            save_callback(staging)
            self._required(staging,required_files)
            if any((staging/name).exists() for name in ('COMPLETE.json','manifest.json','metadata.json')):
                raise ValueError('Save callback wrote reserved control files')
            write_json(staging/'metadata.json',metadata)
            files = self._files(staging)
            write_json(staging/'manifest.json',{'step':step,'files':files,'required_files':required_files})
            write_json(staging/'COMPLETE.json',{'step':step,'manifest_sha256':sha256(staging/'manifest.json')})
            # Flush file contents before publishing the directory and latest pointer.
            for path in staging.rglob('*'):
                if path.is_file():
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
            os.replace(staging,destination)
            write_json(self.run/'latest.json',{'step':step,'directory':destination.name})
            for old in self.run.glob('global_step_*'):
                if old != destination and old.is_dir() and not old.is_symlink() and (old/'COMPLETE.json').is_file():
                    shutil.rmtree(old)
            return destination
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def resume(self, expected_identity: dict, required_files: list[str]) -> dict | None:
        pointer = self.run/'latest.json'
        if not pointer.exists():
            return None
        latest = json.loads(pointer.read_text())
        step = latest['step']
        if not isinstance(step,int) or latest.get('directory') != f'global_step_{step}':
            raise ValueError('Invalid latest checkpoint pointer')
        root = _relative(self.run,latest['directory'])
        _regular(root/'COMPLETE.json')
        _regular(root/'manifest.json')
        complete = json.loads((root/'COMPLETE.json').read_text())
        manifest = json.loads((root/'manifest.json').read_text())
        if complete.get('step') != step or manifest.get('step') != step or complete.get('manifest_sha256') != sha256(root/'manifest.json'):
            raise ValueError('Checkpoint manifest integrity mismatch')
        self._required(root,required_files)
        expected = manifest['files']
        actual = self._files(root)
        actual.pop('manifest.json',None)
        actual.pop('COMPLETE.json',None)
        if actual != expected:
            raise ValueError('Checkpoint file hash integrity mismatch')
        metadata = json.loads((root/'metadata.json').read_text())
        if metadata.get('identity',{}) != expected_identity:
            raise ValueError('Checkpoint identity mismatch')
        return {'step':step,'path':root,'metadata':metadata,'manifest':manifest}
