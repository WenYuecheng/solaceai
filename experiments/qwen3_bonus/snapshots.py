"""Small immutable PEFT copies survive rotation of large trainer checkpoints."""
from pathlib import Path
import shutil
from experiments.qwen3_bonus.common import output_path,write_json,sha256


def save_snapshot(adapter: Path,directory: Path,step: int) -> Path:
    if type(step) is not int or step<=0:raise ValueError('Snapshot step must be positive')
    destination=output_path(directory/f'step-{step:06d}')
    if destination.exists():raise FileExistsError('Snapshot already exists')
    names=('adapter_config.json','adapter_model.safetensors')
    if any(not (adapter/name).is_file() for name in names):raise ValueError('Incomplete adapter export')
    destination.mkdir(parents=True)
    for name in names:shutil.copyfile(adapter/name,destination/name)
    write_json(destination/'snapshot.json',{'step':step,'files':{name:sha256(destination/name) for name in names},
        'scope':'Adapter weights/config only; optimizer resume state remains in the latest full checkpoint.'})
    return destination
