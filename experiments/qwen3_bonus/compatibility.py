"""Pre-rental Metal smoke with explicitly UNTRAINED zero LoRAs; never deployable."""
from __future__ import annotations
import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from experiments.qwen3_bonus.common import ROOT, output_path, write_json, sha256
from experiments.qwen3_bonus.models import SOURCES, verify_model


def create_temporary_adapters(raw_base: Path, output: Path) -> dict:
    """Build Qwen3 q/v PEFT tensors without loading a second model or using Torch."""
    import numpy as np
    from safetensors.numpy import save_file
    source=verify_model(raw_base,'base')
    config=json.loads((raw_base/'config.json').read_text())
    if config.get('model_type')!='qwen3':raise ValueError('Expected Qwen3 config')
    output=output_path(output)
    if output.exists():raise FileExistsError('Temporary adapter directory must be fresh')
    hidden=config['hidden_size'];head=config.get('head_dim',hidden//config['num_attention_heads'])
    sizes={'q_proj':config['num_attention_heads']*head,'v_proj':config['num_key_value_heads']*head}
    result={}
    for task in ('emotion','response'):
        path=output/f'solace-{task}';path.mkdir(parents=True)
        tensors={}
        for layer in range(config['num_hidden_layers']):
            for module,size in sizes.items():
                prefix=f'base_model.model.model.layers.{layer}.self_attn.{module}'
                tensors[prefix+'.lora_A.weight']=np.zeros((8,hidden),dtype=np.float32)
                tensors[prefix+'.lora_B.weight']=np.zeros((size,8),dtype=np.float32)
        save_file(tensors,str(path/'adapter_model.safetensors'))
        write_json(path/'adapter_config.json',{'peft_type':'LORA','task_type':'CAUSAL_LM','r':8,
            'lora_alpha':16,'lora_dropout':0.0,'bias':'none','target_modules':['q_proj','v_proj'],
            'base_model_name_or_path':source['repo_id'],'revision':source['revision'],'inference_mode':True})
        write_json(path/'bonus_adapter.json',{'trained':False,'purpose':'pre-rental zero-adapter compatibility only'})
        result[task]=str(path)
    return result


def validate_temporary(config) -> None:
    """Validate the converted base and zero fixtures separately from production."""
    # Production validation MUST reject these. Only this offline smoke can bypass it.
    for directory in (config.emotion_adapter,config.response_adapter):
        if json.loads((Path(directory)/'bonus_adapter.json').read_text()).get('trained') is not False:
            raise ValueError('Compatibility accepts only explicitly untrained temporary adapters')
    root=Path(config.model_path)
    source=json.loads((root/'bonus_source.json').read_text())
    if any(source.get(k)!=v for k,v in SOURCES['base'].items()) or source.get('quantization_bits')!=4:
        raise ValueError('Expected pinned converted MLX4 base')
    required={'config.json',*(p.name for p in root.glob('*.safetensors'))}
    if len(required)<2 or not required.issubset(source['files']):raise ValueError('Incomplete MLX base')
    for name,digest in source['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path)!=digest:raise ValueError('Base hash mismatch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-base',type=Path,default=ROOT/'artifacts/models/base')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/compatibility')
    args=parser.parse_args()
    from experiments.qwen3_bonus.serving import ServingConfig
    from experiments.qwen3_bonus.probe import live_probe
    output=output_path(args.output)
    paths=create_temporary_adapters(args.raw_base,output/'temporary-adapters')
    config=replace(ServingConfig.from_file(args.config),emotion_adapter=paths['emotion'],response_adapter=paths['response'])
    validate_temporary(config)
    models=('solace-emotion','solace-response')*2
    report=asyncio.run(live_probe(config,validate=False,models=models))
    passed=report['genuine_inference'] and [r['model'] for r in report['requests']]==list(models) and not report['error_type']
    report.update(trained=False,release_gate_passed=False,compatibility_passed=bool(passed),
        base_revision=SOURCES['base']['revision'],base_inventory_sha256=sha256(args.raw_base/'bonus_source.json'),
        scope='Zero LoRA transport compatibility; not model quality.')
    write_json(output/'report.json',report)
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
