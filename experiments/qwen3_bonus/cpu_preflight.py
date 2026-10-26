"""Offline readiness checks usable in AutoDL CPU-only mode; never starts training."""
from __future__ import annotations
import datetime,importlib,importlib.metadata,json,os,platform,sys,traceback
from pathlib import Path
from experiments.qwen3_bonus.common import ROOT,sha256,read_jsonl,write_json


def validate_frozen_data(folder: Path) -> dict:
    manifest=json.loads((folder/'dataset_manifest.json').read_text())
    for name,digest in manifest['output_hashes'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or sha256(path)!=digest:
            raise ValueError(f'Frozen data fingerprint mismatch: {name}')
    counts={name:len(read_jsonl(folder/name)) for name in manifest['output_hashes'] if name.endswith('.jsonl')}
    return {'fingerprints_verified':True,'counts':counts}


def main():
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true')
    report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':'cpu-only preparation',
        'python':sys.version,'platform':platform.platform(),'training_started':False,'gpu_runtime_verified':False,
        'production_enabled':False,'checks':{}}
    def check(name,fn):
        try:
            value=fn();report['checks'][name]={'passed':True,'detail':value}
            print('PASS',name,flush=True)
        except Exception as exc:
            report['checks'][name]={'passed':False,'error':f'{type(exc).__name__}: {exc}'}
            traceback.print_exc();print('FAIL',name,flush=True)
        write_json(ROOT/'artifacts/remote/cpu-readiness.json',report)
    def packages():
        expected={'torch':'2.6.0','vllm':'0.8.5','verl':'0.5.0','transformers':'4.51.3','peft':'0.15.2',
            'bitsandbytes':'0.45.5','accelerate':'1.6.0','ray':'2.43.0','numpy':'1.26.4',
            'matplotlib':'3.10.7','tensorboard':'2.20.0'}
        actual={k:importlib.metadata.version(k) for k in expected}
        for key,value in expected.items():
            if actual[key].split('+')[0]!=value:raise ValueError(f'{key}: {actual[key]} != {value}')
        return actual
    check('locked_package_versions',packages)
    def imports():
        for module in ['torch','transformers','peft','bitsandbytes','vllm','verl','pyarrow','matplotlib','tensorboard',
                       'verl.trainer.main_ppo','verl.workers.fsdp_workers']:
            importlib.import_module(module)
        return 'Trainer/worker imports successful; CUDA kernels remain untested'
    check('trainer_imports',imports)
    from experiments.qwen3_bonus.models import verify_model
    for kind in ('base','reward'):
        def model_check(kind=kind):
            from transformers import AutoConfig,AutoTokenizer
            from safetensors import safe_open
            folder=ROOT/'artifacts/models'/kind
            inventory=verify_model(folder,kind)
            config=AutoConfig.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
            tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
            ids=tokenizer.apply_chat_template([{'role':'user','content':'最近压力很大，想聊一聊。'}],tokenize=True,add_generation_prompt=True)
            tensors=0
            for file in folder.glob('*.safetensors'):
                with safe_open(file,framework='pt',device='cpu') as handle:tensors+=len(handle.keys())
            return {'revision':inventory['revision'],'files_verified':len(inventory['files']),
                    'model_type':config.model_type,'sample_prompt_tokens':len(ids),'tensor_headers':tensors}
        check(f'{kind}_offline_model',model_check)
    check('frozen_data',lambda:validate_frozen_data(ROOT/'artifacts/data'))
    def training_inputs():
        from transformers import AutoTokenizer
        from experiments.qwen3_bonus.training import completion_only_example
        import pyarrow.parquet as pq
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'artifacts/models/base',local_files_only=True)
        rows=read_jsonl(ROOT/'artifacts/data/classification_train.jsonl')
        for row in rows:completion_only_example(tokenizer,row,256)
        tables={split:pq.read_table(ROOT/f'artifacts/data/response_{split}.parquet').num_rows for split in ('train','eval')}
        return {'completion_masks_verified':len(rows),'response_parquet_rows':tables}
    check('training_inputs',training_inputs)
    def grpo_config():
        import verl
        from hydra import compose,initialize_config_dir
        from experiments.qwen3_bonus.training import verl_overrides
        with initialize_config_dir(config_dir=str(Path(verl.__file__).resolve().parent/'trainer/config'),version_base=None):
            config=compose(config_name='ppo_trainer',overrides=verl_overrides(ROOT/'artifacts/models/base',ROOT/'artifacts/models/reward',ROOT/'artifacts/data',ROOT/'artifacts/runs/pilot',2))
        return {'algorithm':config.algorithm.adv_estimator,'rollout':config.actor_rollout_ref.rollout.name,
                'lora_rank':config.actor_rollout_ref.model.lora_rank}
    check('verl_recipe_composition',grpo_config)
    report['passed']=all(row['passed'] for row in report['checks'].values())
    write_json(ROOT/'artifacts/remote/cpu-readiness.json',report)
    print(json.dumps({'passed':report['passed'],'gpu_runtime_verified':False,'training_started':False}),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
