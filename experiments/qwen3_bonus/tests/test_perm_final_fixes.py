import json
from pathlib import Path
import pytest
from experiments.qwen3_bonus import perm_launch as launch, perm_entry as entry, perm_evaluation as ev
from experiments.qwen3_bonus.perm_remote import collection_allowed


def test_actual_snapshot_layout(tmp_path):
    p=tmp_path/'run/adapter_snapshots/step-8/adapter_model.safetensors';p.parent.mkdir(parents=True);p.write_bytes(b'x')
    assert collection_allowed(p,tmp_path)
    p=tmp_path/'run/checkpoints/step-8/adapter_model.safetensors';p.parent.mkdir(parents=True);p.write_bytes(b'x')
    assert not collection_allowed(p,tmp_path)


def test_remaining_accounts_for_elapsed_and_single_save_reserve():
    b=launch.training_budget(rate=3,started_at=0,quantum_seconds=60,preparation_cost=1)
    r=launch.live_resources({'prep_setup_save_reserve_yuan':99},b,1800)
    assert r['remaining_training_yuan']==17.5
    assert r['prep_setup_save_reserve_yuan']==2
    with pytest.raises(ValueError):launch.training_budget(rate=True,started_at=0,quantum_seconds=60,preparation_cost=0)


def test_arbitrary_dev_summary_rejected():
    with pytest.raises(ValueError):ev.select_checkpoint([dict(split='dev',step=1,severe_count=0,semantic_complete=999,**dict.fromkeys(ev.AXES,3))])


def test_prepare_order_once():
    events=[]
    class Trainer:
        def init_workers(self):events.append('init')
        def fit(self):events.append('fit')
    entry.initialize_prepare_fit(Trainer(),lambda:events.append('sample'),lambda baseline:events.append('select'))
    assert events==['sample','init','select','fit']


def test_canonical_base_ignores_readme_but_verifies_content(tmp_path):
    from experiments.qwen3_bonus.models import SOURCES
    from experiments.qwen3_bonus.common import sha256
    (tmp_path/'config.json').write_text('{}');(tmp_path/'model.safetensors').write_bytes(b'weights')
    def inventory():
        (tmp_path/'bonus_source.json').write_text(json.dumps({**SOURCES['base'],'files':{p.name:sha256(p) for p in tmp_path.iterdir() if p.name!='bonus_source.json'}}))
    inventory();first=launch.base_content_identity(tmp_path)
    (tmp_path/'model.safetensors.ranges.json').write_text('{\"download\":true}')
    assert first==launch.base_content_identity(tmp_path)
    (tmp_path/'chat_template.jinja').write_text('injected')
    with pytest.raises(ValueError):launch.base_content_identity(tmp_path)
    (tmp_path/'chat_template.jinja').unlink()
    (tmp_path/'README.md').write_text('documentation');inventory()
    assert first==launch.base_content_identity(tmp_path)
    (tmp_path/'model.safetensors').write_bytes(b'drift')
    with pytest.raises(ValueError):launch.base_content_identity(tmp_path)


def test_preparation_cli_without_passed_selection(tmp_path, monkeypatch):
    import sys, time, types
    from experiments.qwen3_bonus import common, perm_models, gdpo_watchdog, gdpo_platform
    monkeypatch.setattr(common,'ROOT',tmp_path)
    base=tmp_path/'base';base.mkdir();(base/'bonus_source.json').write_text('{}')
    data=tmp_path/'data';data.mkdir();(data/'dataset_manifest.json').write_text('{}')
    preflight=tmp_path/'preflight.json';preflight.write_text('{}')
    first=tmp_path/'preferred';second=tmp_path/'fallback'
    receipts=[]
    monkeypatch.setattr(launch.platform,'system',lambda:'Linux')
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True,device_count=lambda:1)))
    monkeypatch.setattr(gdpo_platform,'verify_platform_script',lambda *a:None)
    monkeypatch.setattr(launch,'verify_base',lambda *a:{})
    monkeypatch.setattr(launch,'verify_data',lambda *a:{})
    monkeypatch.setattr(launch,'validate_preflight',lambda *a:None)
    monkeypatch.setattr(launch,'base_content_identity',lambda *a:'b'*64)
    def inventory(path):
        approved=perm_models.CANDIDATES['1.7b' if path==first else '0.6b']
        return {**approved,'model_path':str(path),'model_inventory_sha256':'c'*64}
    monkeypatch.setattr(perm_models,'verify_model',inventory)
    monkeypatch.setattr(gdpo_watchdog,'launch_watchdog',lambda *a:receipts.append(a) or types.SimpleNamespace(pid=123))
    monkeypatch.setattr(sys,'argv',['perm_launch','--base',str(base),'--data',str(data),'--preflight',str(preflight),
        '--run',str(tmp_path/'run'),'--reward-candidate',str(first),'--reward-candidate',str(second),
        '--hourly-rate','1','--billing-quantum-seconds','60','--rental-started-at',__import__('datetime').datetime.fromtimestamp(time.time()-1,__import__('datetime').timezone.utc).isoformat(),
        '--preparation-cost-yuan','0','--shutdown-script-sha256',gdpo_platform.AUTODL_SHUTDOWN_SHA256])
    launch.main()
    manifest=json.loads((tmp_path/'run/launch.json').read_text())
    assert manifest['preparation']['completed'] is False
    assert manifest['preparation']['candidate_paths']==[str(first),str(second)]
    assert 'reward_inventory_sha256' not in manifest['identity']
    assert len(receipts)==1 and receipts[0][2].cap_yuan==20
    assert receipts[0][3] is None
    assert manifest['instance_shutdown_policy'] == 'preserve_user_allocation'


def test_metadata_only_base_identity_requires_verified_nonweight_inventory(tmp_path):
    from experiments.qwen3_bonus.models import SOURCES
    from experiments.qwen3_bonus.common import sha256
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'bonus_source.json').write_text(json.dumps({**SOURCES['base'],'files':{
        'config.json':sha256(tmp_path/'config.json'),'model.safetensors':'a'*64}}))
    assert len(launch.base_content_identity(tmp_path,allow_missing_raw_weights=True))==64
    with pytest.raises(ValueError):launch.base_content_identity(tmp_path)
    (tmp_path/'config.json').write_text('changed')
    with pytest.raises(ValueError):launch.base_content_identity(tmp_path,allow_missing_raw_weights=True)
