import io
import json
import tarfile
from pathlib import Path
import pytest
from experiments.qwen3_bonus import perm_remote as r


def test_bundle_small_manifest_and_local_evidence(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'a.py').write_text('pass')
    data=root/'artifacts/perm-grpo-20261024/data';data.mkdir(parents=True);(data/'freeze.json').write_text('{}')
    evidence=root/'artifacts/perm-grpo-20261024/calibration';evidence.mkdir();(evidence/'cpu.json').write_text('{}')
    (root/'secret.env').write_text('secret')
    receipt=r.build_bundle(tmp_path/'bundle.tgz',experiment_root=root)
    with tarfile.open(receipt['archive']) as t:
        names=t.getnames();manifest=json.load(t.extractfile('DEPLOYMENT_MANIFEST.json'))
    assert set(names)==set(manifest)|{'DEPLOYMENT_MANIFEST.json'}
    assert any('local-evidence' in n for n in names)
    assert not any('secret.env' in n for n in names)
    with pytest.raises(FileExistsError):r.build_bundle(tmp_path/'bundle.tgz',experiment_root=root)
    r.safe_extract(tmp_path/'bundle.tgz',tmp_path/'out')


def archive(tmp_path, members, manifest):
    p=tmp_path/'bad.tgz'
    with tarfile.open(p,'w:gz') as t:
        for name,payload in members+[('DEPLOYMENT_MANIFEST.json',json.dumps(manifest).encode())]:
            info=tarfile.TarInfo(name);info.size=len(payload);t.addfile(info,io.BytesIO(payload))
    return p

@pytest.mark.parametrize('members,manifest',[
    ([('../escape',b'x')],{}),([('a',b'x'),('a',b'x')],{}),([('a',b'x')],{}),
    ([('a',b'x')],{'a':'0'*64}),([],{'missing':'0'*64}),
    ([('DEPLOYMENT_MANIFEST.json',b'{}')],{}),
])
def test_reject_unsafe_before_writing(tmp_path,members,manifest):
    with pytest.raises(ValueError):r.safe_extract(archive(tmp_path,members,manifest),tmp_path/'out')
    assert not (tmp_path/'out/a').exists()


def test_symlink_ancestor_rejected(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'a.py').write_text('pass')
    bundle=tmp_path/'ok.tgz';r.build_bundle(bundle,experiment_root=root)
    out=tmp_path/'out';out.mkdir();(out/'experiments').symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError):r.safe_extract(bundle,out)


def test_bundle_limits_and_symlinks(tmp_path,monkeypatch):
    root=tmp_path/'source';root.mkdir();(root/'a.py').write_text('1234')
    monkeypatch.setattr(r,'MAX_FILE',3)
    with pytest.raises(ValueError):r.build_bundle(tmp_path/'bundle.tgz',experiment_root=root)


def test_deploy_uses_checked_args_without_launch(tmp_path,monkeypatch):
    root=tmp_path/'source';root.mkdir();(root/'a.py').write_text('pass')
    bundle=tmp_path/'ok.tgz';r.build_bundle(bundle,experiment_root=root)
    calls=[]
    def run(args,**kw):
        calls.append((args,kw));return type('Result',(),{'stdout':'{"gpu_training_started":false}'})()
    monkeypatch.setattr(r.subprocess,'run',run)
    assert r.deploy(bundle,'example.com',22)['gpu_training_started'] is False
    assert len(calls)==3 and all(kw['check'] for _,kw in calls)
    assert all(not kw.get('shell') for _,kw in calls)
    assert r.REMOTE_ROOT=='/root/solace-perm-20261024'
    with pytest.raises(ValueError):r.deploy(bundle,'x;evil',22)


def test_collection_policy(tmp_path):
    for name,allowed in [('logs/train.json',True),('adapter_snapshots/step/adapter_model.safetensors',True),('models/model.safetensors',False),('checkpoints/x.json',False),('logs/secrets.json',False),('elsewhere/adapter_model.safetensors',False)]:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        assert r.collection_allowed(p,tmp_path)==allowed


def test_archive_nonregular_rejected(tmp_path):
    p=tmp_path/'link.tgz'
    with tarfile.open(p,'w:gz') as t:
        info=tarfile.TarInfo('link');info.type=tarfile.SYMTYPE;info.linkname='/tmp';t.addfile(info)
    with pytest.raises(ValueError):r.safe_extract(p,tmp_path/'out')


def test_collect_limits_total_before_transfer(tmp_path,monkeypatch):
    calls=[]
    def run(args,**kw):calls.append((args,kw))
    monkeypatch.setattr(r.subprocess,'run',run)
    receipt=r.collect('example.com',22,tmp_path/'result.tgz')
    assert receipt['sha256']
    assert '500*1024*1024' in calls[0][0][-1]
    assert calls[0][1]['check'] is True
