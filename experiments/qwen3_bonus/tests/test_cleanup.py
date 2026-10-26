from pathlib import Path
import pytest


def test_cleanup_refuses_external_path_and_symlink(tmp_path):
    from experiments.qwen3_bonus.cleanup import checked_target
    root=tmp_path/'project';root.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    (root/'linked').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError):checked_target(root,'../outside')
    with pytest.raises(ValueError):checked_target(root,'linked')
    with pytest.raises(ValueError):checked_target(root,'.')


def test_plan_keeps_runtime_and_evidence_but_removes_download_cache(tmp_path):
    from experiments.qwen3_bonus.cleanup import plan
    root=tmp_path
    for name in ['models/bge-m3/pytorch_model.bin','archive/audits/review.md',
                 'experiments/qwen3_bonus/artifacts/remote/wheel-relay/linux.whl',
                 'archive/releases/old-app.tar.gz','archive/training/unique-training.tar.gz']:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
    targets={row['path'] for row in plan(root)}
    assert 'experiments/qwen3_bonus/artifacts/remote/wheel-relay' in targets
    assert 'archive/releases/old-app.tar.gz' in targets
    assert 'models/bge-m3/pytorch_model.bin' not in targets
    assert 'archive/audits/review.md' not in targets
    assert 'archive/training/unique-training.tar.gz' not in targets


def test_old_model_download_only_removed_when_ollama_copy_is_identical(tmp_path):
    import hashlib,json
    from experiments.qwen3_bonus.cleanup import duplicate_ollama_download
    raw=tmp_path/'project/old.gguf';raw.parent.mkdir();raw.write_bytes(b'real-test-content')
    store=tmp_path/'ollama';(store/'blobs').mkdir(parents=True)
    manifest=store/'manifests/registry.ollama.ai/library/solaceai-qwen2.5-7b-ft/latest'
    manifest.parent.mkdir(parents=True)
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({'layers':[{'mediaType':'application/vnd.ollama.image.model','digest':'sha256:'+digest}]}))
    blob=store/'blobs'/('sha256-'+digest)
    blob.write_bytes(b'different-content')
    assert not duplicate_ollama_download(raw,store)
    blob.write_bytes(raw.read_bytes())
    assert duplicate_ollama_download(raw,store)
