import pytest
from experiments.qwen3_bonus import perm_data as p

def test_metadata_length_excluded_and_subset_semantic():
    row={'category':'delivery','requirements':['交付草稿'],'constraints':[{'kind':'max_chars','value':8}]}
    m=p.metadata(row)
    assert set(m['semantic_requirements'])=={'emotional_understanding','empathic_expression','needs_satisfaction','overall_quality'}
    assert m['checks']==[] and m['semantic_requirements']['needs_satisfaction']==['交付草稿']
    pairs=[{'id':str(i),'category':'delivery' if i<8 else 'listening','requirements':['deliver']} for i in range(20)]
    assert p.task_delivery_ids(pairs)==list(map(str,range(8)))
    with pytest.raises(ValueError):p.task_delivery_ids(pairs[:7])

def test_real_frozen_data_equality_and_schema():
    import json
    from pathlib import Path
    import pyarrow.parquet as pq
    root=Path(p.__file__).parent/'artifacts'
    new=root/'perm-grpo-20261024/data';old=root/'gdpo-20261024/data'
    manifest=json.loads((new/'dataset_manifest.json').read_text())
    assert manifest['train_no_applicable_checks']==192
    for split in ('train','dev','test'):
        assert (new/f'response_{split}.jsonl').read_bytes()==(old/f'response_{split}.jsonl').read_bytes()
        rows=pq.read_table(new/f'response_{split}.parquet').to_pylist()
        assert all(r['extra_info']['messages']==r['prompt'] for r in rows)
