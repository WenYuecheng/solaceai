"""Immutable reuse of the reviewed GDPO freeze with explicit PERM metadata."""
import argparse
import io
import json
from pathlib import Path
from experiments.qwen3_bonus.common import output_path,sha256,read_jsonl
from experiments.qwen3_bonus.gdpo_data import immutable_write,_json
from experiments.qwen3_bonus.gdpo_rewards import constraint_score
from experiments.qwen3_bonus.perm_rewards import REWARD_SPEC
DIMENSIONS=('emotional_understanding','empathic_expression','needs_satisfaction','overall_quality')


def metadata(row):
    constraint_score('',row['constraints'])
    return {'reward_spec':REWARD_SPEC,'checks':[c for c in row['constraints'] if c['kind']!='max_chars'],
        'length_checks':[c for c in row['constraints'] if c['kind']=='max_chars'],
        'semantic_requirements':{'emotional_understanding':['理解材料中有依据的处境与感受，不臆测动机'], 'empathic_expression':['用自然、有分寸且不淡化感受的语言回应'], 'needs_satisfaction':row['requirements'], 'overall_quality':['贴合当前任务和最新偏好，忠实于材料与能力边界，清楚连贯']},
        'check_interpretation':'Declarative surface checks only; do not measure semantic completion.'}


def task_delivery_ids(pairs):
    candidates=[r for r in pairs if r.get('category') in ('context','delivery') and r.get('requirements')]
    if len(candidates)<8:raise ValueError('Existing reviewed pairs cannot cover eight task_delivery conflicts')
    return [r['id'] for r in candidates[:8]]


def parquet_bytes(rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    msg=pa.struct([('role',pa.string()),('content',pa.string())])
    rule=pa.struct([('kind',pa.string()),('count',pa.int64()),('value',pa.int64()),('text',pa.string()),('ids',pa.list_(pa.string()))])
    semantic=pa.struct([(d,pa.list_(pa.string())) for d in DIMENSIONS])
    info=pa.struct([('id',pa.string()),('dialogue_id',pa.string()),('category',pa.string()),('messages',pa.list_(msg)),('constraints',pa.list_(rule)),('reward_spec',pa.string()),('checks',pa.list_(rule)),('length_checks',pa.list_(rule)),('semantic_requirements',semantic),('check_interpretation',pa.string())])
    schema=pa.schema([('data_source',pa.string()),('prompt',pa.list_(msg)),('ability',pa.string()),('reward_model',pa.struct([('style',pa.string()),('ground_truth',pa.string())])),('extra_info',info)])
    projected=[dict(data_source='solace_perm',prompt=r['prompt'],ability=r['category'],reward_model={'style':'rule','ground_truth':''},extra_info={k:r[k] for k in ('id','dialogue_id','category','constraints')}|{'messages':r['prompt']}|metadata(r)) for r in rows]
    stream=io.BytesIO();pq.write_table(pa.Table.from_pylist(projected,schema=schema),stream,compression='snappy')
    assert pq.read_table(io.BytesIO(stream.getvalue())).schema==schema
    return stream.getvalue(),str(schema)


def prepare_dataset(old_dir,reviewed_sheet,output_dir):
    old_dir=Path(old_dir);reviewed_sheet=Path(reviewed_sheet);output_dir=output_path(output_dir)
    manifest_path=old_dir/'dataset_manifest.json';inherited=json.loads(manifest_path.read_text())
    for name,entry in inherited['files'].items():
        if sha256(old_dir/name)!=entry['sha256']:raise ValueError('Inherited data hash mismatch')
    sheet=json.loads(reviewed_sheet.read_text());pairs=sheet['pairs']
    if len(pairs)!=20 or len({r['id'] for r in pairs})!=20 or any(r.get('reviewed') is not True or r.get('expected') not in ('a','b') or not r.get('reviewer') for r in pairs):raise ValueError('Twenty reviewed model-blind pairs required')
    original=json.loads((old_dir/'calibration_pairs.json').read_text())
    for row,base in zip(pairs,original):
        if any(row.get(k)!=base.get(k) for k in ('id','prompt','a','b','expected','constraints','requirements')):raise ValueError('Reviewed sheet changes original pair or label')
    selected=task_delivery_ids(pairs);payloads={};schemas={};audit={};length_only=0
    for split,n in [('train',256),('dev',32),('test',80)]:
        path=old_dir/f'response_{split}.jsonl';rows=read_jsonl(path)
        if len(rows)!=n:raise ValueError('Unexpected inherited split size')
        for r in rows:metadata(r)
        if split=='train':length_only=sum(not metadata(r)['checks'] for r in rows)
        payloads[path.name]=path.read_bytes();payloads[f'reward_metadata_{split}.json']=_json({r['id']:metadata(r) for r in rows})
        payloads[f'response_{split}.parquet'],schemas[split]=parquet_bytes(rows)
        audit[split]={'rows':n,'prompt_id_dialogue_split_equal':True,'jsonl_byte_equal':True,'inherited_sha256':sha256(path)}
    if length_only!=192:raise ValueError('Expected 192 train samples with no applicable task check')
    payloads['calibration_pairs.json']=_json(sheet);payloads['task_delivery_ids.json']=_json(selected)
    manifest={'schema_version':1,'reward_spec':REWARD_SPEC,'frozen':True,'inherited_manifest_sha256':sha256(manifest_path),'inherited_files':inherited['files'],'reviewed_sheet_sha256':sha256(reviewed_sheet),'task_delivery_ids':selected,'subset_policy':'First eight context/delivery pairs in original model-blind order, semantic requirements required','audit':audit,'train_no_applicable_checks':length_only,'parquet_schema':schemas,'semantic_checks':False,'files':{k:{'sha256':__import__('hashlib').sha256(v).hexdigest(),'bytes':len(v)} for k,v in payloads.items()}}
    payloads['dataset_manifest.json']=_json(manifest)
    for k,v in payloads.items():
        path=output_path(output_dir/k)
        if path.exists() and path.read_bytes()!=v:raise FileExistsError(path)
    for k,v in payloads.items():immutable_write(output_path(output_dir/k),v)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old-dir',type=Path,required=True);p.add_argument('--reviewed-sheet',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare_dataset(a.old_dir,a.reviewed_sheet,a.output),indent=2))
