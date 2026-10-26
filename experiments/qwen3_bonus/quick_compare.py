"""GPU-only real predictions and held-out likelihood comparisons; no paid API."""
from __future__ import annotations
import argparse,contextlib,json,math,time
from pathlib import Path
from experiments.qwen3_bonus.common import ROOT,read_jsonl,write_json,sha256,output_path
from experiments.qwen3_bonus.training import completion_only_example
from experiments.qwen3_bonus.evaluation import classification_metrics,blind_pairs


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed
    from peft import PeftModel
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=output_path(args.run);out=run/'comparison';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);set_seed(3407)
    base=ROOT/'artifacts/models/base'
    tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side='left';tok.truncation_side='left'
    original=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    model=PeftModel.from_pretrained(original,str(ROOT/'artifacts/runs/emotion/adapter'),adapter_name='emotion')
    model.load_adapter(str(ROOT/'artifacts/runs/response-weste3/adapter_snapshots/step-000032'),adapter_name='grpo32')
    model.load_adapter(str(run/'adapter'),adapter_name='support')
    model.eval();model.config.use_cache=True
    model.generation_config.temperature=None;model.generation_config.top_p=None;model.generation_config.top_k=None
    def activate(name):
        if name=='base':return model.disable_adapter()
        model.set_adapter(name);return contextlib.nullcontext()
    def example(row,task):
        text=row['expected'] if task=='classification' else tok.decode(tok.encode(row['reference'],add_special_tokens=False)[:500],skip_special_tokens=True)
        return completion_only_example(tok,{'prompt':row['prompt'],'completion':[{'role':'assistant','content':text}]},1280)
    def likelihood(rows,task):
        total=0.;count=0
        for start in range(0,len(rows),4):
            items=[example(row,task) for row in rows[start:start+4]];length=max(len(x['input_ids']) for x in items)
            batch={k:[] for k in ['input_ids','attention_mask','labels']}
            for item in items:
                n=length-len(item['input_ids'])
                batch['input_ids'].append(item['input_ids']+[tok.pad_token_id]*n)
                batch['attention_mask'].append(item['attention_mask']+[0]*n)
                batch['labels'].append(item['labels']+[-100]*n)
            batch={k:torch.tensor(v,device='cuda') for k,v in batch.items()}
            n=int((batch['labels'][:,1:]!=-100).sum())
            with torch.inference_mode():loss=float(model(**batch).loss)
            if not math.isfinite(loss):raise ValueError('Nonfinite evaluation loss')
            total+=loss*n;count+=n
        return total/count
    def generate(rows,task):
        result=[];maximum=16 if task=='classification' else 512
        for start in range(0,len(rows),4):
            batch=rows[start:start+4]
            texts=[tok.apply_chat_template(r['prompt'],tokenize=False,add_generation_prompt=True) for r in batch]
            inputs=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=768,add_special_tokens=False).to('cuda')
            before=time.monotonic()
            with torch.inference_mode():values=model.generate(**inputs,max_new_tokens=maximum,do_sample=False,pad_token_id=tok.pad_token_id)
            elapsed=time.monotonic()-before
            for row,value in zip(batch,values[:,inputs['input_ids'].shape[1]:]):
                answer=tok.decode(value,skip_special_tokens=True).strip()
                tokens=int((value!=tok.pad_token_id).sum())
                record={'id':row['id'],'prompt':row['prompt'],'answer':answer,'generated_tokens':tokens,
                    'at_output_limit':tokens>=maximum,'batch_seconds':elapsed,'batch_size':len(batch)}
                if task=='classification':record.update(expected=row['expected'],predicted=answer)
                result.append(record)
            print(task,start+len(batch),'/',len(rows),flush=True)
        return result
    classification=read_jsonl(ROOT/'artifacts/data/classification_eval.jsonl')
    responses=read_jsonl(ROOT/'artifacts/data/response_eval.jsonl')
    report={'classification':{},'response':{},'classification_scope':'400 synthetic regression cases',
        'response_scope':'40 frozen official-test dialogue response points; no human review yet',
        'precision':'NF4 same base for every variant','paid_api_calls':0,'production_enabled':False,
        'base_revision':'cdbee75f17c01a7cc42f958dc650907174af0554',
        'datasets':{n:sha256(ROOT/f'artifacts/data/{n}_eval.jsonl') for n in ['classification','response']}}
    captures={}
    for name in ['base','emotion']:
        with activate(name):
            loss=likelihood(classification,'classification');rows=generate(classification,'classification')
        metrics=classification_metrics(rows);metrics['reference_token_nll']=loss
        report['classification'][name]=metrics
        write_json(out/f'classification-{name}.json',{'rows':rows,'metrics':metrics})
        write_json(out/'summary.json',report)
    for name in ['base','grpo32','support']:
        with activate(name):
            loss=likelihood(responses,'response');rows=generate(responses,'response')
        captures[name]=rows
        report['response'][name]={'reference_token_nll':loss,'count':len(rows),
            'output_limit_fraction':sum(r['at_output_limit'] for r in rows)/len(rows),
            'mean_generated_tokens':sum(r['generated_tokens'] for r in rows)/len(rows)}
        write_json(out/f'response-{name}.json',{'rows':rows,'metrics':report['response'][name]})
        write_json(out/'summary.json',report)
    sheet,key=blind_pairs(captures['base'],captures['support'])
    write_json(out/'blind-pairs.json',{'review_type':'pending','pairs':sheet})
    write_json(out/'blind-key.json',key)
    initial=json.loads((run/'training_result.json').read_text())
    report['sft_dev']={'before':initial['initial_validation']['eval_loss'],'after':initial['final_validation']['eval_loss']}
    a=report['response']['base']['reference_token_nll'];b=report['response']['support']['reference_token_nll']
    report['reference_nll_relative_reduction']=(a-b)/a
    write_json(out/'summary.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    axes[0].bar(['Base','Emotion LoRA'],[report['classification'][k]['macro_f1'] for k in ['base','emotion']]);axes[0].set(title='Classification Macro-F1',ylim=(0,1.05))
    axes[1].bar(['Base','Emotion LoRA'],[report['classification'][k]['reference_token_nll'] for k in ['base','emotion']]);axes[1].set(title='Classification answer NLL (lower is better)')
    axes[2].bar(['Base','GRPO32','GRPO + SFT'],[report['response'][k]['reference_token_nll'] for k in ['base','grpo32','support']]);axes[2].set(title='40-case reference NLL (lower is better)')
    for axis in axes:
        axis.grid(axis='y',alpha=.25)
        for bar in axis.patches:axis.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{bar.get_height():.4f}',ha='center',va='bottom',fontsize=9)
    for suffix in ['png','svg']:fig.savefig(out/f'comparison.{suffix}',dpi=160)
    plt.close(fig)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
