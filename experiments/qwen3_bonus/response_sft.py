"""Completion-only QLoRA supervised updates for psychological support."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import time

from experiments.qwen3_bonus.common import read_jsonl,write_json,sha256,output_path
from experiments.qwen3_bonus.training import completion_only_example


def main():
    import torch
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments,TrainerCallback,set_seed
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validation-data',type=Path,required=True)
    parser.add_argument('--initial-adapter',type=Path,required=True)
    args=parser.parse_args()
    args.output=output_path(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('Classifier QLoRA training needs the prepared Linux/CUDA environment')
    set_seed(3407)
    deadline=float(os.environ['BONUS_DEADLINE'])
    if time.time()>=deadline:
        raise RuntimeError('Budget expired before model load')
    tokenizer=AutoTokenizer.from_pretrained(str(args.base),local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(str(args.base),local_files_only=True,trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    model=PeftModel.from_pretrained(model,str(args.initial_adapter),is_trainable=True)
    raw=read_jsonl(args.data)
    items=[completion_only_example(tokenizer,row,1280) for row in raw]
    validation=[completion_only_example(tokenizer,row,1280) for row in read_jsonl(args.validation_data)]
    if not items:raise ValueError('Empty classifier training data')
    if any(not any(v!=-100 for v in row['labels']) for row in items):
        raise ValueError('Completion loss mask is empty')

    def collate(rows):
        length=max(len(r['input_ids']) for r in rows)
        result={'input_ids':[],'attention_mask':[],'labels':[]}
        for row in rows:
            pad=length-len(row['input_ids'])
            result['input_ids'].append(row['input_ids']+[tokenizer.pad_token_id]*pad)
            result['attention_mask'].append(row['attention_mask']+[0]*pad)
            result['labels'].append(row['labels']+[-100]*pad)
        return {k:torch.tensor(v) for k,v in result.items()}

    class BudgetCallback(TrainerCallback):
        stop_reason=None
        last_step=None
        longest_step=0
        def on_step_end(self,training_args,state,control,**kwargs):
            now=time.time()
            if self.last_step is not None:self.longest_step=max(self.longest_step,now-self.last_step)
            self.last_step=now
            if now+self.longest_step>=deadline:
                self.stop_reason='budget_deadline'
                control.should_training_stop=True;control.should_save=True
            return control
        def on_log(self,training_args,state,control,logs=None,**kwargs):
            safe={key:(float(value) if math.isfinite(value) else None) for key,value in (logs or {}).items()
                  if isinstance(value,(int,float))}
            with (args.output/'loss_metrics.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'step':int(state.global_step),'metrics':safe},allow_nan=False)+'\n')
            from experiments.qwen3_bonus.plotting import record_scalars
            record_scalars(args.output,int(state.global_step),safe)
            if any(isinstance(v,(int,float)) and not math.isfinite(v) for v in (logs or {}).values()):
                self.stop_reason='nonfinite_metrics';control.should_training_stop=True
            return control
        def on_save(self,training_args,state,control,**kwargs):
            from experiments.qwen3_bonus.snapshots import save_snapshot
            adapter=Path(training_args.output_dir)/f'checkpoint-{state.global_step}'
            save_snapshot(adapter,args.output/'adapter_snapshots',int(state.global_step))
            from experiments.qwen3_bonus.plotting import safe_render
            safe_render(args.output)
            return control

    config=TrainingArguments(output_dir=str(args.output/'checkpoints'),num_train_epochs=3,
        per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=5e-5,
        gradient_checkpointing=True,bf16=True,optim='paged_adamw_8bit',
        save_strategy='steps',save_steps=14,save_total_limit=2,logging_steps=1,
        eval_strategy='steps',eval_steps=14,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,
        prediction_loss_only=True,
        report_to=[],seed=3407,remove_unused_columns=False,logging_nan_inf_filter=False)
    callback=BudgetCallback()
    trainer=Trainer(model=model,args=config,train_dataset=items,eval_dataset=validation,data_collator=collate,
        processing_class=tokenizer,callbacks=[callback])
    initial_validation=trainer.evaluate()
    trainer.train()
    final_validation=trainer.evaluate()
    if callback.stop_reason=='nonfinite_metrics':
        write_json(args.output/'training_result.json',{'status':'failed','trained':False,
            'stop_reason':callback.stop_reason,'steps':trainer.state.global_step})
        raise RuntimeError('Non-finite classifier loss; candidate is not deployable')
    adapter=args.output/'adapter'
    model.save_pretrained(adapter,safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    from experiments.qwen3_bonus.plotting import safe_render
    safe_render(args.output)
    write_json(args.output/'training_result.json',{'status':'stopped' if callback.stop_reason else 'completed',
        'stop_reason':callback.stop_reason,'trained':trainer.state.global_step>0,
        'task':'psychological_support_response','adapter':str(adapter),'steps':trainer.state.global_step,
        'samples':len(items),'data_sha256':sha256(args.data),'completion_mask_verified':True,
        'precision':'NF4 QLoRA','seed':3407,'initial_validation':initial_validation,'final_validation':final_validation,
        'initial_adapter':str(args.initial_adapter),'best_checkpoint':trainer.state.best_model_checkpoint,'log_history':trainer.state.log_history,
        'base_revision':json.loads((args.base/'bonus_source.json').read_text())['revision'],
        'base_inventory_sha256':sha256(args.base/'bonus_source.json'),
        'clinical_validated':False})


if __name__=='__main__':main()
