"""Scoped driver-only dispatch for the pinned, critic-less verl 0.5.0 trainer."""
from __future__ import annotations
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
from experiments.qwen3_bonus.gdpo import KEYS,compute_gdpo


class GDPOAdapter:
    def __init__(self,config,on_result=None):
        if config.get('bonus_adv_estimator')!='gdpo':raise ValueError('GDPO must be explicitly enabled')
        self.config=config;self.on_result=on_result;self.calls=0;self.last_result=None

    def __call__(self,data,*args,**kwargs):
        import torch
        info=data.non_tensor_batch
        required=(*KEYS,'uid','invalid_reason')
        if any(k not in info for k in required):raise ValueError('Missing GDPO reward component or grouping field')
        if 'response_mask' not in data.batch:
            length=data.batch['responses'].shape[-1]
            data.batch['response_mask']=data.batch['attention_mask'][:,-length:]
        mask=data.batch['response_mask']
        result=compute_gdpo({k:info[k] for k in KEYS},info['uid'],
                            mask.detach().cpu().numpy(),list(info['invalid_reason']))
        data.batch['advantages']=torch.as_tensor(result.advantages,device=mask.device,dtype=torch.float32)
        data.batch['returns']=torch.as_tensor(result.returns,device=mask.device,dtype=torch.float32)
        self.calls+=1;self.last_result=result
        if self.on_result is not None:self.on_result(data,result)
        return data

    def self_test(self,module):
        import torch
        from verl import DataProto
        info={'uid':np.array(['x','x','y','y']), 'preference':np.array([.9,.1,.9,.1]),
              'constraint':np.array([1,0,0,1]),'non_repetition':np.ones(4),
              'invalid_reason':np.array(['']*4)}
        data=DataProto.from_dict(tensors={'response_mask':torch.ones(4,2)},non_tensors=info)
        callback=self.on_result;self.on_result=None
        before=self.calls
        try:
            module.compute_advantage(data,config=self.config)
            if self.calls!=before+1 or torch.equal(data.batch['advantages'][0],data.batch['advantages'][2]):
                raise RuntimeError('GDPO dispatch did not execute distinct component normalization')
        finally:
            self.on_result=callback;self.calls=before;self.last_result=None


@contextmanager
def install_gdpo(module,config,on_result=None):
    original=module.compute_advantage
    adapter=GDPOAdapter(config,on_result)
    module.compute_advantage=adapter
    try:yield adapter
    finally:module.compute_advantage=original
