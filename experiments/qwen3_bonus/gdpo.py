"""Independent GDPO formula implementation with explicit task engineering gates.

Reference: Liu et al., arXiv:2601.05242, NVIDIA verl reference commit
4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115. No reference source is copied.
The NumPy kernel is CPU-only, testable without importing the training stack.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

KEYS=('preference','constraint','non_repetition')
WEIGHTS={'preference':.6,'constraint':.3,'non_repetition':.1}
TOLERANCES={'preference':1e-4,'constraint':1e-6,'non_repetition':.02}
REFERENCE_COMMIT='4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115'


@dataclass
class GDPOResult:
    advantages: np.ndarray
    returns: np.ndarray
    processed_rewards: dict
    metrics: dict
    group_records: list
    pre_gate_advantages: np.ndarray


def compute_gdpo(rewards: dict, group_ids, response_mask, invalid_reasons) -> GDPOResult:
    """Two candidates per UID; sample std then unbiased token-masked whitening.

    Validity overrides are applied once, after GDPO. They are reported separately.
    Near-ties are collapsed BEFORE standardization, never by adding noise.
    """
    if any(k not in rewards for k in KEYS):
        raise ValueError('GDPO requires all three reward components')
    raw={k:np.asarray(rewards[k],dtype=np.float64) for k in KEYS}
    n=len(group_ids)
    if any(v.shape!=(n,) or not np.isfinite(v).all() for v in raw.values()):
        raise ValueError('Reward components must be aligned finite vectors')
    mask=np.asarray(response_mask,dtype=np.float32)
    if mask.ndim!=2 or mask.shape[0]!=n or not np.isin(mask,[0,1]).all():
        raise ValueError('Response mask must be a binary [responses,tokens] matrix')
    if len(invalid_reasons)!=n or any(not isinstance(v,str) for v in invalid_reasons):
        raise ValueError('Invalid reasons must be aligned strings')
    groups={}
    for i,uid in enumerate(group_ids):
        if not isinstance(uid,(str,int,np.integer)):
            raise ValueError('Group IDs must be scalar strings/integers')
        groups.setdefault(str(uid),[]).append(i)
    if any(len(indices)!=2 for indices in groups.values()):
        raise ValueError('This recipe requires exactly two responses per group')
    processed={k:v.astype(np.float32) for k,v in raw.items()}
    sequence=np.zeros(n,np.float32);eligible=np.zeros(n,bool)
    active={k:0 for k in KEYS};records=[];mixed=all_invalid=conflicts=multi_active=0
    validity=[]
    for uid,idx in groups.items():
        bad=[bool(invalid_reasons[i]) for i in idx]
        record={'uid':uid,'indices':idx,'raw_rewards':{k:raw[k][idx].tolist() for k in KEYS},
                'invalid_reasons':[invalid_reasons[i] for i in idx], 'differences':{},'active':{}}
        directions=[]
        for k in KEYS:
            delta=abs(float(raw[k][idx[0]]-raw[k][idx[1]]))
            is_active=not any(bad) and delta>TOLERANCES[k]
            record['differences'][k]=delta;record['active'][k]=is_active
            if delta<=TOLERANCES[k]:processed[k][idx]=np.float32(raw[k][idx].mean())
            if is_active:
                active[k]+=1;directions.append(np.sign(raw[k][idx[0]]-raw[k][idx[1]]))
            if not any(bad):
                values=processed[k][idx]
                if is_active:
                    z=(values-values.mean(dtype=np.float32))/(values.std(ddof=1,dtype=np.float32)+np.float32(1e-6))
                    sequence[idx]+=np.float32(WEIGHTS[k])*z
        if len(directions)>=2:
            multi_active+=1;conflicts+=int(min(directions)!=max(directions))
        if all(bad):all_invalid+=1
        elif any(bad):mixed+=1
        else:eligible[idx]=True
        record['processed_rewards']={k:processed[k][idx].tolist() for k in KEYS}
        records.append(record);validity.append((idx,bad))
    token_values=sequence[:,None]*mask
    eligible_mask=mask*eligible[:,None]
    selected=token_values[eligible_mask.astype(bool)]
    before=np.zeros_like(mask,np.float32)
    if selected.size>=2:
        variance=selected.var(ddof=1,dtype=np.float32)
        if variance>0:
            before=(token_values-selected.mean(dtype=np.float32))/np.sqrt(variance+np.float32(1e-8))
            before=(before*eligible_mask).astype(np.float32)
    final=before.copy()
    for idx,bad in validity:
        if any(bad) and not all(bad):
            for i,is_bad in zip(idx,bad):final[i]=(-1. if is_bad else 1.)*mask[i]
    count=max(1,len(groups))
    metrics={f'reward/{k}_active_group_ratio':active[k]/count for k in KEYS}
    metrics.update({'reward/conflict_ratio':conflicts/multi_active if multi_active else 0.,
                    'reward/conflict_denominator':multi_active,
                    'validity/mixed_group_ratio':mixed/count,
                    'validity/all_invalid_group_ratio':all_invalid/count,
                    'reward/all_same_group_ratio':sum(not any(r['active'].values()) for r in records)/count})
    for label,values in [('pre_gate',before),('post_gate',final)]:
        valid=values[mask.astype(bool)]
        metrics[f'advantage/{label}_nonzero_ratio']=float(np.count_nonzero(valid)/valid.size) if valid.size else 0.
        metrics[f'advantage/{label}_std']=float(valid.std(dtype=np.float32)) if valid.size else 0.
    metrics['advantage/nonzero_ratio']=metrics['advantage/post_gate_nonzero_ratio']
    metrics['advantage/std']=metrics['advantage/post_gate_std']
    if not np.isfinite(final).all():raise ValueError('Non-finite GDPO advantage')
    return GDPOResult(final,final.copy(),processed,metrics,records,before)
