"""Real-log plots and TensorBoard scalars; no synthetic metric fallback."""
from __future__ import annotations
import argparse
from functools import lru_cache
import json,math
from pathlib import Path
from experiments.qwen3_bonus.common import output_path,write_json,read_jsonl


def series(rows,key):
    points=[(row['step'],row.get('metrics',{}).get(key)) for row in rows]
    points=[(x,y) for x,y in points if isinstance(y,(int,float)) and math.isfinite(y)]
    return ([x for x,y in points],[y for x,y in points])


@lru_cache(maxsize=2)
def writer(path):
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(str(path))


def record_scalars(run,step,metrics):
    logger=writer(Path(run)/'tensorboard')
    for key,value in metrics.items():
        if isinstance(value,(int,float)) and math.isfinite(value):logger.add_scalar(key,value,step)
    logger.flush()


def render_run(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=output_path(run)
    source=next((run/name for name in ('steps.jsonl','loss_metrics.jsonl') if (run/name).is_file()),None)
    if source is None:return []
    rows=read_jsonl(source)
    if not rows:return []
    directory=run/'figures';directory.mkdir(exist_ok=True)
    metrics=(('actor/pg_loss','Policy loss'),('reward_mean','Mean reward'),('reward_group_std','Within-group reward std'),
        ('actor/kl_loss','KL estimate'),('response_length/mean','Response tokens'),('response_length/clip_ratio','Truncation fraction'))
    if source.name=='loss_metrics.jsonl':
        metrics=(('loss','Training loss'),('eval_loss','Held-out validation loss'),('learning_rate','Learning rate'),('grad_norm','Gradient norm'))
    fig,axes=plt.subplots(2,2 if source.name=='loss_metrics.jsonl' else 3,figsize=(13,7),constrained_layout=True)
    for axis,(key,title) in zip(axes.flat,metrics):
        x,y=series(rows,key)
        if x:axis.plot(x,y,marker='o',markersize=3,linewidth=1.5)
        else:axis.text(.5,.5,'Not logged',ha='center',va='center',transform=axis.transAxes)
        axis.set(title=title,xlabel='Optimization step');axis.grid(alpha=.25)
    for axis in list(axes.flat)[len(metrics):]:axis.set_visible(False)
    fig.suptitle(f'{run.name}: actual training metrics')
    outputs=[]
    for suffix in ('png','svg'):
        file=directory/f'training-curves.{suffix}';fig.savefig(file,dpi=160);outputs.append(str(file))
    plt.close(fig)
    rollouts=run/'rollouts.jsonl'
    if rollouts.exists():
        samples=read_jsonl(rollouts)
        if samples:
            fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
            lengths=[row['response_tokens'] for row in samples];rewards=[row['reward'] for row in samples]
            axes[0].hist(lengths,bins=min(12,len(samples)));axes[0].set(title='Generated length distribution',xlabel='Tokens',ylabel='Samples')
            axes[1].scatter(lengths,rewards,alpha=.6);axes[1].set(title='Reward vs. length',xlabel='Tokens',ylabel='Reward')
            for suffix in ('png','svg'):
                file=directory/f'reward-length.{suffix}';fig.savefig(file,dpi=160);outputs.append(str(file))
            plt.close(fig)
    write_json(directory/'manifest.json',{'source':str(source),'logged_steps':len(rows),'outputs':outputs,'data':'Actual logged training only'})
    return outputs


def safe_render(run):
    try:return render_run(run)
    except Exception as exc:
        write_json(Path(run)/'plotting-error.json',{'error_type':type(exc).__name__,'message':str(exc)})
        return []


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(render_run(args.run),indent=2))

if __name__=='__main__':main()
