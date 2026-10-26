"""Offline four-axis PERM review. No model generation or test-set selection.

CLI: freeze, sheets, select, report, plot, dev-curve. Each source requires its
independently recorded --NAME-sha256. Review JSON uses the public sheet digest,
review_type=model, reviewer, provenance, and rows with id/choice/rationale and
a/b rubrics containing semantic_completed, severe and the four AXES (1..5).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from experiments.qwen3_bonus import gdpo_evaluation as shared
from experiments.qwen3_bonus.common import ROOT, output_path, sha256

VARIANTS = ('base', 'grpo32', 'current', 'perm_grpo')
CONTRASTS = ('current', 'base', 'grpo32')
AXES = ('emotional_understanding', 'empathic_expression', 'needs_satisfaction', 'overall_quality')
digest_json = shared.digest_json
read_verified = shared.read_verified
write_immutable = shared.write_immutable
paired_dialogue_bootstrap = shared.paired_dialogue_bootstrap
checkpoint_digest = shared.checkpoint_digest


def code_checksums():
    return {name: sha256(ROOT / name) for name in ('perm_evaluation.py', 'gdpo_evaluation.py')}


def protocol_document():
    return {'protocol': 'perm-four-axis-model-review-v1', 'actual_algorithm': 'perm_grpo',
      'variants': list(VARIANTS), 'seed': 3407, 'swapped_recheck_fraction': .2,
      'final_unique_cases': 80, 'main_contrast': 'perm_grpo vs current',
      'rubric': {'semantic_completed': 'Required boolean: all user requirements satisfied semantically.',
         'severe': {'categories': list(shared.SEVERE_CATEGORIES), 'reason': 'Required for each flag.'},
         **{axis: 'Required finite numeric 1..5; 1 poor, 3 adequate, 5 excellent.' for axis in AXES}},
      'selection': 'Dev only: severe count min, semantic completion max, arithmetic four-axis mean max, step min.',
      'interpretation': 'Model review with actual reviewer/provenance; not human or clinical validation.',
      'generation': 'Same frozen quantized base and chat template, temperature 0, max_tokens 256; no generation here.'}


def _core(bundle):
    return {k: v for k, v in bundle.items() if k not in ('actual_algorithm', 'protocol_sha256', 'perm_bundle_sha256')}


def freeze_captures(rows, captures, *, split):
    if (not captures or any(not shared._hash(r.get('chat_template_sha256')) or r.get('quantization') != '4bit' for r in captures)
            or len({r['chat_template_sha256'] for r in captures}) != 1):
        raise ValueError('All captures require one identical chat template SHA256 and 4bit quantization')
    bundle = shared.freeze_captures(rows, captures, split=split, variants=VARIANTS)
    # The capture hash already binds exact prompts and any recorded template metadata.
    bundle.update(actual_algorithm='perm_grpo', protocol_sha256=digest_json(protocol_document()))
    bundle['perm_bundle_sha256'] = digest_json(bundle)
    return bundle


def _validate_bundle(bundle):
    if freeze_captures(bundle['cases'], bundle['captures'], split=bundle['split']) != bundle:
        raise ValueError('Frozen PERM bundle or protocol changed')


def make_pair_sheets(bundle, seed=3407, recheck_fraction=.2):
    _validate_bundle(bundle)
    if seed != 3407:
        raise ValueError('Frozen PERM protocol requires seed 3407')
    sheet, key = shared.make_pair_sheets(_core(bundle), seed, recheck_fraction,
                     variants=VARIANTS, target='perm_grpo', contrasts=CONTRASTS)
    sheet['protocol'] = 'perm-four-axis-model-review-v1'
    sheet['rubric'] = protocol_document()['rubric']
    key['sheet_sha256'] = digest_json(sheet)
    key['bundle_sha256'] = bundle['perm_bundle_sha256']
    return sheet, key


def _mean_axes(row):
    if any(not shared._number(row.get(axis)) or not 1 <= row[axis] <= 5 for axis in AXES):
        raise ValueError('All four PERM dimensions require finite numeric scores 1..5')
    return sum(row[axis] for axis in AXES) / 4


def _project_reviews(reviews):
    projected = copy.deepcopy(reviews)
    for row in projected['rows']:
        for side in ('a', 'b'):
            row[side]['quality'] = _mean_axes(row[side])
    return projected


def validate_reviews(sheet, reviews):
    shared.validate_reviews(sheet, _project_reviews(reviews))
    return reviews


def build_dev_candidate(bundle, sheet, key, reviews, checkpoint, step, dev_manifest_path=None):
    """Derive all metrics from authenticated complete dev records; embed replayable inputs."""
    if bundle.get('split') != 'dev' or len(bundle.get('cases', [])) != 32 or len({r['id'] for r in bundle['cases']}) != 32:
        raise ValueError('Exactly 32 frozen unique dev cases required')
    manifest_path=Path(dev_manifest_path or ROOT/'artifacts/perm-grpo-20261024/data/dataset_manifest.json').resolve()
    manifest=json.loads(manifest_path.read_text())
    dev_path=manifest_path.parent/'response_dev.jsonl'
    if (manifest.get('frozen') is not True
            or manifest.get('files',{}).get('response_dev.jsonl',{}).get('sha256') != sha256(dev_path)):
        raise ValueError('Frozen dev manifest and actual response_dev.jsonl digest required')
    canonical=[json.loads(line) for line in dev_path.read_text().splitlines() if line.strip()]
    if len(canonical)!=32 or len({r['id'] for r in canonical})!=32:
        raise ValueError('Canonical dev split requires exactly 32 unique cases')
    if {r['id']:r for r in canonical}!={r['id']:r for r in bundle['cases']}:
        raise ValueError('Dev bundle differs from exact canonical frozen dev case content')
    evidence={'bundle':bundle,'sheet':sheet,'key':key,'reviews':reviews,
              'dev_manifest_path':str(manifest_path)}
    report = evaluate_reviews(bundle, sheet, key, reviews)
    expected = {r['id'] for r in bundle['cases']}
    scores = {identity: [] for identity in expected}
    for row in report['per_item_scoring']:
        for side in ('a','b'):
            if row[side] == 'perm_grpo':
                scores[row['case_id']].append(row['scores'][side])
    if any(len(rows) != 3 for rows in scores.values()):
        raise ValueError('All three comparisons required for every dev case')
    digest = checkpoint_digest(Path(checkpoint))
    if any(r['adapter_sha256'] != digest for r in bundle['captures'] if r['variant']=='perm_grpo'):
        raise ValueError('Dev captures differ from candidate checkpoint')
    return {'split':'dev','step':step,'checkpoint':str(checkpoint),'checkpoint_sha256':digest,
        'semantic_complete':sum(all(r['semantic_completed'] for r in rows) for rows in scores.values()),
        'severe_count':sum(any(r['severe'] for r in rows) for rows in scores.values()),
        **{axis:report['answer_summaries']['perm_grpo'][axis] for axis in AXES},
        'dev_manifest_path':str(manifest_path),'dev_manifest_sha256':sha256(manifest_path),
        'dev_data_sha256':sha256(dev_path),
        'evidence':evidence,'evidence_sha256':digest_json(evidence),
        'report_sha256':digest_json(report)}


def select_checkpoint(candidates):
    for candidate in candidates:
        evidence = candidate.get('evidence')
        if not isinstance(evidence,dict):
            raise ValueError('Authenticated complete dev review evidence required')
        actual = build_dev_candidate(**evidence, checkpoint=candidate['checkpoint'], step=candidate['step'])
        if actual != candidate:
            raise ValueError('Dev summary differs from its complete review evidence')
    projected = [{**r, 'quality': _mean_axes(r)} for r in candidates]
    return shared.select_checkpoint(projected)


def freeze_selection(candidates, destination):
    selected = select_checkpoint(candidates)
    if checkpoint_digest(Path(selected['checkpoint'])) != selected.get('checkpoint_sha256'):
        raise ValueError('Selected checkpoint SHA256 differs from dev candidate')
    result = {'selected': selected, 'actual_algorithm': 'perm_grpo',
      'dev_candidates_sha256': digest_json(candidates), 'protocol_sha256': digest_json(protocol_document()),
      'rule': ['severe_count:min', 'semantic_complete:max', 'four_axis_mean:max', 'step:min'],
      'code_sha256': code_checksums()}
    result['selection_sha256'] = digest_json(result)
    write_immutable(destination, result)
    return result


def verify_selection(path):
    result = shared.verify_selection(path)
    if (result.get('actual_algorithm') != 'perm_grpo'
            or result.get('protocol_sha256') != digest_json(protocol_document())
            or result['selected'].get('quality') != _mean_axes(result['selected'])):
        raise ValueError('Selection does not bind PERM four-axis protocol')
    candidate = {k:v for k,v in result['selected'].items() if k != 'quality'}
    select_checkpoint([candidate])
    return result


def verify_test_selection(bundle, path):
    _validate_bundle(bundle)
    selection = verify_selection(path)
    if bundle['split'] != 'test':
        raise ValueError('Final report requires frozen test split')
    if any(r['adapter_sha256'] != selection['selected']['checkpoint_sha256']
           for r in bundle['captures'] if r['variant'] == 'perm_grpo'):
        raise ValueError('PERM test capture differs from frozen dev-selected adapter')
    return selection


def evaluate_reviews(bundle, sheet, key, reviews):
    _validate_bundle(bundle)
    if (sheet, key) != make_pair_sheets(bundle, key['seed'], key['recheck_fraction']):
        raise ValueError('Sheet or private assignment key changed')
    validate_reviews(sheet, reviews)
    core_sheet, core_key = shared.make_pair_sheets(_core(bundle), variants=VARIANTS, target='perm_grpo')
    projected = _project_reviews(reviews)
    projected['sheet_sha256'] = digest_json(core_sheet)
    result = shared.evaluate_reviews(_core(bundle), core_sheet, core_key, projected,
                                     variants=VARIANTS, target='perm_grpo')
    per_item, by_variant = [], {variant: {} for variant in VARIANTS}
    indexed = shared._index(reviews['rows'])
    for identity, assignment in key['pairs'].items():
        if assignment['recheck_of']:
            continue
        row = indexed[identity]
        per_item.append({'pair_id': identity, **assignment, 'choice': row['choice'],
                         'rationale': row['rationale'], 'scores': {s: copy.deepcopy(row[s]) for s in ('a', 'b')}})
        for side in ('a', 'b'):
            by_variant[assignment[side]].setdefault(assignment['case_id'], []).append(row[side])
    for variant, cases in by_variant.items():
        for axis in AXES:
            result['answer_summaries'][variant][axis] = sum(
                sum(r[axis] for r in rr)/len(rr) for rr in cases.values())/len(cases)
        result['answer_summaries'][variant]['four_axis_mean'] = sum(result['answer_summaries'][variant][a] for a in AXES)/4
    for contrast in result['contrasts'].values():
        contrast['preference_rate'] = contrast['score']
    result.update(actual_algorithm='perm_grpo', per_item_scoring=per_item,
       raw_reviews=copy.deepcopy(reviews), protocol=protocol_document(),
       bundle_sha256=bundle['perm_bundle_sha256'], sheet_sha256=digest_json(sheet),
       key_sha256=digest_json(key), reviews_sha256=digest_json(reviews), code_sha256=code_checksums())
    return result


TRAINING_PANELS = [
 ('reward_components', ('reward/preference_mean', 'reward/constraint_mean', 'reward/main_mean', 'reward/repeat_mean')),
 ('validity', ('reward/invalid_ratio', 'reward/group_gate_ratio', 'reward/constraint_applicable_ratio')),
 ('native_advantages', ('advantage/nonzero_ratio', 'advantage/std', 'actor/kl_loss', 'reward/group_abs_diff_mean', 'reward/group_std', 'reward/zero_group_ratio')),
 ('duration', ('timing_s/cpu_reward', 'timing_s/update_actor', 'timing_s/save_checkpoint', 'timing_s/step'))]


def plot_training_logs(logs, output, *, rollouts=None):
    """Reuse rendering with PERM's recorded native metrics; no inferred time axis."""
    return shared.plot_training_logs(logs, output, rollouts=rollouts, panels=TRAINING_PANELS)


def plot_dev_curve(snapshots, logs, output):
    """Join dev scores to exact recorded steps; never interpolate missing times/scores."""
    points, missing = [], []
    indexed = {}
    for row in logs:
        if type(row.get('step')) is not int or row['step'] in indexed:
            raise ValueError('Recorded steps require unique integer steps')
        indexed[row['step']] = row
    if not snapshots:
        missing.append('dev_snapshots')
    for row in snapshots:
        if row.get('split') != 'dev' or type(row.get('step')) is not int:
            raise ValueError('Quality curve accepts dev snapshots only')
        quality = _mean_axes(row)
        log = indexed.get(row['step'], {})
        elapsed = log.get('cumulative_train_seconds', log.get('elapsed_seconds'))
        if not shared._number(elapsed) or elapsed < 0:
            missing.append(f"recorded_time:step={row['step']}")
            continue
        points.append({'step': row['step'], 'elapsed_seconds': elapsed, 'four_axis_mean': quality})
    normalized = [{'step': p['step'], 'elapsed_seconds': p['elapsed_seconds'], 'dev/four_axis_mean': p['four_axis_mean']} for p in points]
    result = shared.plot_training_logs(normalized, output, panels=[('dev_quality_vs_time', ('dev/four_axis_mean',))])
    result.update(snapshots_sha256=digest_json(snapshots), steps_sha256=digest_json(logs), points=points, missing=missing)
    write_immutable(output_path(output)/'dev-curve.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    def source(command, name, **kwargs):
        command.add_argument('--'+name, type=Path, **kwargs)
        command.add_argument('--'+name+'-sha256', required=kwargs.get('required', False))
    command = sub.add_parser('freeze')
    source(command, 'data', required=True); source(command, 'captures', required=True)
    command.add_argument('--split', choices=['dev', 'test'], required=True)
    command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('sheets')
    source(command, 'bundle', required=True); command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('candidate')
    for name in ('bundle','sheet','key','reviews'): source(command,name,required=True)
    command.add_argument('--checkpoint',type=Path,required=True)
    command.add_argument('--step',type=int,required=True)
    command.add_argument('--output',type=Path,required=True)
    command = sub.add_parser('select')
    source(command, 'candidates', required=True); command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('report')
    for name in ('bundle', 'sheet', 'key', 'reviews', 'selection'): source(command, name, required=True)
    command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('plot')
    source(command, 'logs', required=True); source(command, 'rollouts')
    command.add_argument('--logs-jsonl', action='store_true')
    command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('dev-curve')
    source(command, 'snapshots', required=True); source(command, 'logs', required=True)
    command.add_argument('--logs-jsonl', action='store_true')
    command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output_path(args.output)
    def read(name, jsonl=False):
        return read_verified(getattr(args, name), getattr(args, name+'_sha256'), jsonl=jsonl)
    if args.action == 'freeze':
        write_immutable(args.output, freeze_captures(read('data', True), read('captures'), split=args.split))
    elif args.action == 'sheets':
        output = output_path(args.output)
        if output.exists(): raise FileExistsError('Choose a fresh sheet directory')
        sheet, key = make_pair_sheets(read('bundle'))
        write_immutable(output/'sheet.json', sheet)
        write_immutable(output/'key.private.json', key, private=True)
        write_immutable(output/'protocol.json', protocol_document())
    elif args.action == 'candidate':
        write_immutable(args.output,build_dev_candidate(read('bundle'),read('sheet'),read('key'),read('reviews'),args.checkpoint,args.step))
    elif args.action == 'select':
        freeze_selection(read('candidates'), args.output)
    elif args.action == 'plot':
        plot_training_logs(read('logs', args.logs_jsonl), args.output,
                           rollouts=read('rollouts', True) if args.rollouts else None)
    elif args.action == 'dev-curve':
        plot_dev_curve(read('snapshots'), read('logs', args.logs_jsonl), args.output)
    else:
        read('selection')
        bundle = read('bundle')
        selection = verify_test_selection(bundle, args.selection)
        result = evaluate_reviews(bundle, read('sheet'), read('key'), read('reviews'))
        result['selection_sha256'] = sha256(args.selection)
        result['selected_checkpoint_sha256'] = selection['selected']['checkpoint_sha256']
        result['input_files'] = {name: getattr(args, name+'_sha256') for name in ('bundle', 'sheet', 'key', 'reviews', 'selection')}
        write_immutable(args.output, result)


if __name__ == '__main__': main()
