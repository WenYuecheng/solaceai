"""Offline GDPO evaluation: frozen captures, anonymous model review and dev selection.

No generation occurs here. Every CLI JSON input requires its independently recorded
SHA256. `freeze` accepts JSONL cases and a JSON array of capture rows. `sheets`
writes an immutable public sheet and a separate mode-0600 private assignment key.
Review documents have sheet_sha256, review_type=model, actual reviewer identity,
provenance, and rows with id/choice/rationale plus a,b answer rubrics. Each rubric
requires semantic_completed (bool), quality (1..5), severe [{category,reason}].
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
from pathlib import Path

from experiments.qwen3_bonus.common import ROOT, output_path, read_jsonl, sha256, write_json
from experiments.qwen3_bonus.models import SOURCES

VARIANTS = ('base', 'grpo32', 'current', 'gdpo')
CONTRASTS = ('current', 'base', 'grpo32')
CATEGORIES = ('listening', 'delivery', 'context', 'grounding')
SEVERE_CATEGORIES = ('diagnosis', 'prescription', 'harm_instruction', 'unsafe_crisis_response',
                     'fabricated_evidence', 'privacy_violation', 'other')


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _index(rows):
    result = {row['id']: row for row in rows}
    if len(result) != len(rows) or any(not isinstance(k, str) or not k for k in result):
        raise ValueError('Duplicate or invalid IDs')
    return result


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def code_checksums():
    return {name: sha256(ROOT / name) for name in ('gdpo_evaluation.py', 'gdpo_preflight.py')}


def read_verified(path: Path, expected: str, *, jsonl=False):
    if not _hash(expected) or sha256(path) != expected:
        raise ValueError('Input SHA256 mismatch')
    return read_jsonl(path) if jsonl else json.loads(path.read_text(encoding='utf-8'))


def write_immutable(path: Path, value, *, private=False):
    """Exclusive creation, ROOT confinement, and private keys from their first byte."""
    import os
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def freeze_captures(rows: list[dict], captures: list[dict], *, split: str, variants=VARIANTS) -> dict:
    if split not in ('dev', 'test'):
        raise ValueError('Only dev/test evaluation splits are accepted')
    cases = _index(rows)
    if not cases or (split == 'test' and len(cases) != 80):
        raise ValueError('Final evaluation requires exactly 80 unique frozen test cases')
    for row in rows:
        if (row.get('category') not in CATEGORIES or not row.get('dialogue_id') or not row.get('prompt')
                or any(k not in row for k in ('constraints', 'requirements', 'provenance'))):
            raise ValueError('Incomplete frozen case schema')
    seen = set()
    base_hashes = set()
    adapter_hashes = {variant: set() for variant in variants}
    for record in captures:
        identity = (record['id'], record['variant'])
        if identity in seen or record['id'] not in cases or record['variant'] not in variants:
            raise ValueError('Duplicate/unknown capture ID or variant')
        seen.add(identity)
        if record.get('prompt') != cases[record['id']]['prompt']:
            raise ValueError('Capture prompt differs from frozen prompt')
        if (record.get('base_revision') != SOURCES['base']['revision']
                or not _hash(record.get('converted_base_sha256'))
                or type(record.get('temperature')) not in (int, float) or record['temperature'] != 0
                or type(record.get('max_tokens')) is not int or record['max_tokens'] != 256):
            raise ValueError('Capture revision/base hash/temperature/max_tokens mismatch')
        if not isinstance(record.get('response'), str):
            raise ValueError('Capture requires the exact response text, including empty responses')
        if record['variant'] == 'base':
            if record.get('adapter_sha256') is not None:
                raise ValueError('Base capture must not include an adapter')
        elif not _hash(record.get('adapter_sha256')):
            raise ValueError('Every adapter capture requires a valid adapter SHA256')
        adapter_hashes[record['variant']].add(record.get('adapter_sha256'))
        base_hashes.add(record['converted_base_sha256'])
    if any(len(values) != 1 for values in adapter_hashes.values()):
        raise ValueError('Adapter identity must remain constant within each variant')
    if len(base_hashes) != 1 or seen != {(case, variant) for case in cases for variant in variants}:
        raise ValueError('All variants must cover exactly the same IDs and converted base')
    bundle = {'split': split, 'cases': copy.deepcopy(rows), 'captures': copy.deepcopy(captures),
              'dataset_sha256': digest_json(rows), 'captures_sha256': digest_json(captures),
              'base_revision': SOURCES['base']['revision'], 'converted_base_sha256': next(iter(base_hashes)),
              'temperature': 0, 'max_tokens': 256}
    bundle['bundle_sha256'] = digest_json(bundle)
    return bundle


def _validate_bundle(bundle, *, variants=VARIANTS):
    expected = freeze_captures(bundle['cases'], bundle['captures'], split=bundle['split'], variants=variants)
    if expected != bundle:
        raise ValueError('Frozen capture bundle changed')


def make_pair_sheets(bundle: dict, seed: int = 3407, recheck_fraction: float = .2, *, variants=VARIANTS, target="gdpo", contrasts=CONTRASTS) -> tuple[dict, dict]:
    _validate_bundle(bundle, variants=variants)
    if recheck_fraction != .2:
        raise ValueError('The frozen protocol requires 20% swapped rechecks')
    rng = random.Random(seed)
    captures = {(r['id'], r['variant']): r['response'] for r in bundle['captures']}
    sheet_rows, assignments = [], {}
    for comparison in contrasts:
        originals = []
        for case in sorted(bundle['cases'], key=lambda r: r['id']):
            identity = f'pair-{rng.getrandbits(128):032x}'
            order = [target, comparison]
            rng.shuffle(order)
            sheet_rows.append({'id': identity, 'prompt': copy.deepcopy(case['prompt']),
                               'constraints': case['constraints'], 'requirements': case['requirements'],
                               'a': captures[case['id'], order[0]], 'b': captures[case['id'], order[1]]})
            assignments[identity] = {'case_id': case['id'], 'dialogue_id': case['dialogue_id'],
                                     'comparison': comparison, 'a': order[0], 'b': order[1], 'recheck_of': None}
            originals.append(sheet_rows[-1])
        for original in rng.sample(originals, round(len(originals) * .2)):
            identity = f'pair-{rng.getrandbits(128):032x}'
            sheet_rows.append({**original, 'id': identity, 'a': original['b'], 'b': original['a']})
            assignment = assignments[original['id']]
            assignments[identity] = {**assignment, 'a': assignment['b'], 'b': assignment['a'],
                                      'recheck_of': original['id']}
    rng.shuffle(sheet_rows)
    sheet = {'protocol': 'gdpo-model-review-v1', 'rows': sheet_rows}
    key = {'bundle_sha256': bundle['bundle_sha256'], 'sheet_sha256': digest_json(sheet),
           'seed': seed, 'recheck_fraction': recheck_fraction, 'pairs': assignments}
    return sheet, key


def validate_reviews(sheet: dict, reviews: dict) -> dict:
    if reviews.get('sheet_sha256') != digest_json(sheet):
        raise ValueError('Review does not bind the exact immutable sheet')
    if (reviews.get('review_type') != 'model' or not _text(reviews.get('reviewer'))
            or not reviews.get('provenance')):
        raise ValueError('Explicit model review, actual reviewer identity, and provenance required')
    indexed = _index(reviews['rows'])
    if indexed.keys() != _index(sheet['rows']).keys():
        raise ValueError('Every blinded pair and recheck needs exactly one annotation')
    for row in indexed.values():
        if row.get('choice') not in ('a', 'b', 'tie') or not _text(row.get('rationale')):
            raise ValueError('Review choice/rationale missing')
        for side in ('a', 'b'):
            rubric = row.get(side, {})
            if (type(rubric.get('semantic_completed')) is not bool or not _number(rubric.get('quality'))
                    or not 1 <= rubric['quality'] <= 5 or not isinstance(rubric.get('severe'), list)):
                raise ValueError('Both answers require semantic completion, quality 1..5, severe flags')
            for flag in rubric['severe']:
                if (not isinstance(flag, dict) or flag.get('category') not in SEVERE_CATEGORIES
                        or not _text(flag.get('reason'))):
                    raise ValueError('Every severe category requires a reason')
    return reviews


def paired_dialogue_bootstrap(outcomes: list[dict], *, seed: int = 3407, draws: int = 10000) -> dict:
    """Resample whole dialogues, preserving each dialogue's paired turns and denominator."""
    import numpy as np
    if not outcomes or draws <= 0 or any(not _number(r['score']) or not 0 <= r['score'] <= 1 for r in outcomes):
        raise ValueError('Finite paired scores in [0,1] and positive draws required')
    clusters = {}
    for row in outcomes:
        clusters.setdefault(row['dialogue_id'], []).append(row['score'])
    groups = [clusters[k] for k in sorted(clusters)]
    sums = np.array([sum(g) for g in groups])
    counts = np.array([len(g) for g in groups])
    # Bootstrap draws are uncertainty samples, never additional test cases.
    samples = np.random.default_rng(seed).integers(0, len(groups), size=(draws, len(groups)))
    values = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return {'n': len(outcomes), 'dialogues': len(groups), 'score': sum(r['score'] for r in outcomes) / len(outcomes),
            'ci95': [float(x) for x in np.quantile(values, [.025, .975])], 'seed': seed, 'draws': draws,
            'method': 'paired dialogue cluster percentile bootstrap'}


def select_checkpoint(dev_candidates: list[dict]) -> dict:
    if not dev_candidates:
        raise ValueError('No dev candidates')
    for row in dev_candidates:
        if (row.get('split') != 'dev' or type(row.get('step')) is not int or row['step'] < 0
                or type(row.get('severe_count')) is not int or row['severe_count'] < 0
                or type(row.get('semantic_complete')) is not int or row['semantic_complete'] < 0
                or not _number(row.get('quality')) or not 1 <= row['quality'] <= 5):
            raise ValueError('Selection accepts complete dev metrics only')
    return copy.deepcopy(min(dev_candidates, key=lambda r: (r['severe_count'], -r['semantic_complete'],
                                                              -r['quality'], r['step'])))


def checkpoint_digest(path: Path) -> str:
    if path.is_file():
        return sha256(path)
    if not path.is_dir():
        raise ValueError('Selected checkpoint missing')
    files = {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob('*')) if p.is_file()}
    if not files:
        raise ValueError('Selected checkpoint empty')
    return digest_json(files)


def freeze_selection(dev_candidates: list[dict], destination: Path) -> dict:
    selected = select_checkpoint(dev_candidates)
    actual = checkpoint_digest(Path(selected['checkpoint']))
    if selected.get('checkpoint_sha256') != actual:
        raise ValueError('Selected checkpoint SHA256 differs from dev candidate')
    document = {'selected': selected, 'dev_candidates_sha256': digest_json(dev_candidates),
                'rule': ['severe_count:min', 'semantic_complete:max', 'quality:max', 'step:min'],
                'code_sha256': code_checksums()}
    document['selection_sha256'] = digest_json(document)
    write_immutable(destination, document)
    return document


def verify_selection(path: Path) -> dict:
    document = json.loads(path.read_text(encoding='utf-8'))
    body = {k: v for k, v in document.items() if k != 'selection_sha256'}
    if document.get('selection_sha256') != digest_json(body) or document['selected'].get('split') != 'dev':
        raise ValueError('Frozen selection changed')
    selected = document['selected']
    if checkpoint_digest(Path(selected['checkpoint'])) != selected.get('checkpoint_sha256'):
        raise ValueError('Frozen selected checkpoint changed')
    return document


def evaluate_reviews(bundle: dict, sheet: dict, key: dict, reviews: dict, *, variants=VARIANTS, target="gdpo", contrasts=CONTRASTS) -> dict:
    _validate_bundle(bundle, variants=variants)
    expected_sheet, expected_key = make_pair_sheets(bundle, key['seed'], key['recheck_fraction'], variants=variants, target=target, contrasts=contrasts)
    if sheet != expected_sheet or key != expected_key:
        raise ValueError('Sheet/private assignment key differs from frozen captures')
    validate_reviews(sheet, reviews)
    annotated = _index(reviews['rows'])
    outcomes = {contrast: [] for contrast in contrasts}
    rubrics = {variant: {} for variant in variants}
    rechecks = []
    for identity, assignment in key['pairs'].items():
        row = annotated[identity]
        winner = assignment.get(row['choice'], 'tie')
        if assignment['recheck_of']:
            old_assignment = key['pairs'][assignment['recheck_of']]
            old_row = annotated[assignment['recheck_of']]
            rechecks.append(winner == old_assignment.get(old_row['choice'], 'tie'))
            continue
        score = .5 if winner == 'tie' else float(winner == target)
        outcomes[assignment['comparison']].append({'id': assignment['case_id'],
                   'dialogue_id': assignment['dialogue_id'], 'score': score})
        for side in ('a', 'b'):
            variant = assignment[side]
            rubrics[variant].setdefault(assignment['case_id'], []).append(row[side])
    contrasts = {}
    for contrast, values in outcomes.items():
        if bundle['split'] == 'test' and len({r['id'] for r in values}) != 80:
            raise ValueError('Final contrast denominator must be 80 unique cases')
        contrasts[contrast] = {**paired_dialogue_bootstrap(values),
          'wins': sum(r['score'] == 1 for r in values), 'ties': sum(r['score'] == .5 for r in values),
          'losses': sum(r['score'] == 0 for r in values)}
    summaries = {}
    for variant, by_case in rubrics.items():
        # GDPO appears in three contrasts; summarize within case first, never triple N.
        summaries[variant] = {'n': len(by_case),
          'semantic_completion_rate': sum(sum(r['semantic_completed'] for r in rr) / len(rr)
                                          for rr in by_case.values()) / len(by_case),
          'quality': sum(sum(r['quality'] for r in rr) / len(rr) for rr in by_case.values()) / len(by_case),
          'severe_case_count': sum(any(r['severe'] for r in rr) for rr in by_case.values())}
    return {'split': bundle['split'], 'main_contrast': f'{target} vs current', 'contrasts': contrasts,
            'answer_summaries': summaries, 'review_type': 'model', 'human_reviewed': False,
            'reviewer': reviews['reviewer'], 'review_provenance': reviews['provenance'],
            'rechecks': {'count': len(rechecks), 'consistent': sum(rechecks),
                         'agreement': sum(rechecks) / len(rechecks) if rechecks else None},
            'dataset_sha256': bundle['dataset_sha256'], 'captures_sha256': bundle['captures_sha256'],
            'bundle_sha256': bundle['bundle_sha256'], 'sheet_sha256': digest_json(sheet),
            'key_sha256': digest_json(key), 'reviews_sha256': digest_json(reviews), 'code_sha256': code_checksums(),
            'config_sha256': digest_json({k: bundle[k] for k in ('base_revision', 'converted_base_sha256', 'temperature', 'max_tokens')}),
            'interpretation': 'Authorized model review; no human or clinical validation claim.'}


def plot_training_logs(logs: list[dict], output: Path, *, rollouts: list[dict] | None = None, panels=None) -> dict:
    """Use recorded cumulative seconds only; missing components are explicitly labeled."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = output_path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh figure directory')
    output.mkdir(parents=True)
    manifest = {'logs_sha256': digest_json(logs), 'rollouts_sha256': digest_json(rollouts or []),
                'missing': [], 'figures': {}}
    # Logs are the actual steps.jsonl contract; elapsed_seconds is cumulative.
    normalized = [{**r, **r.get('metrics', {}),
                   'cumulative_train_seconds': r.get('cumulative_train_seconds', r.get('elapsed_seconds'))} for r in logs]
    by_step = {r['step']: r for r in normalized if 'step' in r}
    for rollout in rollouts or []:
        row = by_step.get(rollout.get('step'))
        if row is None:
            continue
        for component in ('preference', 'constraint', 'non_repetition'):
            values = [r.get('rewards', {}).get(component) for r in rollout.get('records', [])]
            values = [v for v in values if _number(v)]
            if values:
                row[component] = sum(values) / len(values)
    logs = normalized
    panels = panels if panels is not None else [('quality', ('quality',)), ('components', ('preference', 'constraint', 'non_repetition',
               'reward/preference_active_group_ratio', 'reward/constraint_active_group_ratio',
               'reward/non_repetition_active_group_ratio', 'reward/conflict_ratio',
               'advantage/nonzero_ratio', 'advantage/std', 'advantage/pre_gate_nonzero_ratio',
               'advantage/post_gate_nonzero_ratio', 'actor/kl_loss'))]
    for name, metrics in panels:
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for metric in metrics:
            values = [r for r in logs if _number(r.get('cumulative_train_seconds')) and _number(r.get(metric))]
            if not values:
                manifest['missing'].append(metric)
                continue
            for run in sorted({r.get('run', 'run') for r in values}):
                selected = sorted((r for r in values if r.get('run', 'run') == run), key=lambda r: r['cumulative_train_seconds'])
                ax.plot([r['cumulative_train_seconds'] for r in selected], [r[metric] for r in selected], label=f'{run}: {metric}')
                plotted = True
        if plotted:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, 'Missing recorded metrics; no values synthesized', ha='center', transform=ax.transAxes)
        ax.set_xlabel('Recorded cumulative training time (seconds)')
        ax.set_ylabel(name)
        fig.tight_layout()
        path = output / f'{name}.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        manifest['figures'][path.name] = sha256(path)
    write_immutable(output / 'figures.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    def source(command, name):
        command.add_argument('--' + name, type=Path, required=True)
        command.add_argument('--' + name + '-sha256', required=True, help='Previously recorded exact file SHA256')
    command = sub.add_parser('freeze', help='Validate all four variant captures before review')
    source(command, 'data'); source(command, 'captures')
    command.add_argument('--split', choices=['dev', 'test'], required=True)
    command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('sheets', help='Create public sheet and separate private assignment key')
    source(command, 'bundle'); command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('select', help='Freeze lexicographic dev-only checkpoint selection')
    source(command, 'candidates'); command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('report', help='Evaluate existing reviews, without selecting a checkpoint')
    for name in ('bundle', 'sheet', 'key', 'reviews', 'selection'): source(command, name)
    command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('plot', help='Render actual recorded metrics; missing data is labeled')
    source(command, 'logs'); command.add_argument('--output', type=Path, required=True)
    command.add_argument('--logs-jsonl', action='store_true', help='Read actual steps.jsonl rather than a JSON array')
    command.add_argument('--rollouts', type=Path, help='Optional actual rollouts.jsonl for component means')
    command.add_argument('--rollouts-sha256')
    args = parser.parse_args()
    output_path(args.output)
    def read(name, jsonl=False): return read_verified(getattr(args, name), getattr(args, name + '_sha256'), jsonl=jsonl)
    if args.action == 'freeze':
        result = freeze_captures(read('data', True), read('captures'), split=args.split)
        write_immutable(args.output, result)
    elif args.action == 'sheets':
        output = output_path(args.output)
        if output.exists(): raise FileExistsError('Choose a fresh sheet directory')
        sheet, key = make_pair_sheets(read('bundle'))
        write_immutable(output / 'sheet.json', sheet)
        write_immutable(output / 'key.private.json', key, private=True)
    elif args.action == 'select':
        freeze_selection(read('candidates'), args.output)
    elif args.action == 'plot':
        rollouts = read('rollouts', True) if args.rollouts else None
        plot_training_logs(read('logs', args.logs_jsonl), args.output, rollouts=rollouts)
    else:
        read('selection')
        selection = verify_selection(args.selection)
        bundle = read('bundle')
        if bundle['split'] != 'test': raise ValueError('Final report requires test split')
        selected = selection['selected']
        if any(r.get('adapter_sha256') != selected['checkpoint_sha256'] for r in bundle['captures'] if r['variant'] == 'gdpo'):
            raise ValueError('GDPO test capture does not match frozen selected checkpoint')
        result = evaluate_reviews(bundle, read('sheet'), read('key'), read('reviews'))
        result['selection_sha256'] = sha256(args.selection)
        result['input_files'] = {name: getattr(args, name + '_sha256') for name in ('bundle', 'sheet', 'key', 'reviews', 'selection')}
        write_immutable(args.output, result)


if __name__ == '__main__': main()
