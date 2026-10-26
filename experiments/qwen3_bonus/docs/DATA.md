# Qwen3 bonus data

Everything generated or downloaded by this pipeline stays inside `qwen3_bonus/`.
The existing application, model files, evaluation inputs, and source datasets are
read-only inputs. Preparation uses Python's standard library; fetching additionally
uses `httpx`. It imports no Torch, Transformers, datasets, or model runtime.

## Reproduce

Run from the SolaceAI project root with the existing Python 3.11 environment:

```sh
.venv/bin/python -m qwen3_bonus.data --help
.venv/bin/python -m qwen3_bonus.data fetch --source qwen3_bonus/artifacts/sources
.venv/bin/python -m qwen3_bonus.data prepare \
  --project-root . \
  --source qwen3_bonus/artifacts/sources \
  --output qwen3_bonus/artifacts/data
.venv/bin/python -m pytest -q qwen3_bonus/tests/test_data.py
```

The source files are already downloaded and prepared in this delivery. Commands
refuse existing output artifacts by default. For reproduction use a fresh directory
inside `qwen3_bonus/`, or explicitly pass `--overwrite` to replace owned artifacts.
The seed is 3407. Repeating preparation with unchanged inputs creates identical
JSONL bytes and manifest hashes. No training or paid service is started.

## Sources and license evidence

The official [SoulChat2.0 repository](https://github.com/scutcyr/SoulChat2.0)
links to [YIRONGCHEN/PsyDTCorpus](https://modelscope.cn/datasets/YIRONGCHEN/PsyDTCorpus).
The complete public files were downloaded from the fixed ModelScope API at revision
`d1d95292df4f87c254e38da93a329d7c303c49f7`:

| File | Raw records | SHA-256 |
| --- | ---: | --- |
| `PsyDTCorpus_train_mulit_turn_packing.json` | 4,760 dialogues | `517fabb22502b84de72ff1d62c7efea0bd16f41cf0939d2cd1d52d285c396345` |
| `PsyDTCorpus_test_single_turn_split.json` | 4,311 response prefixes from 240 dialogues | `5b4eeb46721acbf5e9807fcf46902487855cbcd1ed45770b92f27ab1be95207d` |
| `README.md` | Dataset card | `2882f5a2b2e9bd97a46a0db24e383094ea078068b74df923ad237d2e12bb776e` |

The **dataset's own card**, at the pinned revision, begins with
`license: Apache License 2.0`. This declaration is the dataset-license evidence.
The saved dataset tree contains no standalone LICENSE file; the manifest records
that fact. `SoulChat2.0_CODE_LICENSE` is retained separately from code revision
`13ec529c9e3851eacbbf09bec9029621ac40e773` and is **not** used to infer dataset rights.
The card's research-use/model-safety caution is preserved in the raw README; this
bonus is a research experiment. The pipeline verifies all three pinned source
hashes and the dataset declaration. Missing/changed evidence raises an error;
unresolved license status must block paid training.

`source_manifest.json` records fixed download URLs, revision, hashes, sizes, and
license status. `configs/sources.json` is human-readable provenance, not a source
of arbitrary executable configuration or request URLs. TLS verification remains
on; redirects are limited to the official ModelScope hosts/CDN. Public corpus
contents are untrusted text: only JSON fields are parsed, nothing is executed, and
source system instructions are discarded.

## Classifier: 1,200 training examples and 400 regression cases

`models/psychqa_synthetic.jsonl` contains 2,400 synthetic `instruction/input/output`
rows with four Chinese labels. The pipeline uses one fixed trusted system message,
reserves the 400 inputs from `evaluation/data/drafts/classification.json`, and also
reserves the six development classifier texts in `development-cases.json`.
The latter file contains risk labels LOW/MEDIUM/HIGH and is **not** the existing
400-case emotion regression set.

A seeded shuffle, exact/near duplicate exclusion, and equal per-label selection
produce 300 each of 正常、焦虑、低落、高风险 (1,200 total). There are 69 rejected
near/exact duplicate source inputs in this preparation. The 400 original
regression IDs and emotion labels are retained. Their annotation status is
`synthetic_draft`, requires human review, and does not establish clinical accuracy.

- `classification_train.jsonl`: `id`, `prompt` (system/user messages), `completion`
  (one assistant message with the Chinese label).
- `classification_eval.jsonl`: original `id`, `prompt`, `expected` (Chinese label).

## Response: 128 training contexts and 40 held-out contexts

PsyDTCorpus is **LLM-synthesized counseling dialogue**, informed by a counselor's
style and seed cases; it is not a collection of 5,000 independent real patient
consultations. The test file contains expanding prefixes with repeated original
`id` values. The pipeline first reconstructs the longest coherent prefix per
original dialogue ID, collapsing 4,071 prefix rows. Inconsistent prefixes are
rejected. Entire original official-test dialogues are held out before extracting
response points, including the 200 not selected for the forty-case evaluation.

Selection uses seeded shuffling of whole dialogues. One response point is selected
per dialogue, preferring the second safe response point when available. The
context contains the original alternating user/assistant history ending with a
user turn. The reference assistant reply is retained only as the separate
`reference` field and never inserted into that prompt. Source system messages are
replaced by a trusted support-assistant message with diagnostic/prescribing
boundaries. Contexts have at most 1,800 source-text characters; references have
16–1,200 characters. Trainer token-length limits still apply separately.

Role violations, special model-control tokens, malformed messages, unsafe
assistant diagnostic/prescribing patterns, and repeated references inside their
own context are rejected. The lexical safety screen is conservative and does not
replace expert review. The pipeline checks **all substantive utterances** in a
candidate dialogue, including assistant text, against the complete source
conversations and references saved in `cpsycoune-subset.json`. CPsyCounE remains an
external evaluation source and is not used as response training data.

Both response JSONL files have `id`, `dialogue_id`, `prompt`, `reference`,
`data_source: "psydt"`, and `extra_info` (source revision, official split, topic,
response index, synthetic status, and dataset-license evidence). Dialogue IDs in
train and evaluation are disjoint. No new model completions or fake training
results are generated during data preparation.

## Leakage policy and manifest

Normalize Unicode using NFKC and casefold, retaining Unicode letters/numbers
(including Chinese). Exact normalized input matching works for all lengths. For
texts with at least 24 normalized characters, near duplicates are character
3-gram-set Jaccard similarity >= 0.85. This avoids treating an entire Chinese
sentence as one whitespace token. The inverted-index candidate check computes the
same Jaccard criterion without a semantic model.

Classifier matching uses full input text. Response separation compares full user
transcripts and substantive user utterances against all official held-out
conversations and already-selected dialogues. CPsyCounE exclusion additionally
checks substantive assistant utterances/references. Short conversational phrases
under 24 characters are not used as response-utterance overlap keys; otherwise
common phrases such as 谢谢 would eliminate unrelated dialogues. Exact full
transcript comparisons still apply. Lexical checks cannot prove absence of all
semantic paraphrases or unknown pretraining contamination.

`dataset_manifest.json` records input/output SHA-256 values, source revision,
license evidence, selected IDs, all reserved official-test IDs, counts by class,
filter reasons, duplicate policy, synthetic disclosures, and clinical-validity
limits. `output_hashes` maps every output basename to its SHA-256 for training
preflight; the richer `outputs` map also includes row counts. `filters` describes
processed source rows; processing stops once selected response limits are reached,
so those counters are not a full-corpus quality audit.

Both eval sets provide reproducible development checks. They cannot substantiate
clinical efficacy, real-world safety, or a comparison to a clinician.
