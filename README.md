# OmegaPRM Data Generation

English | [简体中文](README.zh-CN.md)

This repository independently implements the data-generation component of **[Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/abs/2406.06592)** (Luo et al., 2024), which introduces OmegaPRM. This is an unofficial implementation.

An independent implementation of **OmegaPRM process-supervision data generation for small-scale baseline experiments**. The default pipeline uses 32 screening rollouts per question, 8 rollouts per Monte Carlo (MC) estimate, and up to 100 search iterations per retained question.

This is not the authors' official implementation, and it does not claim to reproduce their exact dataset or results. Published algorithm details are followed where specified; remaining implementation choices and experimental differences are documented in [ALIGNMENT.md](ALIGNMENT.md). PRM training and benchmark evaluation are outside this project's scope.

## Generate a small dataset with a real model

Requires Python 3.10+. Install model dependencies in a GPU environment supported by vLLM:

```bash
python -m pip install -r requirements.txt
```

Prepare a JSON or JSONL file of training questions, then run:

```bash
python generate_prm_data.py \
  --problem data/train_questions.jsonl \
  --output runs/omega_small \
  --backend vllm \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --limit 20 \
  --seed 1234
```

Default settings:

| Setting | Default |
| --- | --- |
| Screening rollouts per question | 32 |
| Rollouts per MC estimate | 8 |
| Search iterations per question | 100 |
| Target number of solution segments | 16 |
| Real-model answer verifier | Math-Verify |
| Temperature / top-p | 0.8 / 0.95 |
| Maximum completion length | 2,048 tokens |
| Maximum context length | 4,096 tokens |

The first four settings follow the paper's stated targets. The default generation model, decoding settings, and verifier are implementation choices, not a reproduction of the original experimental setup.

`--limit 20` selects up to 20 **candidate questions before screening**, using a seeded random subset when the input is larger. It does not guarantee 20 retained questions. The default limit is 100; use `--limit 0` for all input questions. The program does not download a dataset or construct the paper's training/test split. For method comparisons, fix the input and selected question IDs recorded in `manifest.json`, and use consistent screening and length calibration.

The default Qwen model has a configured context length of 4,096 tokens. If a prompt or continuation exceeds the available budget, the default policy raises an error. Start a new run with an appropriate model, context limit, and completion limit. Increasing `--max-model-len` alone does not make a model support a longer context.

One MC call generates eight continuations. A search iteration can require multiple MC calls, so 100 iterations may involve hundreds of calls. Check runtime and the actual call/output-token counts in `summary.json` before scaling up. For debugging, use `--search 5 --limit 5` and report this budget change if using those outputs in experiments. The optional `--max-calls` caps search calls per question, excluding screening; it is disabled by default and reports `call_limit` when reached.

## Input format

Use a JSON array, or one JSON object per line in a `.jsonl` file:

```json
{"id":"train-001","problem":"What is 2 + 3?","final_answer":"5"}
```

`problem` and `final_answer` are required. A stable, unique `id` is recommended; it is optional. Supply the final value or LaTeX expression as the answer, not a full worked solution. Missing fields, empty answers, and duplicate IDs are rejected.

Provide a training subset with a known provenance. This project does not reconstruct the paper's 12K training / 500-question test split.

## Generation pipeline

1. **Screen questions.** Generate 32 independent solutions per question. Retain a question only if these contain both correct and incorrect final answers.
2. **Calibrate length.** Average the token lengths of complete screening solutions for retained questions, then divide by 16. This threshold stays fixed for the run. Use `--mean-solution-tokens` to supply an externally calibrated mean instead.
3. **Search.** Generate eight fresh root continuations. Select an unvisited incorrect rollout from a mixed-MC state using Q+U. During binary search, a positive MC estimate—including MC=1—moves the search right; MC=0 moves it left. Splits preserve the original text, whitespace, and Unicode characters.
4. **Construct edges and export.** Keep the explicit path through positive cut points to the localized zero boundary. Preserve longer edges for auditing, but export only edges satisfying the single-step length convention. The destination prefix's MC is the soft label.

The 32-rollout screening batch and eight-rollout root batch use separate seeds and are counted separately. A retained question can still have an all-correct or all-incorrect eight-rollout root batch. That question ends with `root_no_candidates`; the program does not silently retry it.

This root behavior differs from an all-correct/all-incorrect batch encountered **during binary search**: there, MC=1 or MC=0 determines the search direction and does not discard the question. Such states are excluded from subsequent candidate selection because the pool requires `0 < MC < 1`.

For a question that passes screening, the total number of generated completions is `32 + 8 + 8m`, where `m` is the number of additional, newly evaluated prefixes. Cached prefixes are reused without generation.

## Verification and failure policies

`--verifier auto` selects Math-Verify (same as `--verifier math-verify`). Use `--verifier conservative` for a dependency-free numerical verifier. The conservative verifier does not cover general symbolic answers across MATH. Answer correctness is not determined by substring matching.

Every MC estimate retains its configured denominator, k. Unparseable outputs are not removed to create a smaller denominator.

| Condition | Default behavior |
| --- | --- |
| Complete output with a correct final answer | Score 1 |
| Complete output with an incorrect final answer | Score 0 |
| Complete output without an extractable/parseable answer | Score 0; record the reason |
| Generation hits its length limit before completion | Raise an error |
| Prompt exceeds the context budget | Raise an error |
| Verifier execution error or timeout | Raise an error |

Use `--unknown-policy error` to also reject complete but unverifiable answers. `--incomplete-policy incorrect` explicitly scores truncated outputs as zero; disclose this non-default choice in experiments. These policies fill gaps in the published operational details and should not be attributed to the paper.

Math-Verify must provide `raise_on_error` support; incompatible versions trigger an upgrade message.

## Outputs

| Path | Contents |
| --- | --- |
| `manifest.json` | Code/input fingerprints, configuration, versions, selected IDs |
| `screening/*.json` | Screening rollouts and verdicts, including filtered-out questions |
| `calibration.json` | Mean length, threshold, calibration source and sample count |
| `problems/*.json` | States, edges, rollouts, search events and annotations for retained questions |
| `samples.jsonl` | Deduplicated single-step soft-label training candidates |
| `summary.json` | Screening outcomes, positive/negative sample counts, calls, completions, output-token counts and failure mappings |

Each sample includes:

```text
question, parent_prefix, step, prefix
parent_id, child_id, step_tokens
label, hard_label, mc_source, n_correct, n_rollouts
problem_id, sample_id, sampling_seed, gold_answer
```

`parent_prefix + step == prefix` holds exactly. `label` is the child state's MC, and `hard_label = int(label > 0)`. Use the question, parent prefix and current step as PRM inputs; `gold_answer` is for auditing and must not be included as an input feature. Multi-step edges remain in the per-question audit files.

An already-verified incorrect complete solution is treated as an absorbing terminal state with value zero. Such labels have `mc_source=terminal_answer` and `n_rollouts=0`: no additional sampling is claimed. Other estimates use `mc_source=monte_carlo` with their actual numerator and denominator.

The search determines the positive/negative balance; no extra 1:1 resampling or duplication is applied. All unique small-scale samples are exported without large-scale downsampling. **Split training and validation data by question**, not by individual steps from the same question.

## Resume runs and customize prompts

Add `--resume` to the original command:

```bash
python generate_prm_data.py --problem data/train_questions.jsonl \
  --output runs/omega_small --backend vllm --limit 20 --seed 1234 --resume
```

Screening and search have separate atomic per-question checkpoints. At most the current unfinished question needs to be recomputed. Aggregate outputs can be rebuilt from checkpoints. Changes to the input, configuration, source code, or real-model dependency versions reject resume. v1 outputs are incompatible with v2; use a new run directory.

Only one process may write to a run directory. If a forcibly terminated process leaves `.run.lock`, confirm that the old process has stopped before removing it and resuming.

The default prompt uses the model's chat template and appends the solution prefix verbatim. For base models, use `--prompt-mode plain`. For custom few-shot formatting, pass `--prompt-template prompts/custom.txt`. The file must contain exactly one `{{problem}}` and one `{{prefix}}`, and end with `{{prefix}}` with no trailing newline. Ordinary LaTeX braces do not need escaping. Template contents are included in the run fingerprint.

The paper's Gemma2 four-shot prompt is not bundled or reconstructed. Use `--revision <commit>` to pin the model and tokenizer; an unpinned remote model revision can change.

## Validation and reporting

Validated: standard-library regression tests, controlled error-boundary cases, Q/U values, screening/calibration/edge construction, and checkpoint recovery. Actual execution records are in `VALIDATION.txt`.

Not validated: real GPU/vLLM inference or integration with an installed Math-Verify package. The development environment's package source did not provide that dependency. The paper's original models were not run, and no model accuracy improvement is claimed.

Suggested experimental name: **OmegaPRM data-generation baseline (our implementation)**. Report the generator, dataset subset, screening/MC/search budgets, length calibration, verifier and failure policies. See [ALIGNMENT.md](ALIGNMENT.md) for the full list of implementation choices and remaining differences.

## Visualize a saved search graph

No additional packages are required:

```bash
python tools/visualize_problem.py "runs/omega_small/problems/<hash>.json" -o tree.html
```

Open `tree.html` in your browser. It works offline. Pan/zoom, find a state by ID, click nodes to inspect full prefixes and sampled continuations, or click edges to inspect actions. Colors show MC values; solid edges identify exported training samples. Use **Show disconnected probes** to inspect states without saved connections. Only recorded edges are drawn; rollouts are not expanded into artificial branches. The saved structure can be a DAG with shared states rather than a strict tree.

Use **Export SVG** to save a static diagram. Add `--force` to replace an existing HTML. A ready-to-open dummy-backend example is included at [examples/search_graph.html](examples/search_graph.html). HTML embeds the full checkpoint content; share it only when you intend to share that problem's data.
