# Paper Alignment and Implementation Choices

English | [简体中文](ALIGNMENT.zh-CN.md)

Reference: [OmegaPRM, arXiv v2, December 11, 2024](https://arxiv.org/html/2406.06592v2).

This document specifies the behavior of this independent implementation. It does not establish equivalence to the authors' code or reproduction of their data or experimental results.

## Settings aligned with the published method

| Component | Default implementation | Paper reference |
| --- | --- | --- |
| Question screening | 32 rollouts; retain questions with both correct and incorrect answers | Appendix A |
| MC estimation | Eight rollouts per prefix; number correct divided by eight | Sections 3.2 and 4 |
| Search budget | 100 iterations per question, including root selections | Section 4 |
| Q | `0.5**(1-MC) * 0.9**(rollout_tokens/500)` | Sections 3.3 and 4 |
| U | `0.125 * sqrt(sum(visits)) / (1+visits)` | Sections 3.3 and 4 |
| Binary search | Positive MC: search right; zero MC: search left | Section 3.2 |
| Segment threshold | Mean solution token length divided by 16 | Section 4.2 |
| Training targets | Soft MC labels for single-step edges; hard labels also retained | Sections 3.4 and 4.3 |

## Operational choices not fully determined by the paper

The following choices make this implementation concrete and reproducible. They should be revisited if the authors' code or further clarification becomes available.

### Population used to estimate mean length

The mean uses complete solutions generated during the 32-rollout screening stage for questions retained in this run. Both correct and incorrect solutions contribute. Lengths use the generator's tokenizer and exclude invisible termination tokens.

The threshold is fixed across the run, rather than recalculated for every selected rollout. An external mean can be supplied to share calibration across baselines.

### Cut positions and indivisible spans

Fast-tokenizer offset mappings select an original-text character boundary nearest the token midpoint. Text is preserved exactly; the implementation does not decode partial Unicode characters or split by lines or words.

Binary search stops when the remaining span has strictly fewer tokens than the threshold. It also stops at a single token or when no valid internal character boundary exists.

### Single-step edges

The stored path connects the source, positive cut points, and final localized zero boundary. The exact text appended between adjacent nodes is the action.

Only actions satisfying the same length stopping rule are exported as single-step samples. Longer actions remain as multi-step edges in the audit data. MC values are not fabricated for unmeasured prefixes, and a long action's label is not copied to each constituent token.

### Candidate pool and state maintenance

Candidates are unvisited incorrect rollouts from states with `0 < MC < 1`. Correct rollouts remain part of the MC denominator. Each newly sampled prefix receives k continuations; cached prefixes and their continuations are reused.

Only the selected state's visit count is incremented; ancestors are not recursively updated. Insertion order breaks score ties.

The paper describes the pool using “all rollouts,” while its error-localization and prioritization discussion focuses on incorrect solutions. This implementation explicitly adopts the incorrect-rollout interpretation.

### Screening versus root estimation

The 32-rollout screening stage and eight-rollout root estimate are independent. Screening outputs are not silently reused as an eight-sample root estimate.

If the new root batch is all correct or all incorrect, the question ends with `root_no_candidates`, without additional retries. All-correct or all-incorrect batches encountered later during binary search instead guide its direction and do not discard the question.

### Terminal states

An already-verified incorrect complete solution is an absorbing terminal state with value zero. No further continuation is generated from it. Its labels retain `mc_source=terminal_answer` and an actual additional-rollout count of zero.

This is an explicit terminal-state convention. Nonterminal prefix labels are based on sampled continuations.

### Caching and deduplication

Identical text prefixes are merged, so the audit structure is a prefix DAG representing a deduplicated conceptual search tree. Each parent–child edge is exported once. Repeated selection of identical text does not create duplicate training samples.

### Verification failures

Complete answers that cannot be recognized as correct receive zero by default, preserving the MC denominator and recording the reason. Verifier execution errors/timeouts, truncated generation and context overflow stop the run by default.

These policies are recorded in the manifest and are not claimed to be explicitly specified by the paper.

## Deliberate experimental differences

- The default Qwen math 7B instruction model supports small-scale generation; **it is not equivalent to the paper's Gemini or Gemma2 model and prompt configurations**.
- Math-Verify substitutes for the original answer-checking implementation, which was not recovered from the paper. Inspect its judgments on your dataset.
- Temperature 0.8, top-p 0.95, the 2,048-token completion cap, concrete prompts and seeds are project settings, not attributed paper hyperparameters.
- The user supplies a training subset; the original train/test partition is not reconstructed automatically.
- All unique single-step edges are exported for small-scale runs, without additional class balancing or a forced dataset size.
- Non-default call caps or policies that score truncated outputs as incorrect must be disclosed as budget or failure-policy differences.

## What to fix and report when using this baseline

Record the generator and revision, training subset and excluded evaluation questions, retained question IDs, calibrated mean length, prompt template and tokenizer, temperature/top-p/completion limit, seed, verifier, failure policies, actual screening/search costs, and positive/negative sample counts.

The manifest, calibration, summary and per-question audit files capture most of this information. You must additionally establish the provenance and split of the supplied dataset.

Conclusions apply to the current generator and data configuration. Passing software tests does not remove process-label noise from finite sampling or non-monotonic self-correction. A binary-search boundary is not a formal proof of the first logically incorrect step.

Implementation references: [vLLM documentation](https://docs.vllm.ai/en/latest/), [Math-Verify](https://github.com/huggingface/Math-Verify), and the [default Qwen model configuration](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct/blob/main/config.json).
