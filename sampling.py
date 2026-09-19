"""Fixed-denominator MC sampling, filtering and token-length calibration."""
import hashlib
import math
from dataclasses import asdict
from structure import Rollout
from utils import verify_answer


def stable_seed(base_seed, problem_id, phase):
    return int(hashlib.sha256(f"{base_seed}:{problem_id}:{phase}".encode()).hexdigest()[:8], 16) % (2**31)


def sample_rollouts(backend, problem, prefix, gold, n, seed, config, verifier=verify_answer):
    completions = backend.generate(problem, prefix, n, seed)
    if len(completions) != n:
        raise ValueError(f"expected {n} rollouts; backend returned {len(completions)}")
    result = []
    for completion in completions:
        if completion.finish_reason == "context_length":
            raise ValueError("prompt exceeds the context budget; choose a suitable model/context/prompt configuration")
        if completion.finish_reason != "stop" and config.incomplete_policy == "error":
            raise ValueError("incomplete rollout (" + completion.finish_reason + "); increase completion/context budget, "
                             "or explicitly choose --incomplete-policy incorrect in a NEW run")
        verdict = verifier(gold, prefix + completion.text, completion.finish_reason)
        if verdict.reason.startswith("math_verify_error:"):
            raise ValueError("verifier execution failed: " + verdict.reason)
        unknown = verdict.correct is None
        if unknown and completion.finish_reason == "stop" and config.unknown_policy == "error":
            raise ValueError("unverifiable final answer: " + verdict.reason)
        result.append(Rollout(completion.text, completion.finish_reason,
                              completion.token_count or backend.count_tokens(completion.text),
                              verdict.correct is True, verdict.extracted, verdict.reason, unknown))
    return result


def filter_problem(backend, row, config, seed, verifier=verify_answer):
    rollouts = sample_rollouts(backend, row["problem"], "", row["final_answer"],
                              config.filter_rollouts, seed, config, verifier)
    n_correct = sum(r.correct for r in rollouts)
    return {"problem_id": row["id"], "question": row["problem"], "gold_answer": row["final_answer"],
            "sampling_seed": seed, "n_rollouts": len(rollouts), "n_correct": n_correct,
            "keep": 0 < n_correct < len(rollouts),
            "status": "retained" if 0 < n_correct < len(rollouts) else ("too_hard" if n_correct == 0 else "too_easy"),
            "rollouts": [asdict(r) for r in rollouts], "model_calls": 1,
            "sampled_completions": len(rollouts), "generated_text_tokens": sum(r.token_count for r in rollouts)}


def calibrate_lengths(filter_results, target_parts=16, mean_override=None):
    """Corpus mean from all complete screening rollouts of retained questions.

    The paper does not specify the population used for its length average.
    This explicit operational choice is recorded in every run.
    """
    count, total = 0, 0
    for record in filter_results:
        if record["keep"]:
            for rollout in record["rollouts"]:
                if rollout["finish_reason"] == "stop" and rollout["token_count"] > 0:
                    count += 1
                    total += rollout["token_count"]
    mean = mean_override if mean_override is not None else (total / count if count else None)
    if mean is not None and (not math.isfinite(mean) or mean <= 0):
        raise ValueError("mean_solution_tokens must be finite and positive")
    return {"mean_solution_tokens": mean,
            "step_token_threshold": mean / target_parts if mean is not None else None,
            "target_parts": target_parts, "n_calibration_rollouts": count,
            "source": "user_override" if mean_override is not None else "retained_screening_rollouts",
            "stop_rule": "span_tokens < threshold; indivisible token/Unicode spans also stop"}
