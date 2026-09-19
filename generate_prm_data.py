"""Small-scale, paper-aligned OmegaPRM data generation (arXiv v2)."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
from dataclasses import asdict
from completer import VLLMCompleter
from search import Search
from structure import SearchConfig
from sampling import filter_problem, calibrate_lengths, stable_seed
from utils import MathVerifier, verify_answer

VERSION = "2.0.0"


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            if isinstance(text, str):
                f.write(text)
            else:
                f.writelines(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def load_problems(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)
    if not isinstance(data, list) or not data:
        raise ValueError("input must be a nonempty JSON array or JSONL file")
    rows, ids = [], set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"row {index}: expected an object")
        problem, answer = item.get("problem"), item.get("final_answer")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"row {index}: nonempty problem string required")
        if isinstance(answer, bool) or not isinstance(answer, (str, int, float)):
            raise ValueError(f"row {index}: final_answer must be a string or finite number")
        if isinstance(answer, float) and not math.isfinite(answer):
            raise ValueError(f"row {index}: non-finite answer")
        answer = str(answer).strip()
        if not answer:
            raise ValueError(f"row {index}: empty final_answer")
        raw_id = item.get("id", index)
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
            raise ValueError(f"row {index}: id must be a string or integer")
        identifier = str(raw_id)
        if not identifier or identifier in ids:
            raise ValueError(f"row {index}: empty or duplicate id")
        ids.add(identifier)
        rows.append({"id": identifier, "problem": problem, "final_answer": answer})
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, help="training questions: JSON array or JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["vllm"], default="vllm")
    parser.add_argument("--verifier", choices=["auto", "conservative", "math-verify"], default="auto")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--revision", help="model/tokenizer commit or revision; pin for experiments")
    parser.add_argument("--limit", type=int, default=100, help="seeded subset BEFORE screening; 0 means all input")
    parser.add_argument("--filter-rollouts", type=int, default=32)
    parser.add_argument("--rollout", "--rollouts", dest="rollouts", type=int, default=8)
    parser.add_argument("--search", "--searches", dest="searches", type=int, default=100)
    parser.add_argument("--target-parts", type=int, default=16)
    parser.add_argument("--mean-solution-tokens", type=float, help="optional externally calibrated mean; default uses screening")
    parser.add_argument("--max-calls", type=int, help="optional cap on SEARCH calls per question, excluding screening")
    parser.add_argument("--unknown-policy", choices=["incorrect", "error"], default="incorrect")
    parser.add_argument("--incomplete-policy", choices=["error", "incorrect"], default="error")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--prompt-mode", choices=["chat", "plain"], default="chat")
    parser.add_argument("--prompt-template", help="UTF-8 few-shot/raw template with {{problem}} and final {{prefix}}")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.max_tokens < 1 or args.max_model_len <= args.max_tokens:
        parser.error("require limit >= 0 and 0 < max-tokens < max-model-len")
    if not 0 < args.temperature <= 2 or not 0 < args.top_p <= 1 or args.tensor_parallel_size < 1:
        parser.error("invalid temperature/top-p/tensor-parallel-size")
    if args.mean_solution_tokens is not None and (not math.isfinite(args.mean_solution_tokens) or args.mean_solution_tokens <= 0):
        parser.error("mean-solution-tokens must be finite and positive")
    if args.verifier == "auto":
        args.verifier = "math-verify"
    return args


def record_path(directory, problem_id):
    return directory / (hashlib.sha256(problem_id.encode()).hexdigest() + ".json")


def read_record(path, row, fingerprint):
    value = json.loads(path.read_text(encoding="utf-8"))
    if any(value.get(k) != v for k, v in {
        "fingerprint": fingerprint, "problem_id": row["id"],
        "question": row["problem"], "gold_answer": row["final_answer"]}.items()):
        raise ValueError("checkpoint does not match this run: " + str(path))
    return value


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if args.resume and not output.is_dir():
        raise ValueError("--resume requires an existing run directory")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError("run locked; remove .run.lock only after confirming the previous process stopped") from None
    os.close(fd)
    try:
        run_locked(args)
    finally:
        lock.unlink(missing_ok=True)


def run_locked(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows = load_problems(args.problem)
    n_input = len(rows)
    if args.limit and len(rows) > args.limit:
        indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.limit))
        rows = [rows[i] for i in indices]
    config = SearchConfig(filter_rollouts=args.filter_rollouts, rollouts=args.rollouts,
                          searches=args.searches, target_parts=args.target_parts,
                          max_calls=args.max_calls, unknown_policy=args.unknown_policy,
                          incomplete_policy=args.incomplete_policy)
    template = Path(args.prompt_template).read_text(encoding="utf-8") if args.prompt_template else None
    if template is not None and (template.count("{{problem}}") != 1 or template.count("{{prefix}}") != 1
                                 or not template.endswith("{{prefix}}")):
        raise ValueError("prompt template needs exactly one {{problem}}, and must end with exactly one {{prefix}}")
    options = {k: v for k, v in vars(args).items() if k not in {"problem", "output", "resume", "prompt_template"}}
    options["prompt_template_text"] = template
    runtime = {"python": platform.python_version(), "packages": {}}
    for name in ["vllm", "transformers", "torch", "math-verify", "tokenizers"]:
        try: runtime["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: runtime["packages"][name] = None
    code_files = sorted(Path(__file__).parent.glob("*.py"))
    code_hash = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in code_files)).hexdigest()
    fingerprint_data = {"rows": rows, "options": options, "code": code_hash, "runtime": runtime}
    fingerprint = hashlib.sha256(json_text(fingerprint_data).encode()).hexdigest()
    output = Path(args.output)
    manifest_path = output / "manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise ValueError("missing manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("input, configuration, code or model runtime changed; use a NEW output directory")
    else:
        if any(p.name != ".run.lock" for p in output.iterdir()):
            raise ValueError("output directory not empty; use --resume or a new directory")
        manifest = {"version": VERSION, "paper": "arXiv:2406.06592v2",
                    "implementation": "independent paper-aligned data-generation baseline",
                    "fingerprint": fingerprint, "code_sha256": code_hash,
                    "options": options, "search_config": asdict(config), "runtime": runtime,
                    "n_input": n_input, "n_selected": len(rows), "selected_ids": [r["id"] for r in rows],
                    "token_length_unit": "model_tokenizer"}
        atomic_write(manifest_path, json_text(manifest))
    screening_dir, problem_dir = output / "screening", output / "problems"
    screening_dir.mkdir(exist_ok=True)
    problem_dir.mkdir(exist_ok=True)
    backend = verifier = None

    def get_backend():
        nonlocal backend, verifier
        if backend is None:
            verifier = MathVerifier() if args.verifier == "math-verify" else verify_answer
            backend = VLLMCompleter(
                model=args.model, max_tokens=args.max_tokens, temperature=args.temperature,
                top_p=args.top_p, max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size, prompt_mode=args.prompt_mode,
                revision=args.revision, prompt_template=template)
        return backend

    # Phase 1: Appendix A screening. These 32 samples are NOT silently reused
    # as an 8-sample root estimate: the two stages have separate seeds and costs.
    screen_summaries = []
    retained = []
    for row in rows:
        path = record_path(screening_dir, row["id"])
        if path.exists():
            screened = read_record(path, row, fingerprint)
        else:
            b = get_backend()
            screened = filter_problem(b, row, config, stable_seed(args.seed, row["id"], "screen"), verifier)
            screened["fingerprint"] = fingerprint
            atomic_write(path, json_text(screened))
        if screened["keep"]:
            retained.append(row)
        screen_summaries.append({k: screened[k] for k in ["problem_id", "n_correct", "n_rollouts", "status", "sampled_completions", "generated_text_tokens"]}
                               | {"unknown_as_incorrect": sum(r["unknown_mapped_to_incorrect"] for r in screened["rollouts"])})
        logging.info("screen %s: %s (%d/%d)", row["id"], screened["status"], screened["n_correct"], screened["n_rollouts"])
    # Phase 2: fixed corpus-wide target length. Never recalibrate independently
    # for each selected branch, which would alter the binary stopping rule.
    calibration = calibrate_lengths((read_record(record_path(screening_dir, r["id"]), r, fingerprint) for r in retained),
                                    args.target_parts, args.mean_solution_tokens)
    calibration["fingerprint"] = fingerprint
    atomic_write(output / "calibration.json", json_text(calibration))
    if retained and calibration["step_token_threshold"] is None:
        raise ValueError("no complete screening rollouts for length calibration; fix generation settings")
    # Phase 3: search, with one restartable checkpoint per retained question.
    summaries, n_samples, n_positive = [], 0, 0
    for row in retained:
        path = record_path(problem_dir, row["id"])
        if path.exists():
            result = read_record(path, row, fingerprint)
        else:
            b = get_backend()
            result = Search(b, config, calibration["step_token_threshold"],
                            stable_seed(args.seed, row["id"], "search"), verifier).run(row["id"], row["problem"], row["final_answer"])
            result["fingerprint"] = fingerprint
            atomic_write(path, json_text(result))
        n_samples += len(result["samples"])
        n_positive += sum(s["hard_label"] for s in result["samples"])
        summaries.append({k: result[k] for k in ["problem_id", "status", "model_calls", "searches", "unknown_as_incorrect", "sampled_completions", "generated_text_tokens"]}
                         | {"samples": len(result["samples"])})
        logging.info("search %s: %s; calls=%d samples=%d", row["id"], result["status"], result["model_calls"], len(result["samples"]))

    def sample_lines():
        for row in retained:
            value = read_record(record_path(problem_dir, row["id"]), row, fingerprint)
            for sample in value["samples"]:
                yield json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n"

    atomic_write(output / "samples.jsonl", sample_lines())
    atomic_write(output / "summary.json", json_text({
        "n_selected": len(rows), "n_retained": len(retained), "n_samples": n_samples,
        "n_positive": n_positive, "n_negative": n_samples - n_positive,
        "screening_model_calls": len(rows), "search_model_calls": sum(s["model_calls"] for s in summaries),
        "total_sampled_completions": sum(s["sampled_completions"] for s in screen_summaries + summaries),
        "total_generated_text_tokens": sum(s["generated_text_tokens"] for s in screen_summaries + summaries),
        "screening": screen_summaries, "problems": summaries}))
    logging.info("Saved %d samples from %d retained / %d selected questions", n_samples, len(retained), len(rows))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
