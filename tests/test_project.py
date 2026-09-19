import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from completer import VLLMCompleter
import generate_prm_data
from search import Search
from structure import Completion, Rollout, State, SearchConfig
from sampling import sample_rollouts, filter_problem, calibrate_lengths
from utils import MathVerifier, Verdict, numeric_value, verify_answer

GOOD = "\\boxed{42}"
BAD = "\\boxed{43}"


class BoundaryBackend:
    """Controlled success boundary on a fixed wrong trajectory."""
    def __init__(self, boundary=20, text=None):
        self.boundary = boundary
        self.text = text if text is not None else "a" * 30 + BAD
        self.calls = []
    def count_tokens(self, text): return len(text)
    def split_midpoint(self, text):
        return (text[:len(text)//2], text[len(text)//2:]) if len(text) > 1 else None
    def generate(self, problem, prefix, n, seed):
        self.calls.append((prefix, n, seed))
        if not prefix: values = [GOOD if i % 2 == 0 else self.text for i in range(n)]
        else: values = [GOOD if len(prefix) < self.boundary else BAD] * n
        return [Completion(x, "stop", len(x)) for x in values]


class AdditionBackend:
    """Deterministic 'What is A + B?' toy standing in for vLLM in CLI tests."""
    def count_tokens(self, text): return len(text)
    def split_midpoint(self, text):
        return (text[:len(text)//2], text[len(text)//2:]) if len(text) > 1 else None
    def generate(self, problem, prefix, n, seed):
        match = re.fullmatch(r"\s*What is (\d+) \+ (\d+)\?\s*", problem)
        answer = int(match[1]) + int(match[2])
        results = []
        for i in range(n):
            correct = "BAD" not in prefix and i % 2 == 0
            if not prefix:
                text = "Add the two numbers.\n" + ("Compute the sum.\n" if correct else "BAD arithmetic.\n")
            elif "BAD" in prefix:
                text = ""
            else:
                text = "Compute the sum.\n" if correct else "BAD arithmetic.\n"
            text += "\\boxed{" + str(answer if correct else answer + 1) + "}"
            results.append(Completion(text, "stop", len(text)))
        return results


def search_result(boundary=20, threshold=3.0, searches=1, max_calls=None):
    backend = BoundaryBackend(boundary)
    engine = Search(backend, SearchConfig(rollouts=2, searches=searches, max_calls=max_calls), threshold)
    result = engine.run("p", "q", "42")
    return backend, engine, result


class Answers(unittest.TestCase):
    def test_numeric_false_positive_regression(self):
        self.assertIs(verify_answer("2", "The answer is 12.").correct, False)
        self.assertIs(verify_answer("2", "The answer is -2.").correct, False)

    def test_equivalent_arithmetic(self):
        for answer in [r"\boxed{\frac{1}{2}}", "Final answer: 50%", "0.5", "5e-1"]:
            with self.subTest(answer=answer):
                self.assertIs(verify_answer("0.5", answer).correct, True)

    def test_nested_box_and_final_box(self):
        self.assertIs(verify_answer("0.5", r"Earlier \boxed{9}; final \boxed{\frac{1}{2}}.").correct, True)

    def test_latest_final_marker_overrides_earlier_box(self):
        self.assertIs(verify_answer("2", r"Earlier \boxed{2}. Final answer: 12").correct, False)
        self.assertIsNone(verify_answer("2", r"Earlier \boxed{2}. Final \boxed{").correct)

    def test_ambiguous_numeric_whitespace(self):
        self.assertIsNone(verify_answer("12", "Answer: 1 2").correct)

    def test_symbolic_unknown_is_not_false(self):
        self.assertIsNone(verify_answer("x+x", r"\boxed{2*x}").correct)

    def test_missing_answer_not_guessed_from_reasoning(self):
        self.assertIsNone(verify_answer("42", "We started with 42 marbles and kept thinking.").correct)

    def test_truncation_not_negative_even_if_boxed(self):
        self.assertIsNone(verify_answer("42", GOOD, "length").correct)
        self.assertIsNone(verify_answer("42", GOOD, "context_length").correct)

    def test_parser_rejects_code_and_large_exponents(self):
        for value in ["__import__('os').system('echo no')", "1e999999999", "2**1000000", "1/0"]:
            self.assertIsNone(numeric_value(value))

    def test_thousands_and_decimals(self):
        self.assertIs(verify_answer("1000", "Answer: 1,000").correct, True)
        self.assertIs(verify_answer("0.3", "Answer: 0.1+0.2").correct, True)


class Sampling(unittest.TestCase):
    def test_paper_defaults(self):
        c = SearchConfig()
        self.assertEqual((c.filter_rollouts, c.rollouts, c.searches, c.target_parts), (32, 8, 100, 16))
        self.assertEqual((c.alpha, c.beta, c.length_scale, c.c_puct), (0.5, 0.9, 500.0, 0.125))
        self.assertIsNone(c.max_calls)

    def test_filter_32_and_both_extremes(self):
        row = {"id": "q", "problem": "q", "final_answer": "42"}
        for k in [0, 1, 16, 31, 32]:
            class Backend:
                def generate(self, q, p, n, seed):
                    self.n = n
                    return [Completion(GOOD if i < k else BAD) for i in range(n)]
                def count_tokens(self, text): return len(text)
            b = Backend()
            record = filter_problem(b, row, SearchConfig(), 1)
            self.assertEqual(b.n, 32)
            self.assertEqual(record["keep"], 0 < k < 32)

    def test_fixed_denominator_for_unparsed_answers(self):
        class Backend:
            def generate(self, *args): return [Completion(GOOD), Completion("No final answer")]
            def count_tokens(self, text): return len(text)
        rollouts = sample_rollouts(Backend(), "q", "", "42", 2, 1, SearchConfig())
        self.assertEqual(sum(r.correct for r in rollouts) / len(rollouts), 0.5)
        self.assertTrue(rollouts[1].unknown_mapped_to_incorrect)
        with self.assertRaises(ValueError):
            sample_rollouts(Backend(), "q", "", "42", 2, 1, SearchConfig(unknown_policy="error"))

    def test_incomplete_is_fail_fast_by_default(self):
        class Backend:
            def generate(self, *args): return [Completion(GOOD), Completion(BAD, "length")]
            def count_tokens(self, text): return len(text)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            sample_rollouts(Backend(), "q", "", "42", 2, 1, SearchConfig())
        rollouts = sample_rollouts(Backend(), "q", "", "42", 2, 1, SearchConfig(incomplete_policy="incorrect"))
        self.assertEqual([r.correct for r in rollouts], [True, False])

    def test_context_failure_is_not_a_negative(self):
        class Backend:
            def generate(self, *args): return [Completion("", "context_length")]
        with self.assertRaisesRegex(ValueError, "context"):
            sample_rollouts(Backend(), "q", "", "42", 1, 1, SearchConfig(incomplete_policy="incorrect"))

    def test_mean_length_uses_retained_complete_rollouts(self):
        records = [
            {"keep": True, "rollouts": [{"finish_reason": "stop", "token_count": 32}, {"finish_reason": "stop", "token_count": 64}]},
            {"keep": False, "rollouts": [{"finish_reason": "stop", "token_count": 9999}]},
        ]
        c = calibrate_lengths(records)
        self.assertEqual(c["mean_solution_tokens"], 48)
        self.assertEqual(c["step_token_threshold"], 3)
        self.assertEqual(calibrate_lengths(records, mean_override=160)["step_token_threshold"], 10)
        self.assertIsNone(calibrate_lengths([])["step_token_threshold"])


class SearchTests(unittest.TestCase):
    def test_figure_two_path_and_child_mc_labels(self):
        class FigureBackend(BoundaryBackend):
            def generate(self, question, prefix, n, seed):
                positives = {"": 2, "abcd": 2, "abcdef": 4, "abcdefg": 0}[prefix]
                bad = "abcdefgh" if not prefix else "N"
                return [Completion("Y" if i < positives else bad) for i in range(n)]
        verifier = lambda gold, response, reason: Verdict(response.endswith("Y"), None, "controlled")
        result = Search(FigureBackend(), SearchConfig(searches=1), 1.5, verifier=verifier).run("q", "q", "Y")
        nodes = {s["id"]: s for s in result["states"]}
        path = [nodes[i]["prefix"] for i in result["events"][-1]["path_state_ids"]]
        self.assertEqual(path, ["", "abcd", "abcdef", "abcdefg"])
        self.assertEqual([edge["mc"] for edge in result["edges"]], [0.25, 0.5, 0.0])
        self.assertEqual(result["samples"][0]["step"], "g")

    def test_all_boundary_locations_and_short_edge_export(self):
        for boundary in range(1, 41):
            b, e, r = search_result(boundary=boundary)
            event = r["events"][-1]
            self.assertEqual(event["status"], "boundary_found")
            states = {s["id"]: s for s in r["states"]}
            low, high = states[event["positive_state_id"]], states[event["negative_state_id"]]
            self.assertLess(len(low["prefix"]), boundary)
            self.assertGreaterEqual(len(high["prefix"]), boundary)
            self.assertLess(event["boundary_tokens"], 3)
            self.assertLessEqual(r["model_calls"], 7)
            self.assertTrue(r["samples"])
            for sample in r["samples"]:
                self.assertEqual(sample["parent_prefix"] + sample["step"], sample["prefix"])
                self.assertLess(sample["step_tokens"], 3)
                self.assertEqual(sample["label"], states[sample["child_id"]]["mc"])

    def test_mc_one_continues_right(self):
        b, e, r = search_result(boundary=35)
        self.assertGreater(len(b.calls), 2)
        self.assertTrue(any(s["mc"] == 1 and s["in_tree"] for s in r["states"]))
        self.assertEqual(r["events"][-1]["status"], "boundary_found")

    def test_only_wrong_rollout_selected(self):
        b, e, r = search_result()
        root = r["states"][0]
        self.assertFalse(root["rollouts"][0]["visited"])
        self.assertTrue(root["rollouts"][1]["visited"])

    def test_no_positive_midpoint_still_exports_negative(self):
        b, e, r = search_result(boundary=1)
        self.assertEqual(len(r["samples"]), 1)
        self.assertEqual(r["samples"][0]["label"], 0)

    def test_multi_step_edges_retained_but_not_exported(self):
        b, e, r = search_result(boundary=29)
        self.assertTrue(any(not edge["single_step"] for edge in r["edges"]))
        exported = {(s["parent_id"], s["child_id"]) for s in r["samples"]}
        self.assertEqual(exported, {(x["parent_id"], x["child_id"]) for x in r["edges"] if x["single_step"]})

    def test_root_obeys_search_zero(self):
        b, e, r = search_result(searches=0)
        self.assertEqual(r["model_calls"], 1)
        self.assertEqual(r["searches"], 0)
        self.assertEqual(r["samples"], [])

    def test_optional_call_cap_is_reported(self):
        b, e, r = search_result(max_calls=1)
        self.assertEqual(r["status"], "call_limit")
        self.assertEqual(r["model_calls"], 1)
        self.assertEqual(r["samples"], [])

    def test_terminal_mc_has_honest_provenance(self):
        b, e, r = search_result(boundary=40)
        terminal = [s for s in r["samples"] if s["mc_source"] == "terminal_answer"]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["n_rollouts"], 0)
        self.assertEqual(terminal[0]["label"], 0)

    def test_same_prefix_is_cached(self):
        b, e, r = search_result()
        prefix = b.calls[1][0]
        before = len(b.calls)
        e.evaluate(prefix)
        self.assertEqual(len(b.calls), before)

    def test_seed_reproducibility_and_change_across_calls(self):
        a, _, _ = search_result()
        b, _, _ = search_result()
        self.assertEqual(a.calls, b.calls)
        self.assertEqual(len(set(seed for _, _, seed in a.calls)), len(a.calls))

    def test_q_u_match_equations(self):
        e = Search(BoundaryBackend(), SearchConfig(), 3)
        r = Rollout("bad", "stop", 500, False, "43", "numeric_exact")
        s = State(0, "", 0.5, visits=3, rollouts=[r])
        e.states[("", "monte_carlo")] = s
        selected = e.select()
        self.assertAlmostEqual(selected[2], 0.5**0.5 * 0.9)
        self.assertAlmostEqual(selected[3], 0.125 * math.sqrt(3) / 4)

    def test_only_selected_state_visit_increments(self):
        b, e, r = search_result()
        self.assertEqual(r["states"][0]["visits"], 1)
        self.assertTrue(all(s["visits"] == 0 for s in r["states"][1:]))

    def test_no_extra_root_resampling(self):
        class AllCorrect(BoundaryBackend):
            def generate(self, problem, prefix, n, seed): return [Completion(GOOD)] * n
        r = Search(AllCorrect(), SearchConfig(), 3).run("q", "q", "42")
        self.assertEqual(r["status"], "root_no_candidates")
        self.assertEqual(r["model_calls"], 1)

    def test_one_token_threshold_cannot_loop_forever(self):
        b, e, r = search_result(threshold=0.5)
        self.assertEqual(r["events"][-1]["boundary_tokens"], 1)


class Adapter(unittest.TestCase):
    def test_chat_prefix_bytes_preserved(self):
        adapter = object.__new__(VLLMCompleter)
        adapter.prompt_template = None
        adapter.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: "<assistant>")
        adapter.prompt_mode = "chat"
        prefix = "  A\nB \n"
        self.assertTrue(adapter.make_prompt("q", prefix).endswith(prefix))

    def test_raw_template_does_not_reexpand_placeholders(self):
        adapter = object.__new__(VLLMCompleter)
        adapter.prompt_template = "Question: {{problem}}\nAnswer: {{prefix}}"
        self.assertEqual(adapter.make_prompt("literal {{prefix}}", "hello"), "Question: literal {{prefix}}\nAnswer: hello")

    def test_token_midpoint_preserves_unicode_and_whitespace(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"offset_mapping": [(0,1), (0,1), (1,2), (2,3)]}
        adapter = object.__new__(VLLMCompleter)
        adapter.tokenizer = Tokenizer()
        left, right = adapter.split_midpoint("中 x")
        self.assertEqual(left + right, "中 x")
        self.assertTrue(left and right)
        self.assertEqual(left, "中")

    def test_vllm_seed_and_finish_reason(self):
        captured = []
        class FakeLLM:
            def generate(self, prompts, params, use_tqdm):
                captured.append((prompts, params))
                return [SimpleNamespace(outputs=[SimpleNamespace(text="cut", finish_reason="length")])]
        a = object.__new__(VLLMCompleter)
        a.llm, a.sampling_class = FakeLLM(), SimpleNamespace
        a.max_tokens, a.max_model_len, a.temperature, a.top_p = 8, 100, 0.8, 0.95
        a.prompt_mode = "chat"
        a.tokenizer = SimpleNamespace(encode=lambda t, **kw: list(range(len(t))))
        a.make_prompt = lambda q,p: q+p
        result = a.generate("q", "prefix", 1, 55)
        self.assertEqual(result[0].finish_reason, "length")
        self.assertEqual(captured[0][1].seed, 55)
        a.max_model_len = 2
        self.assertEqual(a.generate("q", "prefix", 1, 55)[0].finish_reason, "context_length")
        self.assertEqual(len(captured), 1)

    def test_math_verify_errors_are_auditable(self):
        v = object.__new__(MathVerifier)
        class VerificationTimeout(BaseException): pass
        v.handled_errors, v.extraction = (Exception, VerificationTimeout), []
        v.parse = lambda *a, **kw: [object()]
        v.verify = lambda *a, **kw: True
        self.assertIs(v("x+x", r"\boxed{2*x}").correct, True)
        def timeout(*args, **kwargs): raise VerificationTimeout()
        v.verify = timeout
        self.assertIn("math_verify_error", v("x+x", r"\boxed{2*x}").reason)


class CLI(unittest.TestCase):
    def command(self, output, *extra):
        return ["--verifier", "conservative",
                "--problem", str(ROOT / "examples/problems.json"), "--output", str(output),
                "--search", "12", *extra]

    def run_cli(self, args):
        with mock.patch.object(generate_prm_data, "VLLMCompleter", lambda **kw: AdditionBackend()):
            generate_prm_data.main(args)

    def test_end_to_end_screen_calibration_search_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            self.run_cli(self.command(output))
            before = (output / "samples.jsonl").read_bytes()
            records = [json.loads(line) for line in before.splitlines()]
            self.assertGreater(len(records), 0)
            self.assertEqual(len(records), len({r["sample_id"] for r in records}))
            screen_files = list((output / "screening").glob("*.json"))
            old_mtimes = [p.stat().st_mtime_ns for p in screen_files]
            for path in screen_files:
                self.assertEqual(json.loads(path.read_text())["n_rollouts"], 32)
            for path in (output / "problems").glob("*.json"):
                state = json.loads(path.read_text())["states"][0]
                self.assertEqual(len(state["rollouts"]), 8)
            next((output / "problems").glob("*.json")).unlink()
            (output / "samples.jsonl").unlink()
            self.run_cli(self.command(output, "--resume"))
            self.assertEqual(before, (output / "samples.jsonl").read_bytes())
            self.assertEqual(old_mtimes, [p.stat().st_mtime_ns for p in screen_files])
            with self.assertRaisesRegex(ValueError, "changed"):
                self.run_cli(self.command(output, "--resume", "--seed", "9"))
            with self.assertRaises(ValueError):
                self.run_cli(self.command(output))

    def test_subset_limit_and_zero_search(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            self.run_cli(self.command(out, "--limit", "1", "--search", "0"))
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["n_selected"], 1)
            self.assertEqual(summary["screening_model_calls"], 1)
            self.assertEqual(summary["n_samples"], 0)

    def test_invalid_input(self):
        from generate_prm_data import load_problems
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.json"
            for rows in [[{"problem": "q"}], [{"problem": "q", "final_answer": ""}],
                         [{"id": "x", "problem": "q", "final_answer": 1}] * 2]:
                path.write_text(json.dumps(rows))
                with self.assertRaises(ValueError): load_problems(path)

    def test_all_filtered_run_exports_empty_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "questions.json"
            path.write_text(json.dumps([{"id": "q", "problem": "What is 2 + 3?", "final_answer": "999"}]))
            output = Path(temp) / "run"
            self.run_cli(self.command(output) + ["--problem", str(path)])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["n_retained"], 0)
            self.assertEqual(summary["total_sampled_completions"], 32)
            self.assertEqual((output / "samples.jsonl").read_text(), "")


if __name__ == "__main__":
    unittest.main()
