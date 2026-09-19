"""OmegaPRM §3.2–3.4: Q+U selection, divide-and-rollout, explicit edges.

Operational choices not fixed by the paper are documented in ALIGNMENT.md.
"""
import hashlib
import math
import random
from dataclasses import asdict
from structure import State
from sampling import sample_rollouts
from utils import verify_answer


class CallBudgetReached(Exception):
    pass


class Search:
    def __init__(self, completer, config, step_token_threshold, seed=1234, verifier=verify_answer):
        if not math.isfinite(step_token_threshold) or step_token_threshold <= 0:
            raise ValueError("step_token_threshold must be finite and positive")
        self.backend, self.config, self.verifier = completer, config, verifier
        self.threshold = step_token_threshold
        self.seed, self.rng = seed, random.Random(seed)
        self.states = {}
        self.edges = {}
        self.calls = self.searches = 0
        self.events = []

    def evaluate(self, prefix):
        key = (prefix, "monte_carlo")
        if key in self.states:
            return self.states[key]
        if self.config.max_calls is not None and self.calls >= self.config.max_calls:
            raise CallBudgetReached
        seed = self.rng.randrange(2**31)
        self.calls += 1
        rollouts = sample_rollouts(self.backend, self.problem, prefix, self.answer,
                                   self.config.rollouts, seed, self.config, self.verifier)
        state = State(len(self.states), prefix, sum(r.correct for r in rollouts) / len(rollouts),
                      rollouts=rollouts, sampling_seed=seed)
        self.states[key] = state
        return state

    def terminal(self, prefix):
        # The selected rollout's final answer was already checked to be wrong.
        # Terminal states are absorbing: no further continuation is sampled.
        key = (prefix, "terminal_answer")
        if key not in self.states:
            self.states[key] = State(len(self.states), prefix, 0.0, mc_source="terminal_answer")
        return self.states[key]

    def select(self):
        total_visits = sum(s.visits for s in self.states.values())
        best, best_score = None, -math.inf
        for state in self.states.values():
            if state.mc_source != "monte_carlo" or not 0 < state.mc < 1:
                continue
            for index, rollout in enumerate(state.rollouts):
                if rollout.visited or rollout.correct:
                    continue
                q = self.config.alpha ** (1 - state.mc) * self.config.beta ** (
                    rollout.token_count / self.config.length_scale)
                u = self.config.c_puct * math.sqrt(total_visits) / (1 + state.visits)
                if q + u > best_score:
                    best, best_score = (state, index, q, u), q + u
        return best

    def is_single_step(self, action):
        n = self.backend.count_tokens(action)
        return n < self.threshold or n <= 1 or self.backend.split_midpoint(action) is None

    def add_edge(self, parent, child):
        if not child.prefix.startswith(parent.prefix) or len(child.prefix) <= len(parent.prefix):
            raise AssertionError("a tree edge must append a nonempty exact text span")
        action = child.prefix[len(parent.prefix):]
        key = (parent.id, child.id)
        parent.in_tree = child.in_tree = True
        self.edges[key] = {"parent_id": parent.id, "child_id": child.id,
                           "action": action, "action_tokens": self.backend.count_tokens(action),
                           "single_step": self.is_single_step(action), "mc": child.mc}

    def locate(self, source, rollout):
        if not rollout.text:
            self.events.append({"status": "empty_wrong_rollout", "source_id": source.id})
            return
        low = source
        high = self.terminal(source.prefix + rollout.text)
        positive_path = [source]
        while True:
            span = high.prefix[len(low.prefix):]
            if self.backend.count_tokens(span) < self.threshold:
                break
            halves = self.backend.split_midpoint(span)
            if halves is None:
                break
            left, right = halves
            if not left or not right or left + right != span:
                raise AssertionError("split_midpoint must return two nonempty lossless spans")
            midpoint = self.evaluate(low.prefix + left)
            if midpoint.mc > 0:  # Includes MC=1; do not stop early.
                low = midpoint
                positive_path.append(low)
            else:
                high = midpoint
        # Explicit path contains measured positive cut points and the first
        # localized zero boundary, not discarded later-zero probes.
        path = positive_path + [high]
        for parent, child in zip(path, path[1:]):
            self.add_edge(parent, child)
        self.events.append({"status": "boundary_found", "source_id": source.id,
                            "path_state_ids": [s.id for s in path],
                            "positive_state_id": low.id, "negative_state_id": high.id,
                            "boundary_tokens": self.backend.count_tokens(high.prefix[len(low.prefix):])})

    def samples(self):
        by_id = {s.id: s for s in self.states.values()}
        rows = []
        for edge in self.edges.values():
            if not edge["single_step"]:
                continue
            parent, child = by_id[edge["parent_id"]], by_id[edge["child_id"]]
            identity = "\0".join([self.problem_id, parent.prefix, child.prefix, child.mc_source])
            rows.append({"schema_version": 2,
                         "sample_id": hashlib.sha256(identity.encode()).hexdigest(),
                         "problem_id": self.problem_id, "question": self.problem,
                         "gold_answer": self.answer, "parent_id": parent.id, "child_id": child.id,
                         "parent_prefix": parent.prefix, "step": edge["action"], "prefix": child.prefix,
                         "step_tokens": edge["action_tokens"], "label": child.mc,
                         "hard_label": int(child.mc > 0), "label_kind": "soft_mc",
                         "mc_source": child.mc_source, "n_rollouts": len(child.rollouts),
                         "n_correct": sum(r.correct for r in child.rollouts),
                         "sampling_seed": child.sampling_seed})
        return rows

    def run(self, problem_id, problem, answer):
        if self.states:
            raise RuntimeError("create one Search per problem")
        self.problem_id, self.problem, self.answer = problem_id, problem, answer
        root = self.evaluate("")
        root.in_tree = True
        status = "search_limit"
        # The separate 32-rollout screening can retain a question whose NEW
        # 8-rollout root estimate is all correct/wrong. Do not resample it silently.
        if root.mc == 0 or root.mc == 1:
            status = "root_no_candidates"
        else:
            for _ in range(self.config.searches):
                selected = self.select()
                if selected is None:
                    status = "candidates_exhausted"
                    break
                state, index, q, u = selected
                state.rollouts[index].visited = True
                state.visits += 1
                self.searches += 1
                self.events.append({"status": "selected", "state_id": state.id,
                                    "rollout_index": index, "Q": q, "U": u})
                try:
                    self.locate(state, state.rollouts[index])
                except CallBudgetReached:
                    status = "call_limit"
                    break
        # Prefix caching can merge identical nodes reached through different
        # parents. The exported object is a prefix DAG (a deduplicated tree).
        return {"schema_version": 2, "problem_id": problem_id, "question": problem,
                "gold_answer": answer, "seed": self.seed, "config": asdict(self.config),
                "step_token_threshold": self.threshold, "status": status,
                "model_calls": self.calls, "searches": self.searches,
                "sampled_completions": sum(len(s.rollouts) for s in self.states.values()),
                "generated_text_tokens": sum(r.token_count for s in self.states.values() for r in s.rollouts),
                "unknown_as_incorrect": sum(r.unknown_mapped_to_incorrect for s in self.states.values() for r in s.rollouts),
                "samples": self.samples(), "states": [asdict(s) for s in self.states.values()],
                "edges": list(self.edges.values()), "events": self.events}
