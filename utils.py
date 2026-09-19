"""Answer verification, independent from binary-label failure policies.

Numeric arithmetic is evaluated with a bounded AST interpreter, never eval().
Unsupported equivalence is UNKNOWN, not an automatic negative label.
"""
import ast
import re
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Verdict:
    correct: bool | None
    extracted: str | None
    reason: str


def boxed_answer(text: str) -> str | None:
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos].strip()
    return None


def extract_answer(text: str) -> str | None:
    boxes = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    markers = list(re.finditer(r"(?:####|(?:final\s+)?answer\s*(?:is|:|=))\s*([^\n]+)", text, re.I))
    if boxes and (not markers or boxes[-1].start() >= markers[-1].start()):
        return boxed_answer(text)
    if markers:
        return markers[-1].group(1).strip().rstrip(".")
    # Plain numeric expressions are allowed only if the ENTIRE response is one.
    candidate = text.strip().rstrip(".")
    return candidate if numeric_value(candidate) is not None else None


def normalize(text: str) -> str:
    text = text.strip().replace("−", "-").replace("\\left", "").replace("\\right", "")
    for left, right in [("\\(", "\\)"), ("\\[", "\\]"), ("$", "$")]:
        if text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)].strip()
    text = text.replace("\\,", "").replace("\\!", "").replace("\\ ", " ")
    return re.sub(r"\s+", "", text)


def numeric_value(text: str) -> Fraction | None:
    """Exact rationals: decimals, scientific notation, fractions, + - * / **.

    No variables, function calls, implicit multiplication, units or general LaTeX.
    Size/depth/exponent limits keep model-generated input bounded.
    """
    if len(text) > 512 or re.search(r"\d\s+\d", text):
        return None
    text = normalize(text)
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    for _ in range(16):
        changed = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", text)
        if changed == text:
            break
        text = changed
    text = text.replace(r"\times", "*").replace(r"\cdot", "*").replace("^", "**")
    # Commas only mean thousands separators when the full value has that shape.
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    if text.endswith(r"\%"):
        text = "(" + text[:-2] + ")/100"
    elif text.endswith("%"):
        text = "(" + text[:-1] + ")/100"
    if not text or not re.fullmatch(r"[0-9eE+*/().\-]+", text):
        return None
    try:
        tree = ast.parse(text, mode="eval")
        if len(list(ast.walk(tree))) > 100:
            return None

        def visit(node, depth=0):
            if depth > 20:
                raise ValueError("expression too deep")
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                literal = ast.get_source_segment(text, node)
                # Limit decimal exponents before constructing a large Fraction.
                exponent = re.search(r"[eE]([+-]?\d+)$", literal)
                if exponent and abs(int(exponent.group(1))) > 100:
                    raise ValueError("exponent too large")
                result = Fraction(literal)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                result = visit(node.operand, depth + 1)
                if isinstance(node.op, ast.USub):
                    result = -result
            elif isinstance(node, ast.BinOp):
                a, b = visit(node.left, depth + 1), visit(node.right, depth + 1)
                if isinstance(node.op, ast.Add): result = a + b
                elif isinstance(node.op, ast.Sub): result = a - b
                elif isinstance(node.op, ast.Mult): result = a * b
                elif isinstance(node.op, ast.Div): result = a / b
                elif isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 12:
                    result = a ** int(b)
                else: raise ValueError("unsupported operator")
            else:
                raise ValueError("unsupported expression")
            if max(result.numerator.bit_length(), result.denominator.bit_length()) > 4096:
                raise ValueError("value too large")
            return result

        return visit(tree.body)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def verify_answer(gold: str, full_response: str, finish_reason: str = "stop") -> Verdict:
    if finish_reason != "stop":
        return Verdict(None, None, "incomplete:" + finish_reason)
    candidate = extract_answer(full_response)
    if candidate is None or not candidate.strip():
        return Verdict(None, candidate, "missing_final_answer")
    reference = boxed_answer(gold)
    reference = gold if reference is None else reference
    if not reference.strip():
        raise ValueError("empty gold answer")
    if re.search(r"\d\s+\d", reference) or re.search(r"\d\s+\d", candidate):
        return Verdict(None, candidate, "ambiguous_numeric_whitespace")
    a, b = numeric_value(reference), numeric_value(candidate)
    if a is not None and b is not None:
        return Verdict(a == b, candidate, "numeric_exact")
    if normalize(reference) == normalize(candidate):
        return Verdict(True, candidate, "normalized_exact")
    return Verdict(None, candidate, "unsupported_equivalence")


class MathVerifier:
    """Optional Math-Verify adapter for symbolic/LaTeX answers, with timeouts.

    First use exact built-in checks. Parse ONLY an explicitly extracted final
    answer, never an arbitrary number from the reasoning trace. Unparseable or
    timed-out comparisons are UNKNOWN. A parsed non-equivalence is negative.
    """
    def __init__(self):
        from math_verify import LatexExtractionConfig, parse, verify
        from math_verify.errors import TimeoutException
        from inspect import signature
        if "raise_on_error" not in signature(parse).parameters or "raise_on_error" not in signature(verify).parameters:
            raise ImportError("update math-verify: parse/verify must support raise_on_error")
        self.parse, self.verify = parse, verify
        self.extraction = [LatexExtractionConfig()]
        self.handled_errors = (Exception, TimeoutException)

    def __call__(self, gold, full_response, finish_reason="stop"):
        initial = verify_answer(gold, full_response, finish_reason)
        if initial.correct is not None or initial.reason != "unsupported_equivalence":
            return initial
        reference = boxed_answer(gold)
        reference = gold if reference is None else reference
        try:
            kwargs = dict(extraction_config=self.extraction, fallback_mode="no_fallback",
                          extraction_mode="first_match", parsing_timeout=5, raise_on_error=True)
            parsed_gold = self.parse("$" + normalize(reference) + "$", **kwargs)
            parsed_answer = self.parse("$" + normalize(initial.extracted) + "$", **kwargs)
            if not parsed_gold or not parsed_answer:
                return Verdict(None, initial.extracted, "math_verify_unparseable")
            correct = self.verify(parsed_gold, parsed_answer, timeout_seconds=5, raise_on_error=True)
            return Verdict(bool(correct), initial.extracted, "math_verify")
        except self.handled_errors as exc:
            return Verdict(None, initial.extracted, "math_verify_error:" + type(exc).__name__)
