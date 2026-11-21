"""Euclid30K benchmark answer grading."""

import math
import re
from dataclasses import dataclass

from math_verify import LatexExtractionConfig, parse, verify
from sympy import Eq, Expr, Set, Symbol, Tuple
from sympy.core.relational import Relational

PROTOCOL = "euclid30k-v1"
REL_TOL = 1e-6
BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")
FINAL_LINE = re.compile(
    r"^(?:(?:the\s+)?(?:final\s+)?answer|最终答案|答案)(?:\*\*)?(?:\s*[:：是]|\s+is\b)\s*",
    re.IGNORECASE,
)
SIMPLE_LABEL = re.compile(r"[A-D]|yes|no|true|false", re.IGNORECASE)
TEXT = re.compile(r"[A-Za-z]+(?:[ -][A-Za-z]+)*")
LENGTH_UNIT = re.compile(
    r"(?<=[\d}\s])\s*(?:\\\s*)?(?:\\(?:mathrm|text)\{\s*~?\s*)?"
    r"(mm|cm|dm|km|m)\s*\}?\s*(?:\^\s*\{?([23])\}?|([²³]))?$"
)
LATEX = LatexExtractionConfig(boxed_match_priority=0)


@dataclass(frozen=True)
class Grade:
    correct: bool
    answer: str | None
    method: str


def extract_answer(response: str) -> str | None:
    """Extract the last box, an explicit final-answer line, or a bare answer."""
    response = re.split(r"</think\s*>", response, flags=re.IGNORECASE)[-1].strip()
    if re.search(r"<think\b", response, flags=re.IGNORECASE):
        return None

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    for index in reversed(range(len(lines))):
        line = lines[index].strip("*")
        label = FINAL_LINE.match(line)
        if label:
            first = line[label.end() :].lstrip("* ")
            lines = ([first] if first else []) + lines[index + 1 :]
            response = "\n".join(lines)
            break

    boxes = list(BOX.finditer(response))
    if boxes:
        start = boxes[-1].end()
        depth = 1
        for index in range(start, len(response)):
            if response[index] == "{":
                depth += 1
            elif response[index] == "}":
                depth -= 1
                if depth == 0:
                    return response[start:index].strip() or None
        return None

    if len(lines) == 1:
        return lines[0]
    return None


def normalize(answer: str) -> str:
    answer = answer.strip().strip("$").strip()
    for left, right in ((r"\(", r"\)"), (r"\[", r"\]")):
        if answer.startswith(left) and answer.endswith(right):
            answer = answer[len(left) : -len(right)].strip()
    wrapped_text = re.fullmatch(r"\\(?:text|mathrm)\{([^{}]+)\}", answer)
    if wrapped_text and TEXT.fullmatch(wrapped_text[1].strip()):
        answer = wrapped_text[1].strip()
    answer = re.sub(r"(?<=\d)°|\^\s*\{\s*\\circ\s*\}|\^?\\circ\b", "", answer)
    return answer.replace("−", "-").replace("π", r"\pi").replace("，", ",").replace("；", ";").strip()


def normalized_text(answer: str) -> str:
    return " ".join(answer.removesuffix(".").split())


def ordered_parts(answer: str) -> list[str]:
    """Split only top-level separators and preserve thousands separators."""
    parts = []
    start = depth = 0
    for index, char in enumerate(answer):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char in ",;" and answer[index - 1 : index] != "\\":
            if char == "," and index and answer[index - 1].isdigit() and re.match(r"\d{3}\b", answer[index + 1 :]):
                continue
            parts.append(answer[start:index])
            start = index + 1
    return [*parts, answer[start:]]


def length_unit(answer: str):
    unit = LENGTH_UNIT.search(answer)
    if unit:
        return unit[1], unit[2] or {"²": "2", "³": "3"}.get(unit[3], "1")
    return None


def parse_answer(answer: str):
    expressions = parse(
        r"\boxed{" + answer + "}",
        extraction_config=(LATEX,),
        extraction_mode="first_match",
        fallback_mode="no_fallback",
    )
    return expressions[0] if expressions else None


def numeric_expression(expression) -> bool:
    return isinstance(expression, Expr) and expression.is_number and expression.is_real is True and expression.is_finite is True


def compare_answer(answer: str, reference: str) -> Grade:
    prediction, gold = normalize(answer), normalize(reference)
    pred_text, gold_text = normalized_text(prediction), normalized_text(gold)
    if not pred_text or not gold_text:
        return Grade(False, answer, "empty_answer")
    if SIMPLE_LABEL.fullmatch(gold_text):
        return Grade(pred_text.casefold() == gold_text.casefold(), answer, "label")
    if pred_text == gold_text:
        return Grade(True, answer, "text_match")

    expected_parts, submitted_parts = ordered_parts(gold), ordered_parts(prediction)
    if len(expected_parts) > 1 or len(submitted_parts) > 1:
        correct = len(expected_parts) == len(submitted_parts) and all(compare_answer(pred, ref).correct for pred, ref in zip(submitted_parts, expected_parts))
        return Grade(correct, answer, "ordered")
    if TEXT.fullmatch(gold_text):
        correct = " " in gold_text and pred_text.casefold() == gold_text.casefold()
        return Grade(correct, answer, "text_match")

    expected_unit, submitted_unit = length_unit(gold), length_unit(prediction)
    if expected_unit and submitted_unit and expected_unit != submitted_unit:
        return Grade(False, answer, "unit_mismatch")
    expected, submitted = parse_answer(gold), parse_answer(prediction)
    if expected is None or submitted is None:
        return Grade(False, answer, "unparsed")
    if isinstance(expected, Expr) and expected.is_number:
        expected = expected.doit()
    if isinstance(submitted, Eq) and isinstance(submitted.lhs, Symbol) and numeric_expression(expected):
        submitted = submitted.rhs
    if isinstance(submitted, Expr) and submitted.is_number:
        submitted = submitted.doit()
    # Math-Verify can compare a general equation's RHS or a singleton set with a scalar.
    if isinstance(submitted, (Relational, Set, Tuple)) and numeric_expression(expected):
        return Grade(False, answer, "mismatch")
    if numeric_expression(expected) and numeric_expression(submitted):
        if expected.is_integer is True or (expected.is_Float and (expected % 1).is_zero):
            correct = bool(expected.equals(submitted))
        else:
            expected_value, submitted_value = float(expected), float(submitted)
            correct = math.isfinite(expected_value) and math.isclose(expected_value, submitted_value, rel_tol=REL_TOL)
        return Grade(correct, answer, "numeric")
    return Grade(bool(verify(expected, submitted, strict=True)), answer, "symbolic")


def grade_answer(response: str, reference: str) -> Grade:
    answer = extract_answer(response)
    if answer is None:
        return Grade(False, None, "no_final_answer")
    return compare_answer(answer, reference)
