"""Euclid30K benchmark answer grading."""

import math
import re
from dataclasses import dataclass

from math_verify import LatexExtractionConfig, parse, verify
from sympy import Eq, Expr, Set, Symbol, Tuple
from sympy.core.relational import Relational

PROTOCOL = "euclid30k-v1.1"
REL_TOL = 1e-6
BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")
BOX_SEPARATOR = re.compile(r"[\s$]*(?:[,;，；]|and|\\text\{\s*and\s*\})[\s$]*", re.IGNORECASE)
FINAL_LINE = re.compile(
    r"^(?:(?:the\s+)?(?:final\s+)?answer|最终答案|答案)(?:\*\*)?(?:\s*[:：是]|\s+is\b)\s*",
    re.IGNORECASE,
)
SIMPLE_LABEL = re.compile(r"[A-D]|yes|no|true|false", re.IGNORECASE)
TEXT = re.compile(r"[A-Za-z]+(?:[ -][A-Za-z]+)*")
LENGTH_UNIT = re.compile(
    r"(?<=[\d}\s])\s*(?:\\\s*)?(?:\\(?:mathrm|text)\{\s*~?\s*)?"
    r"(mm|cm|dm|km|m|inches|inch|in|feet|foot|ft)\s*\}?\s*(?:\^\s*\{?([23])\}?|([²³]))?$"
)
LATEX = LatexExtractionConfig(boxed_match_priority=0)


@dataclass(frozen=True)
class Grade:
    correct: bool
    answer: str | None
    method: str


def box_content(response: str, start: int) -> tuple[str | None, int]:
    depth = 1
    for index in range(start, len(response)):
        if response[index] == "{":
            depth += 1
        elif response[index] == "}":
            depth -= 1
            if depth == 0:
                return response[start:index].strip() or None, index + 1
    return None, len(response)


def extract_answer(response: str, *, multipart: bool = False) -> str | None:
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
        last = boxes[-1]
        answer, _ = box_content(response, last.end())
        if not multipart or answer is None:
            return answer
        # Multi-part answers can end with a group of independently boxed values.
        parts = [answer]
        for previous in reversed(boxes[:-1]):
            answer, end = box_content(response, previous.end())
            if answer is None or end > last.start() or not BOX_SEPARATOR.fullmatch(response[end : last.start()]):
                break
            parts.append(answer)
            last = previous
        return "; ".join(reversed(parts))

    if len(lines) == 1:
        return lines[0]
    return None


def normalize(answer: str) -> str:
    answer = re.sub(r"^(?:volume|surface area|s\.a\.)\s*\$?\s*=\s*", "", answer.strip(), flags=re.IGNORECASE)
    answer = answer.removesuffix(".").rstrip().strip("$").strip()
    for left, right in ((r"\(", r"\)"), (r"\[", r"\]")):
        if answer.startswith(left) and answer.endswith(right):
            answer = answer[len(left) : -len(right)].strip()
    wrapped_text = re.fullmatch(r"\\(?:text|mathrm)\{([^{}]+)\}", answer)
    if wrapped_text and TEXT.fullmatch(wrapped_text[1].strip()):
        answer = wrapped_text[1].strip()
    answer = re.sub(r"\\number\{([^{}]+)\}", r"\1", answer)
    answer = re.sub(r"\\\\(?=pi\b|\s+\\(?:mathrm|text)\{)", r"\\", answer)
    answer = re.sub(r"\\(?:left|right)\b\s*", "", answer)
    answer = re.sub(r"\\[lr]vert\b", "|", answer)
    answer = re.sub(r"(?:\{\}\s*)?\^\s*(?:\{\s*\\circ\s*\}|\\circ\b)|(?<=\d)\s*[°º]|\\circ\b", "", answer)
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
        symbol = {"inch": "in", "inches": "in", "foot": "ft", "feet": "ft"}.get(unit[1], unit[1])
        return symbol, unit[2] or {"²": "2", "³": "3"}.get(unit[3], "1")
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

    # One published reference ends in an extraneous multiplication sign.
    gold = re.sub(r"\\cdot\s*$", "", gold).strip()
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
    answer = extract_answer(response, multipart=len(ordered_parts(normalize(reference))) > 1)
    if answer is None:
        return Grade(False, None, "no_final_answer")
    return compare_answer(answer, reference)
