# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List
import math
from sympy import Expr

from mathruler.grader import extract_boxed_content, grade_answer

try:
    from math_verify import parse 
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def _to_float(expr):
    if isinstance(expr, Expr):
        return float(expr.evalf())
    return float(expr)

def accuracy_reward_ref(pred: str, gt: str, rel_tol: float = 0.01) -> float:
    """Compare two LaTeX strings for mathematical equivalence."""
    
    try:
        p_val = float(pred)
        g_val = float(gt)
        return math.isclose(p_val, g_val, rel_tol=rel_tol)
    except (Exception):
        pass
    
    try:
        if not isinstance(pred, str):
            pred = str(pred)
        if not isinstance(gt, str):
            gt = str(gt)

        gt = "\\boxed{" + gt + "}"

        p_exprs = parse(pred)
        g_exprs = parse(gt)

        p = p_exprs[0] if p_exprs else None
        g = g_exprs[0] if g_exprs else None
        if p is None or g is None:
            return 0

        p_val, g_val = _to_float(p), _to_float(g)
        return math.isclose(p_val, g_val, rel_tol=rel_tol)

    except Exception:
        return 0
    
    return 0

def accuracy_reward_math(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:
    
    if not isinstance(model_output, str):
        model_output = str(model_output)
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        if accuracy_score == 0:
            accuracy_score = accuracy_reward_math(response, reward_input["ground_truth"])
        if accuracy_score == 0:
            accuracy_score = accuracy_reward_ref(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores

