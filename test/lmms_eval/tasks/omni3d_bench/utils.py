import numpy as np
import datetime
import json
import os
import re
from collections import defaultdict
from functools import partial
from loguru import logger as eval_logger

from mathruler.grader import extract_boxed_content, grade_answer

# This is the prompt that might exist in original data and needs to be replaced
replace_prompt = ""  # If there's no specific prompt to replace, keep it empty

def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

MRA = partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)

def omni3d_bench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def omni3d_bench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question

def omni3d_bench_process_results(doc, results):
    """
    Process results for Omni Bench.

    Args:
        doc: Document containing the ground truth answer in doc[answer"] (XML format)
        results: List containing model predictions (simple format: single word/number/Yes/No)

    Returns:
        Dict with exact_match score for this sample
    """

    ground_truth = doc["answer"]
    # match = re.search(r'<answer>\s*(.*?)\s*</answer>', ground_truth, re.IGNORECASE | re.DOTALL)
    # ground_truth = "" if not match else match.group(1).strip()

    response = results[0] if results else ""
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)  # handle qwen2.5vl-32b format
    answer = extract_boxed_content(response)
    if answer == "" or answer is None or answer == "None": # answer not in \\boxed{}, try <answer></answer>
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.IGNORECASE | re.DOTALL)
        answer = "" if not match else match.group(1).strip()
    
    
    exact_match = 0.0
    if "." in ground_truth:
        try:
            gt_float = float(ground_truth)
            answer_float = float(answer)
            exact_match = MRA(answer_float, gt_float)
        except ValueError:
            pass
    
    exact_match =  1.0 if grade_answer(answer, ground_truth) else exact_match
    if exact_match != 1.0:
        eval_logger.debug(f"Question: {doc['question'].strip()}, GT : {ground_truth}, Answer: {answer}, Exact Match: {exact_match}")

    return {
        "exact_match": exact_match
    }
