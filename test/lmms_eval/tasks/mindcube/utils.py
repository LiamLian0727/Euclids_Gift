import datetime
import json
import os
import re
from collections import defaultdict
import pandas as pd
from loguru import logger as eval_logger

import numpy as np
from functools import partial
from mathruler.grader import extract_boxed_content, grade_answer

# This is the prompt that might exist in original data and needs to be replaced
replace_prompt = ""  # If there's no specific prompt to replace, keep it empty

#  features: ['image', 'image_filename', 'question', 'answer', 'question_index', 'program', 'source']

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

MRA = partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)

def mindcube_doc_to_visual(doc):
    return [i.convert("RGB") for i in doc["images"] ]

def mindcube_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["problem"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question

def mindcube_process_results(doc, results):
    """
    Process results for mindcube.

    Args:
        doc: Document containing the ground truth answer in doc[answer"] (XML format)
        results: List containing model predictions (simple format: single word/number/Yes/No)

    Returns:
        Dict with exact_match score for this sample
    """

    ground_truth = doc["answer"]
    
    response = results[0] if results else ""
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)  # handle qwen2.5vl-32b format
    answer = extract_boxed_content(response)
    if answer == "" or answer is None or answer == "None": # answer not in \\boxed{}, try <answer></answer>
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.IGNORECASE | re.DOTALL)
        answer = "" if not match else match.group(1).strip()
        if answer:                                    
            first_char = answer[0].upper()            
            if first_char in "ABCDEFGH":
                answer = first_char                   
        else:
            answer = response   
        
    score = 0.0
    if "." in ground_truth:
        try:
            gt_float = float(ground_truth)
            answer_float = float(answer)
            score = MRA(answer_float, gt_float)
        except ValueError:
            pass

    score =  1.0 if grade_answer(answer, ground_truth) else score
    if score == 0:
        eval_logger.debug(f"Question: {doc['problem'].strip()}, GT : {ground_truth}, Answer: {answer}, Exact Match: {score}, Source: {doc['id']}")

    return {
        "exact_match": {"score": score, "source": doc["id"]},
    }
    
def mindcube_aggregate_results(results):
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby("source").groups.items():
        per_question_type = results.iloc[question_type_indexes]

        output[f"{question_type}"] = per_question_type["score"].mean()

    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output
