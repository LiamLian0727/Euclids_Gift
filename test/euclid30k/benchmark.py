"""Run the Euclid30K benchmark against vLLM or score saved submissions."""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import re

from huggingface_hub import hf_hub_download
from PIL import Image
from pyarrow.parquet import ParquetFile

from grader import PROTOCOL, grade_answer
from prompt import PROMPT_SUFFIX

DATASET = "LiamLian0727/Euclid30K"
DATA_FILES = {"train": "Euclid30K_train.parquet", "validation": "Euclid30K_val.parquet"}
MAX_TOKENS = 4096
NUMBERED_IMAGE = re.compile(r"<image\s*(\d+)>", re.IGNORECASE)


def dataset_file(split):
    return ParquetFile(hf_hub_download(DATASET, DATA_FILES[split], repo_type="dataset"))


def image_content(image):
    image_bytes = image["bytes"]
    with Image.open(BytesIO(image_bytes)) as decoded:
        mime = decoded.get_format_mimetype()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def messages(problem, images):
    # Some training questions point to numbered panels within a single supplied image.
    parts = NUMBERED_IMAGE.sub(r"(image \1)", problem).split("<image>")
    if len(parts) - 1 != len(images):
        raise ValueError("The number of <image> placeholders differs from the number of images.")
    content = []
    for text, image in zip(parts, images):
        if text.strip():
            content.append({"type": "text", "text": text.strip()})
        content.append(image_content(image))
    content.append({"type": "text", "text": f"{parts[-1].rstrip()}\n\n{PROMPT_SUFFIX}".lstrip()})
    return [{"role": "user", "content": content}]


def sample_id(split, index):
    return f"{split}:{index}"


def write_jsonl(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate(args):
    from openai import OpenAI

    source = dataset_file(args.split)
    count = min(source.metadata.num_rows, args.limit or source.metadata.num_rows)
    sampling = {"temperature": 0, "top_p": 1, "max_tokens": MAX_TOKENS, "seed": args.seed}
    args.output.mkdir(parents=True, exist_ok=False)

    def infer(item):
        index, row = item
        completion = client.chat.completions.create(
            model=args.model, messages=messages(row["problem"], row["images"]), **sampling
        )
        choice = completion.choices[0]
        return {
            "id": sample_id(args.split, index),
            "response": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "completion_tokens": completion.usage.completion_tokens if completion.usage else None,
        }

    with (
        OpenAI(base_url=args.base_url, api_key=os.environ.get("VLLM_API_KEY") or "EMPTY", timeout=args.timeout) as client,
        ThreadPoolExecutor(max_workers=args.concurrency) as pool,
        (args.output / "predictions.jsonl").open("x", encoding="utf-8") as out,
    ):
        (args.output / "run.json").write_text(json.dumps({
            "protocol": PROTOCOL,
            "dataset": DATASET,
            "split": args.split,
            "samples": count,
            "model": args.model,
            "prompt_suffix": PROMPT_SUFFIX,
            "sampling": sampling,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        start = 0
        for batch in source.iter_batches(batch_size=args.concurrency, columns=["problem", "images"]):
            rows = batch.slice(0, count - start).to_pylist()
            for prediction in pool.map(infer, enumerate(rows, start)):
                write_jsonl(out, prediction)
            out.flush()
            start += len(rows)
            if start == count:
                break
    print(f"Saved {count} predictions to {args.output / 'predictions.jsonl'}")


def read_predictions(path):
    predictions = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            identifier = row["id"]
            if identifier in predictions:
                raise ValueError(f"Duplicate prediction ID {identifier!r} on line {line_number}.")
            if not isinstance(row["response"], str):
                raise ValueError(f"Prediction {identifier!r} must have a string response.")
            predictions[identifier] = row
    return predictions


def score(args):
    predictions = read_predictions(args.predictions)
    source = dataset_file(args.split)
    answers = source.read(columns=["answer"])["answer"].to_pylist()
    answers = answers[:args.limit]
    identifiers = [sample_id(args.split, index) for index in range(len(answers))]
    expected = set(identifiers)
    missing, extra = expected.difference(predictions), predictions.keys() - expected
    if missing or extra:
        raise ValueError(f"Wrong prediction coverage: {len(missing)} missing, {len(extra)} unexpected IDs.")

    methods = Counter()
    correct = 0
    truncated = 0
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "scores.jsonl").open("x", encoding="utf-8") as out:
        for identifier, answer in zip(identifiers, answers):
            prediction = predictions[identifier]
            result = grade_answer(prediction["response"], answer)
            finish_reason = prediction.get("finish_reason")
            truncated += finish_reason == "length"
            correct += result.correct
            methods[result.method] += 1
            write_jsonl(out, {
                "id": identifier,
                "reference": answer,
                "extracted_answer": result.answer,
                "correct": result.correct,
                "method": result.method,
                "finish_reason": finish_reason,
            })
    summary = {
        "protocol": PROTOCOL,
        "math_verify_version": version("math-verify"),
        "dataset": DATASET,
        "split": args.split,
        "complete_split": len(answers) == source.metadata.num_rows,
        "samples": len(answers),
        "correct": correct,
        "accuracy": correct / len(answers),
        "length_stops_reported": truncated,
        "grading_methods": dict(methods),
    }
    output = json.dumps(summary, ensure_ascii=False, indent=2)
    (args.output / "summary.json").write_text(output + "\n", encoding="utf-8")
    print(output)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate", help="Generate responses through a local vLLM server.")
    generate_parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    generate_parser.add_argument("--model", required=True, help="Model name served by vLLM.")
    generate_parser.add_argument("--concurrency", type=positive_int, default=8)
    generate_parser.add_argument("--timeout", type=positive_int, default=300)
    generate_parser.add_argument("--seed", type=int, default=0)
    score_parser = commands.add_parser("score", help="Score vLLM predictions or external agent submissions.")
    score_parser.add_argument("--predictions", type=Path, required=True)
    for command in (generate_parser, score_parser):
        command.add_argument("--split", choices=tuple(DATA_FILES), default="validation")
        command.add_argument("--limit", type=positive_int, help="Use the first N rows for a smoke test.")
        command.add_argument("--output", type=Path, required=True, help="New output directory.")
    args = parser.parse_args()
    (generate if args.command == "generate" else score)(args)


if __name__ == "__main__":
    main()
