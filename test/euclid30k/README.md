# Euclid30K benchmark

Euclid30K Benchmark v1 evaluates the 500 public validation examples in
[Euclid30K](https://huggingface.co/datasets/LiamLian0727/Euclid30K). The runner
downloads the selected Parquet file from the dataset's default branch.
Use `validation` for benchmark reporting. The standalone runner also supports
the 29,195-example `train` split for diagnostics. Label scores from checkpoints
exposed to Euclid30K validation during model selection or tuning accordingly.

## Run a local vLLM baseline

From the repository root, install the client and scorer. Install vLLM in a
separate environment with a [supported GPU and model](https://docs.vllm.ai/en/latest/features/multimodal_inputs/).

```bash
python -m pip install -r test/euclid30k/requirements.txt
vllm serve Qwen/Qwen3-VL-4B-Instruct --generation-config vllm
```

In another terminal, run inference and grade the saved responses:

```bash
python test/euclid30k/benchmark.py generate \
  --model Qwen/Qwen3-VL-4B-Instruct --output runs/euclid30k-qwen3vl-4b
python test/euclid30k/benchmark.py score \
  --predictions runs/euclid30k-qwen3vl-4b/predictions.jsonl \
  --output runs/euclid30k-qwen3vl-4b-scores
```

The server defaults to `http://127.0.0.1:8000/v1`. To use a different
endpoint, pass `--base-url`. If serving with an API key, export
`VLLM_API_KEY` in the client environment. Pass `--limit 4` to both commands
for a smoke test, or `--split train` to both for training-set diagnostics.
Use fresh output directories for each run.
Some training examples contain up to eight images. If your server sets a
per-request image limit or a context limit, size it for those inputs.

The baseline sends the original problem and images with a short final-answer
instruction. Generation uses temperature 0, top-p 1, seed 0 and a 4,096-token
output cap. `run.json` records the generation settings. Record the model
revision, vLLM version, and any changes to the server's image processing or
chat template when reporting a score.

## Grade an agent or an existing run

Submit one JSON object per line with an `id` and a string `response`.
IDs use the zero-based Parquet row order, for example `validation:0`.
Use the agent's submitted final answer as `response`. The same scorer also
extracts a final answer from model output. `score` checks that the IDs cover
the selected rows exactly, so omitting failed generations cannot raise the
reported accuracy. Record a completed trial with no final answer as an empty
string. The scorer writes per-example decisions to `scores.jsonl` and overall
accuracy to `summary.json`.

### Answer rules

- Grade the last `\boxed{...}` or `\fbox{...}` in the final answer section.
  The scorer discards closed reasoning blocks and uses the section starting at
  the last explicit final-answer line, when present. A bare answer also works.
- Normalize presentation before comparing answers. For example, `30°` and
  `30^{\circ}` represent the same degree value. If both answers include a
  recognized metric length, area or volume unit, the units must match.
- Compare choice letters and simple labels directly. Compare top-level comma-
  or semicolon-separated answers in order, with each part graded independently.
- Use Math-Verify for symbolic equivalence. Finite scalar integers require an
  exact value. Other finite scalars use a relative tolerance of `1e-6`, with
  zero absolute tolerance. A missing or ungradable final answer scores zero.
  Accuracy is the fraction correct over the selected rows.

For a comparison, use the same dataset files, answer rules and denominator
for every method. Report model-only and agent inference budgets alongside
accuracy, including any tool, critic or retry calls. Audit grading changes on
saved outputs and use a versioned protocol for published scores.

The scorer uses the published reference strings. In `validation:281`, the
question lets the solver choose one of three conditions, while the reference
lists the answers for all three. The ordered-answer rule can mark a valid
single-condition answer incorrect on this item. Record it during error review.

[Visual Board §4.1](https://arxiv.org/html/2611.23872v1#S4.SS1) reports
500 held-out Euclid30K questions, an output cap of 4,096 tokens for single-model
baselines, and symbolic/numeric equivalence for geometry answers. The paper does
not give the Euclid30K row IDs, exact extraction rules or numeric tolerance.
Euclid30K Benchmark v1 defines its own reproducible protocol.
