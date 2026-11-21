# Euclid30K benchmark

Euclid30K Benchmark evaluates the 500 public validation examples in
[Euclid30K](https://huggingface.co/datasets/LiamLian0727/Euclid30K). The runner
downloads the selected Parquet file from the dataset's default branch.
Use `validation` for benchmark reporting. The standalone runner also supports
the 29,195-example `train` split for diagnostics. 

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

## Answer rules

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

