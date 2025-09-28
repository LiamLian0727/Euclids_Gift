# Euclid’s Gift: Enhancing Spatial Perception and Reasoning in Vision‑Language Models via Geometric Surrogate Tasks

## Abstract
Spatial intelligence spans abilities such as visualizing and transforming shapes, mental rotation, reasoning about relative positions and containment, and counting/estimation. These remain challenging for modern Multimodal Large Language Models (MLLMs). We propose solving Euclidean geometry problems as a surrogate task and construct Euclid30K, a dataset of roughly 30K 2D and 3D geometry questions. We then fine‑tune Qwen2.5‑VL and RoboBrain2.0 models with Group Relative Policy Optimization (GRPO), enabling the models to internalize and apply Euclidean principles for shape recognition, counting, relation extraction, and multi‑step deductive reasoning. Without task‑specific adaptations, our models achieve significant zero‑shot gains on four spatial‑reasoning benchmarks: Super‑CLEVR, Omni3DBench, VSI‑Bench, and MindCube. For example, on VSI‑Bench, average accuracy improves from 34.5% to 40.5% (+5.5 percentage points); RoboBrain2.0‑Euclid‑7B reaches 49.6%, surpassing the previous SOTA (Spatial‑MLLM).

![Architecture](assert/arch.png)

![Gain](assert/gain.png)

## Quick Start

### 1) Environment Setup
Training
- Install [EasyR1](https://github.com/hiyouga/EasyR1) following the official documentation.
- Install the required Python dependencies: `pip install -r requirements.txt`.

Evaluation
- Install [lmms‑eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) following its official documentation. You can either:
  - Use the [`lmms-eval/`](https://github.com/EvolvingLMMs-Lab/lmms-eval) copy included in this repository; or
  - Copy the four task folders provided under `lmms-eval/lmms_eval/tasks/` into your existing lmms‑eval setup.

## Citation
If you find this project or the dataset helpful, please cite:
```bibtex
@misc{Euclids_Gift,
    title={Euclid’s Gift: Enhancing Spatial Perception and Reasoning in Vision-Language Models via Geometric Surrogate Tasks},
    author={Shijie Lian and Changti Wu and Laurence Tianruo Yang and Hang Yuan and Bin Yu and Lei Zhang and Kai Chen},
    year={2025},
    eprint={2505.09xxxx},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2505.09xxxx}
}
```

## Acknowledgements
We thank the [VeRL](https://github.com/volcengine/verl) / [EasyR1](https://github.com/hiyouga/EasyR1) training framework, as well as the benchmark suites [Super‑CLEVR](https://huggingface.co/datasets/MMInstruction/SuperClevr_Val), [Omni3DBench](https://huggingface.co/datasets/dmarsili/Omni3D-Bench), [VSI‑Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench), and [MindCube](https://huggingface.co/datasets/MLL-Lab/MindCube).