set -x

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=DEBUG
# export TORCH_CUDA_ARCH_LIST="9.0"

model_path=/mnt/models/Qwen2.5-VL-7B-Instruct

python3 -m lmms_eval \
    --model vllm \
    --model_args model_version=${model_path},tensor_parallel_size=4,gpu_memory_utilization=0.8 \
    --tasks super_clevr_qwen,Omni3D-Bench_qwen,vsibench_qwen,mindcube_qwen \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vllm \
    --output_path ./output/Qwen25VL7B \
    --verbosity DEBUG \