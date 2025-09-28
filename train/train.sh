set -o pipefail

MASTER_ADDR=${MASTER_ADDR:-"localhost"}
NODE_RANK=${RANK:-"0"}
NNODES=${WORLD_SIZE:-"1"}

# 准备 ray command，
RAY_START_CMD="ray start"
# 获取 head 节点的 ip 地址：即 $MASTER_ADDR 的 ip 地址，也就是 node 0 节点
HEAD_NODE_ADDRESS=$(getent hosts $MASTER_ADDR | awk '{ print $1 }')
echo "Resolved MASTER_IP: $HEAD_NODE_ADDRESS"

if [ "$NODE_RANK" -eq 0 ]; then
    # node 0 作为 head 节点
    RAY_START_CMD+=" --head --port=6379"
else
    # 等待 head 节点起来后再启动 worker 节点
    while ! nc -zv $MASTER_ADDR 6379 >/dev/null 2>&1; do
        echo "Waiting for ray head node to be ready..."
        sleep 1
    done
    # 其他节点作为 worker 节点加⼊到 head 节点构成 ray cluster
    RAY_START_CMD+=" --block --address=${HEAD_NODE_ADDRESS}:6379"
fi

echo $RAY_START_CMD
$RAY_START_CMD

if [ "$NODE_RANK" -eq 0 ]; then
    # 等待所有 ray workers 都加⼊ ray cluster
    while :; do
        ready_nodes=$(ray list nodes | grep Total | awk '{print $2}')
        echo "Ready Nodes: $ready_nodes"
        [ "$ready_nodes" -eq "$NNODES" ] && break
        echo "Waiting for ray worker nodes to be ready..."
        sleep 1
    done

    cd /workspace/EasyR1

    python3 -m verl.trainer.main \
        config=examples/config.yaml \
        data.train_files=/mnt/datasets/EuclidsGigt/Euclid30K_train.parquet \
        data.val_files=/mnt/datasets/EuclidsGigt/Euclid30K_train.parquet \
        worker.actor.model.model_path=/mnt/models/Qwen2.5-VL-7B-Instruct \
        trainer.experiment_name=EXPERIMENT_NAME \
        worker.actor.micro_batch_size_per_device_for_update=1 \
        worker.actor.micro_batch_size_per_device_for_experience=8 \
        worker.actor.clip_ratio_low=0.2 \
        worker.actor.clip_ratio_high=0.28 \
        worker.reward.reward_function=/mnt/code/Euclids_Gift/train/euclid.py:compute_score \
        algorithm.online_filtering=True \
        trainer.total_epochs=10 \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.save_checkpoint_path=/mnt/project_ai4edu/lsj/E_Plus/CHECK_DATASET/Qwen25VL7B_Epoch10_New_Reward
fi