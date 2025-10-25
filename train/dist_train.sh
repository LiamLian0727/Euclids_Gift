set -o pipefail

MASTER_ADDR=${MASTER_ADDR:-"localhost"}
NODE_RANK=${RANK:-"0"}
NNODES=${WORLD_SIZE:-"1"}

# Prepare the Ray start command
RAY_START_CMD="ray start"
# Get the head node IP address: the IP of $MASTER_ADDR (node 0)
HEAD_NODE_ADDRESS=$(getent hosts $MASTER_ADDR | awk '{ print $1 }')
echo "Resolved MASTER_IP: $HEAD_NODE_ADDRESS"

if [ "$NODE_RANK" -eq 0 ]; then
    # Node 0 acts as the head node
    RAY_START_CMD+=" --head --port=6379"
else
    # Wait for the head node to be ready before starting worker nodes
    while ! nc -zv $MASTER_ADDR 6379 >/dev/null 2>&1; do
        echo "Waiting for ray head node to be ready..."
        sleep 1
    done
    # Other nodes join as workers to form the Ray cluster with the head
    RAY_START_CMD+=" --block --address=${HEAD_NODE_ADDRESS}:6379"
fi

echo $RAY_START_CMD
$RAY_START_CMD

if [ "$NODE_RANK" -eq 0 ]; then
    # Wait for all Ray workers to join the cluster
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
        data.train_files=/mnt/datasets/Euclid30K/Euclid30K_train.parquet \
        data.val_files=/mnt/datasets/Euclid30K/Euclid30K_val.parquet \
        worker.actor.model.model_path=/mnt/models/Qwen2.5-VL-7B-Instruct \
        trainer.experiment_name=EXPERIMENT_NAME \
        worker.actor.micro_batch_size_per_device_for_update=1 \
        worker.actor.micro_batch_size_per_device_for_experience=8 \
        worker.actor.clip_ratio_low=0.2 \
        worker.actor.clip_ratio_high=0.28 \
        worker.reward.reward_function=/mnt/code/Euclids_Gift/train/euclid.py:compute_score \
        trainer.total_epochs=10 \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.save_checkpoint_path=/mnt/models/Qwen2.5-VL-7B-Euclid
fi
