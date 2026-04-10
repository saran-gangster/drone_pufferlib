#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/drone_pufferlib}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/kaggle/working/artifacts_kaggle}"
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-540}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-3000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250000}"
NUM_WORKERS_PER_RANK="${NUM_WORKERS_PER_RANK:-2}"
ENVS_PER_WORKER="${ENVS_PER_WORKER:-4}"

cd "${REPO_DIR}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install Cython ninja
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
python -m pip install --no-build-isolation pufferlib==3.0.0
python -m pip install -e .

GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU ' || true)"
if [ "${GPU_COUNT}" -lt 1 ]; then
    echo "No GPUs detected. Aborting."
    exit 1
fi
if [ "${GPU_COUNT}" -gt 2 ]; then
    GPU_COUNT=2
fi

export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1

mkdir -p "${ARTIFACTS_DIR}"

echo "Visible GPUs:"
nvidia-smi -L
echo "Using ${GPU_COUNT} process(es) across CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --standalone --nproc-per-node="${GPU_COUNT}" \
    -m drone_pufferlib.training.train \
    --config configs/train_kaggle_2x_t4.yaml \
    --override "artifacts_dir=${ARTIFACTS_DIR}" \
    --override "run_name=kaggle_dual_t4_long" \
    --override "vectorization.num_workers=${NUM_WORKERS_PER_RANK}" \
    --override "vectorization.envs_per_worker=${ENVS_PER_WORKER}" \
    --override "training.total_env_steps=${TOTAL_ENV_STEPS}" \
    --override "training.eval_interval=${EVAL_INTERVAL}" \
    --override "evaluation.episodes=100" \
    --override "evaluation.random_policy_episodes=30" \
    --max-runtime-minutes "${MAX_RUNTIME_MINUTES}" \
    2>&1 | tee "${ARTIFACTS_DIR}/train_stdout.log"

python -m drone_pufferlib.training.evaluate \
    --config configs/eval.yaml \
    --checkpoint "${ARTIFACTS_DIR}/checkpoints/best.pt" \
    --override "artifacts_dir=${ARTIFACTS_DIR}" \
    2>&1 | tee "${ARTIFACTS_DIR}/eval_stdout.log"

tar -czf /kaggle/working/drone_pufferlib_artifacts.tar.gz \
    -C /kaggle/working \
    "$(basename "${ARTIFACTS_DIR}")"

echo "Artifacts archived at /kaggle/working/drone_pufferlib_artifacts.tar.gz"
