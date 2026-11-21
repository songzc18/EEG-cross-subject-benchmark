#!/bin/bash
set -euo pipefail

mkdir -p fm_evaluation/log

# 1. 准备 POP 共享模型
# 20 k × 3 seeds × 3 phases × 2 sessions = 360 tasks
pop_job=$(sbatch --parsable <<'POP'
#!/bin/bash
#SBATCH --job-name=pop_v2
#SBATCH --output=fm_evaluation/log/pop_v2_%A_%a.log
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-359%20

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

task_id=$SLURM_ARRAY_TASK_ID
phases=(offline online offon)

k=$(((task_id % 20 + 1) * 4))
seed=$(((task_id / 20) % 3))
phase=${phases[$(((task_id / 60) % 3))]}
ses=$((task_id / 180 + 1))

echo "POP: session=${ses}, phase=${phase}, seed=${seed}, k=${k}"

python -u run_ev.py \
    --protocol POP \
    --sub 1 \
    --train_ses "$ses" \
    --train_phase "$phase" \
    --protocol_ses "$ses" \
    --protocol_phase "$phase" \
    --seed "$seed" \
    --k_values "$k"
POP
)

pop_job=${pop_job%%;*}
echo "POP preparation job: ${pop_job}"

# 2. POP 全部成功后运行评估
# 54 subjects × 2 sessions × 3 phases × 3 seeds = 972 tasks
sbatch --dependency=afterok:"$pop_job" <<'EVALUATION'
#!/bin/bash
#SBATCH --job-name=eegnet_v2
#SBATCH --output=fm_evaluation/log/eegnet_v2_%A_%a.log
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-971%20

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

task_id=$SLURM_ARRAY_TASK_ID
phases=(offline online offon)
k_values=(4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80)

seed=$((task_id % 3))
train_phase=${phases[$(((task_id / 3) % 3))]}
train_ses=$(((task_id / 9) % 2 + 1))
sub=$((task_id / 18 + 1))

# 先运行匹配条件的 LOSO，准备源模型
# LOSO 不依赖 k，只需运行一次
python -u run_ev.py \
    --protocol LOSO \
    --sub "$sub" \
    --train_ses "$train_ses" \
    --train_phase "$train_phase" \
    --protocol_ses "$train_ses" \
    --protocol_phase "$train_phase" \
    --seed "$seed" \
    --k_values 0

# 匹配条件：SELF、POP、CAL
# 每次 Python 调用只传一个 k，分别保存 CSV
for k in "${k_values[@]}"; do
    for protocol in SELF POP CAL; do
        echo "${protocol}: sub=${sub}, train=${train_ses}/${train_phase}, target=${train_ses}/${train_phase}, seed=${seed}, k=${k}"

        python -u run_ev.py \
            --protocol "$protocol" \
            --sub "$sub" \
            --train_ses "$train_ses" \
            --train_phase "$train_phase" \
            --protocol_ses "$train_ses" \
            --protocol_phase "$train_phase" \
            --seed "$seed" \
            --k_values "$k"
    done
done

# 再运行训练与 protocol 设置不同的条件
for protocol_ses in 1 2; do
    for protocol_phase in "${phases[@]}"; do

        if [[ "$train_ses" == "$protocol_ses" &&
              "$train_phase" == "$protocol_phase" ]]; then
            continue
        fi

        # LOSO 复用相同源模型，保存该条件下的预测
        python -u run_ev.py \
            --protocol LOSO \
            --sub "$sub" \
            --train_ses "$train_ses" \
            --train_phase "$train_phase" \
            --protocol_ses "$protocol_ses" \
            --protocol_phase "$protocol_phase" \
            --seed "$seed" \
            --k_values 0

        for k in "${k_values[@]}"; do
            echo "CAL: sub=${sub}, train=${train_ses}/${train_phase}, target=${protocol_ses}/${protocol_phase}, seed=${seed}, k=${k}"

            python -u run_ev.py \
                --protocol CAL \
                --sub "$sub" \
                --train_ses "$train_ses" \
                --train_phase "$train_phase" \
                --protocol_ses "$protocol_ses" \
                --protocol_phase "$protocol_phase" \
                --seed "$seed" \
                --k_values "$k"
        done
    done
done
EVALUATION