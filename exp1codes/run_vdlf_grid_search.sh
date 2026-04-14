#!/bin/bash
export OMP_NUM_THREADS=8
# VDLF-Net 五重循环网格搜索
# 对 VDLF_LR、VDLF_WEIGHT_DECAY、VDLF_ALPHA、VDLF_LATENT_DIM、KL_ANNEAL_EPOCHS 进行参数实验
# 每次运行结果会追加到 vdlf_results.csv

# 可在此修改参数范围，或通过环境变量覆盖
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-128}
OUTPUT_CSV=${OUTPUT_CSV:-vdlf_results.csv}
DATA_ROOT=${DATA_ROOT:-data}

# 学习率列表
VDLF_LR_LIST=(0.002)
# 权重衰减列表
VDLF_WEIGHT_DECAY_LIST=(0.00015)
# Alpha 列表 (VAE 损失权重)
VDLF_ALPHA_LIST=(0.02)
# 潜在空间维度列表
VDLF_LATENT_DIM_LIST=(32 64 80 100 144 160 200 240)
# KL退火周期列表 (0=不退火)
KL_ANNEAL_EPOCHS_LIST=(0)

echo "=============================================="
echo "VDLF-Net 参数网格搜索"
echo "=============================================="
echo "LR: ${VDLF_LR_LIST[@]}"
echo "WEIGHT_DECAY: ${VDLF_WEIGHT_DECAY_LIST[@]}"
echo "ALPHA: ${VDLF_ALPHA_LIST[@]}"
echo "LATENT_DIM: ${VDLF_LATENT_DIM_LIST[@]}"
echo "KL_ANNEAL_EPOCHS: ${KL_ANNEAL_EPOCHS_LIST[@]}"
echo "EPOCHS: $EPOCHS"
echo "OUTPUT_CSV: $OUTPUT_CSV"
echo "=============================================="

total_runs=$((${#VDLF_LR_LIST[@]} * ${#VDLF_WEIGHT_DECAY_LIST[@]} * ${#VDLF_ALPHA_LIST[@]} * ${#VDLF_LATENT_DIM_LIST[@]} * ${#KL_ANNEAL_EPOCHS_LIST[@]}))
run_count=0

for VDLF_LR in "${VDLF_LR_LIST[@]}"; do
  for VDLF_WEIGHT_DECAY in "${VDLF_WEIGHT_DECAY_LIST[@]}"; do
    for VDLF_ALPHA in "${VDLF_ALPHA_LIST[@]}"; do
      for VDLF_LATENT_DIM in "${VDLF_LATENT_DIM_LIST[@]}"; do
        for KL_ANNEAL_EPOCHS in "${KL_ANNEAL_EPOCHS_LIST[@]}"; do
          run_count=$((run_count + 1))
          echo ""
          echo "========== 实验 $run_count/$total_runs =========="
          echo "VDLF_LR=$VDLF_LR, VDLF_WEIGHT_DECAY=$VDLF_WEIGHT_DECAY, VDLF_ALPHA=$VDLF_ALPHA, VDLF_LATENT_DIM=$VDLF_LATENT_DIM, KL_ANNEAL_EPOCHS=$KL_ANNEAL_EPOCHS"
          echo "================================================"

          python train_vdlfnet.py \
            --VDLF_LR "$VDLF_LR" \
            --VDLF_WEIGHT_DECAY "$VDLF_WEIGHT_DECAY" \
            --VDLF_ALPHA "$VDLF_ALPHA" \
            --VDLF_LATENT_DIM "$VDLF_LATENT_DIM" \
            --KL_ANNEAL_EPOCHS "$KL_ANNEAL_EPOCHS" \
            --EPOCHS "$EPOCHS" \
            --BATCH_SIZE "$BATCH_SIZE" \
            --DATA_ROOT "$DATA_ROOT" \
            --OUTPUT_CSV "$OUTPUT_CSV"

          if [ $? -ne 0 ]; then
            echo "错误: 实验 $run_count 失败 (LR=$VDLF_LR, WD=$VDLF_WEIGHT_DECAY, Alpha=$VDLF_ALPHA, Latent=$VDLF_LATENT_DIM, KLAnneal=$KL_ANNEAL_EPOCHS)"
          fi
        done
      done
    done
  done
done

echo ""
echo "=============================================="
echo "全部 $total_runs 个实验完成！结果保存在 $OUTPUT_CSV"
echo "=============================================="
echo "5秒后关机..."
sleep 5
shutdown -h now
