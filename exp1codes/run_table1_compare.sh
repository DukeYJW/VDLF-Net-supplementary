#!/bin/bash
# Table 1 对比实验：训练一次至 MAX_EPOCHS，每隔 RECORD_INTERVAL 记录结果，避免重复训练
# 仅对 BATCH_SIZE 做循环；每个 BATCH_SIZE 下训练一次，得到多个 epoch  checkpoint 记录

export OMP_NUM_THREADS=4
MAX_EPOCHS=${MAX_EPOCHS:-400}
RECORD_INTERVAL=${RECORD_INTERVAL:-20}
DATA_ROOT=${DATA_ROOT:-data}
OUTPUT_CSV=${OUTPUT_CSV:-table1_compare_results.csv}

# BATCH_SIZE 列表
BATCH_SIZE_LIST=(32 64 128 256 512 1024 2048 4096)

echo "=============================================="
echo "Table 1 对比实验 (训练一次，每 ${RECORD_INTERVAL} epoch 记录)"
echo "=============================================="
echo "MAX_EPOCHS: $MAX_EPOCHS"
echo "RECORD_INTERVAL: $RECORD_INTERVAL"
echo "BATCH_SIZE: ${BATCH_SIZE_LIST[@]}"
echo "OUTPUT_CSV: $OUTPUT_CSV"
echo "=============================================="

total_runs=${#BATCH_SIZE_LIST[@]}
run_count=0

for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"; do
  run_count=$((run_count + 1))
  echo ""
  echo "========== 实验 $run_count/$total_runs =========="
  echo "BATCH_SIZE=$BATCH_SIZE (训练至 ${MAX_EPOCHS} epoch，每 ${RECORD_INTERVAL} 记录)"
  echo "================================================"

  python table1_compare.py \
    --MAX_EPOCHS "$MAX_EPOCHS" \
    --RECORD_INTERVAL "$RECORD_INTERVAL" \
    --BATCH_SIZE "$BATCH_SIZE" \
    --DATA_ROOT "$DATA_ROOT" \
    --OUTPUT_CSV "$OUTPUT_CSV"

  if [ $? -ne 0 ]; then
    echo "错误: 实验 $run_count 失败 (BATCH_SIZE=$BATCH_SIZE)"
  fi
done

echo ""
echo "=============================================="
echo "全部 $total_runs 个 BATCH_SIZE 实验完成！结果保存在 $OUTPUT_CSV"
echo "=============================================="
echo "5秒后关机..."
sleep 5
shutdown -h now