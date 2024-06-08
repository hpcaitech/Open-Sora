# !/bin/bash

VIDEO_DIR=$1
CKPT_DIR=$2
LOG_BASE=$CKPT_DIR
mkdir -p $LOG_BASE
echo "Logging to $LOG_BASE"

GPUS=(0 1 2 3 4 5 6 7)
CALC_I2V_LIST=(True True False False False False False False)
CALC_QUALITY_LIST=(False False True True True True True True)
START_INDEX_LIST=(0 2 0 2 3 4 5 6)
END_INDEX_LIST=(2 -1 2 3 4 5 6 -1)
TASK_ID_LIST=(calc_vbench_i2v_a calc_vbench_i2v_b calc_vbench_i2v_c calc_vbench_i2v_d calc_vbench_i2v_e calc_vbench_i2v_f calc_vbench_i2v_g calc_vbench_i2v_h) # for log records only


for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[i]} python eval/vbench_i2v/calc_vbench_i2v.py $VIDEO_DIR $CKPT_DIR --calc_i2v ${CALC_I2V_LIST[i]} --calc_quality ${CALC_QUALITY_LIST[i]} --start ${START_INDEX_LIST[i]} --end ${END_INDEX_LIST[i]} > ${LOG_BASE}/${TASK_ID_LIST[i]}.log 2>&1 &
done
