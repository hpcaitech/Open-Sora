# !/bin/bash

CKPT=$1
NUM_FRAMES=$2
MODEL_NAME=$3
RES=$4
ASP_RATIO=$5

NUM_SAMPLING_STEPS=$6
FLOW=$7
LLM_REFINE=$8

if [[ $CKPT == *"ema"* ]]; then
    parentdir=$(dirname $CKPT)
    CKPT_BASE=$(basename $parentdir)_ema
else
    CKPT_BASE=$(basename $CKPT)
fi
LOG_BASE=$(dirname $CKPT)/eval
echo "Logging to $LOG_BASE"

GPUS=(0 1 2 3 4 5 6 7)
TASK_ID_LIST=(4a 4b 4c 4d 4e 4f 4g 4h) # for log records only
START_INDEX_LIST=(0 120 240 360 480 600 720 840)
END_INDEX_LIST=(120 240 360 480 600 720 840 2000)

## Modify the following to run on multiple machines for faster results
## 720p will take quite long on a single machine
# START_INDEX_LIST=(60 180 300 420 540 660 780 900)
# END_INDEX_LIST=(120 240 360 480 600 720 840 2000)
# LOG_BASE=$(dirname $CKPT)/eval/last_60
# mkdir -p ${LOG_BASE}
# echo "Logging to $LOG_BASE"



for i in "${!GPUS[@]}"; do
    if [ -z ${RES} ] || [ -z ${ASP_RATIO} ]  ;
        then
            CUDA_VISIBLE_DEVICES=${GPUS[i]} bash eval/sample.sh $CKPT ${NUM_FRAMES} ${MODEL_NAME} -4 ${START_INDEX_LIST[i]} ${END_INDEX_LIST[i]} > ${LOG_BASE}/${TASK_ID_LIST[i]}.log 2>&1 &
        else
            if [ -z ${NUM_SAMPLING_STEPS} ];
                then
                    CUDA_VISIBLE_DEVICES=${GPUS[i]} bash eval/sample.sh $CKPT ${NUM_FRAMES} ${MODEL_NAME} -4 ${START_INDEX_LIST[i]} ${END_INDEX_LIST[i]} ${RES} ${ASP_RATIO} > ${LOG_BASE}/${TASK_ID_LIST[i]}.log 2>&1 &
                else
                    if [ -z ${FLOW} ];
                    then
                        CUDA_VISIBLE_DEVICES=${GPUS[i]} bash eval/sample.sh $CKPT ${NUM_FRAMES} ${MODEL_NAME} -4 ${START_INDEX_LIST[i]} ${END_INDEX_LIST[i]} ${RES} ${ASP_RATIO} ${NUM_SAMPLING_STEPS} > ${LOG_BASE}/${TASK_ID_LIST[i]}.log 2>&1 &
                    else
                        if [ -z ${LLM_REFINE} ];
                            then
                                CUDA_VISIBLE_DEVICES=${GPUS[i]} bash eval/sample.sh $CKPT ${NUM_FRAMES} ${MODEL_NAME} -4 ${START_INDEX_LIST[i]} ${END_INDEX_LIST[i]} ${RES} ${ASP_RATIO} ${NUM_SAMPLING_STEPS} ${FLOW} > ${LOG_BASE}/${TASK_ID_LIST[i]}.log 2>&1 &
                            else
                                CUDA_VISIBLE_DEVICES=${GPUS[i]} bash eval/sample.sh $CKPT ${NUM_FRAMES} ${MODEL_NAME} -4 ${START_INDEX_LIST[i]} ${END_INDEX_LIST[i]} ${RES} ${ASP_RATIO} ${NUM_SAMPLING_STEPS} ${FLOW} ${LLM_REFINE} > ${LOG_BASE}/${TASK_ID_LIST[i]}.log 2>&1 &
                        fi
                    fi
            fi
    fi
done
