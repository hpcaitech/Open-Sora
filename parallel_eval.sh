#!/bin/bash

CKPT=$1
NUM_FRAMES=$2
MODEL_NAME=$3
LOG_BASE=$(dirname $CKPT)/eval
if [[ $CKPT == *"ema"* ]]; then
  parentdir=$(dirname $CKPT)
  CKPT_BASE=$(basename $parentdir)_ema
else
  CKPT_BASE=$(basename $CKPT)
fi
VBENCH_SAMPLE_DIR=samples/samples_${MODEL_NAME}_${CKPT_BASE}_vbench

sleep_time=10m

human_eval_ready=0
human_ready_count=8
loss_eval_ready=0
loss_ready_count=6
vbench_gen_ready=0
vbench_gen_ready_count=8
vbench_calc_ready=0
vbench_calc_ready_count=8
vbench_i2v_gen_ready=0
vbench_i2v_gen_ready_count=8
vbench_i2v_calc_ready=0
vbench_i2v_calc_ready_count=8

# check if human eval ready
function check_human_eval(){
    term='Runtime:'
    finished_list=()
    TASK_ID_LIST=(2a 2b 2c 2d 2e 2f 2g 2h)

    for i in "${!TASK_ID_LIST[@]}"; do
        last_line=$(tail -n 1 ${LOG_BASE}/${TASK_ID_LIST[i]}.log)
        if [[ $last_line == *${term}* ]];
            then
                finished_list+=(${TASK_ID_LIST[i]})

        fi
    done

    echo "completed tasks: ${finished_list[@]}"

    if [ ${#finished_list[@]}=$human_ready_count ];
        then
            human_eval_ready=1
        else
            echo ${#finished_list[@]}
    fi
}

function check_loss_eval(){
    term='Evaluation losses: {('
    finished_list=()
    TASK_ID_LIST=(img 144p_vid 240p_vid 360p_vid 480p_vid 720p_vid)
    for i in "${!TASK_ID_LIST[@]}"; do
        last_line=$(tail -n 1 ${LOG_BASE}/${TASK_ID_LIST[i]}.log)
        if [[ $last_line == *${term}* ]];
            then
                finished_list+=(${TASK_ID_LIST[i]})
        fi
    done

    echo "completed tasks: ${finished_list[@]}"

    if [ ${#finished_list[@]}=$loss_ready_count ];
        then
            loss_eval_ready=1
        else
            echo ${#finished_list[@]}
    fi
}

function check_vbench_gen(){
    term='Runtime:'
    finished_list=()
    TASK_ID_LIST=(4a 4b 4c 4d 4e 4f 4g 4h)
    for i in "${!TASK_ID_LIST[@]}"; do
        last_line=$(tail -n 1 ${LOG_BASE}/${TASK_ID_LIST[i]}.log)
        if [[ $last_line == *${term}* ]];
            then
                finished_list+=(${TASK_ID_LIST[i]})
        fi
    done

    echo "completed tasks: ${finished_list[@]}"

    if [ ${#finished_list[@]}=$vbench_gen_ready_count ];
        then
            vbench_gen_ready=1
        else
            echo ${#finished_list[@]}
    fi
}

function check_vbench_calc(){
    term='Runtime:'
    finished_list=()
    TASK_ID_LIST=(calc_vbench_a calc_vbench_b calc_vbench_c calc_vbench_d calc_vbench_e calc_vbench_f calc_vbench_g calc_vbench_h)
    for i in "${!TASK_ID_LIST[@]}"; do
        last_line=$(tail -n 1 ${LOG_BASE}/${TASK_ID_LIST[i]}.log)
        if [[ $last_line == *${term}* ]];
            then
                finished_list+=(${TASK_ID_LIST[i]})
        fi
    done

    echo "completed tasks: ${finished_list[@]}"

    if [ ${#finished_list[@]}=$vbench_calc_ready_count ];
        then
            vbench_calc_ready=1
        else
            echo ${#finished_list[@]}
    fi
}

function check_vbench_i2v_gen(){
    term='Runtime:'
    finished_list=()
    TASK_ID_LIST=(5a 5b 5c 5d 5e 5f 5g 5h)
    for i in "${!TASK_ID_LIST[@]}"; do
        last_line=$(tail -n 1 ${LOG_BASE}/${TASK_ID_LIST[i]}.log)
        if [[ $last_line == *${term}* ]];
            then
                finished_list+=(${TASK_ID_LIST[i]})
        fi
    done

    echo "completed tasks: ${finished_list[@]}"

    if [ ${#finished_list[@]}=$vbench_i2v_gen_ready_count ];
        then
            vbench_i2v_gen_ready=1
        else
            echo ${#finished_list[@]}
    fi
}

function check_vbench_i2v_calc(){
    term='Runtime:'
    finished_list=()
    TASK_ID_LIST=(calc_vbench_i2v_a calc_vbench_i2v_b calc_vbench_i2v_c calc_vbench_i2v_d calc_vbench_i2v_e calc_vbench_i2v_f calc_vbench_i2v_g calc_vbench_i2v_h)
    for i in "${!TASK_ID_LIST[@]}"; do
        last_line=$(tail -n 1 ${LOG_BASE}/${TASK_ID_LIST[i]}.log)
        if [[ $last_line == *${term}* ]];
            then
                finished_list+=(${TASK_ID_LIST[i]})
        fi
    done

    echo "completed tasks: ${finished_list[@]}"

    if [ ${#finished_list[@]}=$vbench_i2v_calc_ready_count ];
        then
            vbench_i2v_calc_ready=1
        else
            echo ${#finished_list[@]}
    fi
}


### ===== Main =====

start=$(date +%s)

### human eval, ~60min
echo "$(date): running human eval"
bash eval/human_eval/launch.sh $CKPT $NUM_FRAMES $MODEL_NAME
while [ $human_eval_ready -eq 0 ]
do
    sleep ${sleep_time}
    check_human_eval
done

### eval loss, ~160min
echo "$(date): running eval loss"
bash eval/eval_loss/launch.sh $CKPT $MODEL_NAME
while [ $loss_eval_ready -eq 0 ]
do
    sleep ${sleep_time}
    check_loss_eval
done
python eval/loss/tabulate_rl_loss.py --log_dir $LOG_BASE

### vbench gen, ~80min
echo "$(date): running vbench gen"
bash eval/vbench/launch.sh $CKPT $NUM_FRAMES $MODEL_NAME
while [ $vbench_gen_ready -eq 0 ]
do
    sleep ${sleep_time}
    check_vbench_gen
done

### vbench calc, ~30min
echo "$(date): running vbench calc"
bash eval/vbench/launch_calc.sh $VBENCH_SAMPLE_DIR $LOG_BASE
while [ $vbench_calc_ready -eq 0 ]
do
    sleep ${sleep_time}
    check_vbench_calc
done
python eval/vbench/tabulate_vbench_scores.py --score_dir ${LOG_BASE}/vbench

### vbench_i2v gen, ~65min
echo "$(date): running vbench_i2v gen"
bash eval/vbench_i2v/launch.sh $CKPT $NUM_FRAMES $MODEL_NAME
while [ $vbench_i2v_gen_ready -eq 0 ]
do
    sleep ${sleep_time}
    check_vbench_i2v_gen
done

### vbench_i2v calc, ~48min
echo "$(date): running vbench_i2v calc"
bash eval/vbench_i2v/launch_calc.sh $VBENCH_SAMPLE_DIR $LOG_BASE
while [ $vbench_i2v_calc_ready -eq 0 ]
do
    sleep ${sleep_time}
    check_vbench_i2v_calc
done
python eval/vbench_i2v/tabulate_vbench_i2v_scores.py --score_dir ${LOG_BASE}/vbench_i2v

### End
echo "$(date): eval completed for ${CKPT}"
end=$(date +%s)
runtime=$((end - start))
echo "Runtime: $runtime seconds"
