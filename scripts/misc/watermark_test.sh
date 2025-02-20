CKPT=$1
RES=$2
OUTPUT=$3
TIME=$4
LOG_BASE=$5

# RAND_TYPES=(随机风景镜头 随机森林镜头 随机梯田镜头 随机河流镜头 随机沼泽镜头 随机热气球镜头 随机海底镜头 随机动物镜头 随机老虎镜头 随机候鸟迁徙镜头)

# RAND_TYPES=(随机新闻播报镜头 随机直播镜头 随机社交媒体镜头 随机电视剧镜头 随机网课镜头 随机广告镜头 随机宣传片镜头 随机电影预告片镜头 随机Vlog镜头 随机MV镜头 随机电竞镜头 随机奥运会镜头 随机足球镜头 随机乒乓球镜头)

# RAND_TYPES=(随机新闻播报镜头 随机直播镜头 随机社交媒体镜头)
# RAND_TYPES=(随机电视剧镜头 随机网课镜头 随机广告镜头)
# RAND_TYPES=(随机宣传片镜头 随机电影预告片镜头 随机Vlog镜头)
# RAND_TYPES=(随机MV镜头 随机电竞镜头 随机奥运会镜头)


RAND_REPEAT=10 # each generates 5 random samples

CMD="python scripts/inference.py configs/opensora-pro/inference/stage1.py"

if [[ -z "${OPENAI_API_KEY}" ]];
    then
        echo "Error: Required environment variable 'OPENAI_API_KEY' is not set."
        exit 1
    else
        for rand_idx in "${!RAND_TYPES[@]}"; do
        PROMPT=${RAND_TYPES[rand_idx]}
        for ((rand_count=1;rand_count<=${RAND_REPEAT};rand_count++)); do
            eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames ${TIME} --resolution ${RES} --aspect-ratio 9:16 --sample-name ${PROMPT}_${TIME}_${RES}_${rand_count} --batch-size 1 --llm-refine True > ${LOG_BASE}/${PROMPT}_${TIME}_${RES}_${rand_count}.log 2>&1
        done
        done
fi
