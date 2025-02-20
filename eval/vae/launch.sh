#!/bin/bash
CKPT_PATH=$1

if [ -z $IMG_PATH ]; then
    IMG_PATH="/mnt/jfs-hdd/sora/meta/validation/img_1k.csv"
fi

if [ -z $VID_PATH ]; then
    VID_PATH="/mnt/jfs-hdd/sora/meta/validation/vid_100.csv"
fi

if [ -z $NUM_FRAMES ]; then
    NUM_FRAMES=17
fi

if [ -z $FORCE_HUGGINGFACE ]; then
    FORCE_HUGGINGFACE=False
fi

if [ -z $CKPT_PATH ]; then # huggingface model
    STEP_RECORD=epoch0-global_step0
    LOG_DIR=outputs/OpenSoraVAE_V1_3/eval
    FORCE_HUGGINGFACE=True
    CKPT_PATH=pretrained_models/OpenSoraVAE_V1_3
else
    if [[ -d $CKPT_PATH ]] ; then
        STEP_RECORD=$(basename $CKPT_PATH)
    elif [[ -f $CKPT_PATH ]]; then
        STEP_RECORD=$(basename $(dirname $CKPT_PATH))
    else
        echo "$CKPT_PATH is not valid";
        exit 1
    fi
    LOG_DIR=$(dirname $CKPT_PATH)/eval
fi


echo "saving losses and metrics to $LOG_DIR"
echo "video path: ${VID_PATH}"
mkdir -p $LOG_DIR

# generate video, 256x256
torchrun --standalone --nproc_per_node=1 scripts/inference_opensoravae_v1_3.py configs/vae_v1_3/inference/video_16z_256x256.py --data-path $VID_PATH --save-dir samples/opensoravae_v1_3/${STEP_RECORD}/${NUM_FRAMES}x256x256 --ckpt-path ${CKPT_PATH} --num-frames $NUM_FRAMES --force-huggingface ${FORCE_HUGGINGFACE}
# calc metrics, 17x256x256
python eval/vae/eval_common_metric.py --batch_size 4 --real_video_dir samples/opensoravae_v1_3/${STEP_RECORD}/${NUM_FRAMES}x256x256_ori --generated_video_dir samples/opensoravae_v1_3/${STEP_RECORD}/${NUM_FRAMES}x256x256_rec --device cuda --sample_fps 24 --crop_size 256 --resolution 256 --num_frames $NUM_FRAMES --sample_rate 1 --metric ssim psnr lpips flolpips --type video --res_dir ${LOG_DIR}

# # generate video, 512x512
torchrun --standalone --nproc_per_node=1 scripts/inference_opensoravae_v1_3.py configs/vae_v1_3/inference/video_16z_512x512.py --data-path $VID_PATH --save-dir samples/opensoravae_v1_3/${STEP_RECORD}/${NUM_FRAMES}x512x512 --ckpt-path ${CKPT_PATH} --num-frames $NUM_FRAMES --force-huggingface ${FORCE_HUGGINGFACE}
# # calc metrics, 17x512x512
python eval/vae/eval_common_metric.py --batch_size 4 --real_video_dir samples/opensoravae_v1_3/${STEP_RECORD}/${NUM_FRAMES}x512x512_ori --generated_video_dir samples/opensoravae_v1_3/${STEP_RECORD}/${NUM_FRAMES}x512x512_rec --device cuda --sample_fps 24 --crop_size 512 --resolution 512 --num_frames $NUM_FRAMES --sample_rate 1 --metric ssim psnr lpips flolpips --type video --res_dir ${LOG_DIR}

# # generate image, 1024x1024
torchrun --standalone --nproc_per_node=1 scripts/inference_opensoravae_v1_3.py configs/vae_v1_3/inference/image_16z.py --data-path $IMG_PATH --save-dir samples/opensoravae_v1_3/${STEP_RECORD}/1x1024x1024 --ckpt-path ${CKPT_PATH} --num-frames 1 --force-huggingface ${FORCE_HUGGINGFACE}
# # calc metrics, 1x1024x1024
python eval/vae/eval_common_metric.py --batch_size 4 --real_video_dir samples/opensoravae_v1_3/${STEP_RECORD}/1x1024x1024_ori --generated_video_dir samples/opensoravae_v1_3/${STEP_RECORD}/1x1024x1024_rec --device cuda --sample_fps 1 --crop_size 1024 --resolution 1024 --num_frames 1 --sample_rate 1 --metric ssim psnr lpips --type image --res_dir ${LOG_DIR}
