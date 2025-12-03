#!/bin/bash
# Set default values
BATCH_SIZE=8
VOCAB_SIZE=100
INPUT_IMAGE_SIZE=256
NUM_INPUT_FRAMES=""
NUM_INPUT_POSES=""
RESUME_FROM_CHECKPOINT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --vocab_size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        --img_size)
            INPUT_IMAGE_SIZE="$2"
            shift 2
            ;;
        --num_input_frames)
            NUM_INPUT_FRAMES="$2"
            shift 2
            ;;
        --num_input_poses)
            NUM_INPUT_POSES="$2"
            shift 2
            ;;
        --resume_from_ckpt)
            RESUME_FROM_CHECKPOINT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch_size BATCH_SIZE           Batch size per device (default: 8)"
            echo "  --vocab_size VOCAB_SIZE           Vocabulary size for pose tokenization (default: 100)"
            echo "  --img_size IMG_SIZE               Input image size in pixels (default: 256)"
            echo "  --num_input_frames NUM_FRAMES     Number of input frames (optional, uses config default)"
            echo "  --num_input_poses NUM_POSES       Number of input poses (optional, uses config default)"
            echo "  --resume_from_ckpt PATH           Path to checkpoint directory to resume from (optional)"
            echo "  -h, --help                        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use all defaults"
            echo "  $0 --batch_size 4                     # Custom batch size"
            echo "  $0 --batch_size 4 --vocab_size 200    # Custom batch and vocab"
            echo "  $0 --batch_size 4 --vocab_size 100 --img_size 256  # All custom"
            echo "  $0 --num_input_frames 8 --num_input_poses 8  # Custom frame/pose counts"
            echo "  $0 --resume_from_ckpt ./output/nv_vocab100_img256_bs8/checkpoint-10000  # Resume training"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Clear any distributed environment variables
unset NCCL_ALGO
unset NCCL_PROTO
unset NCCL_MIN_NCHANNELS
unset NCCL_CUMEM_ENABLE

unset WORLD_SIZE
unset RANK
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT
# Also clear SLURM variables
unset SLURM_PROCID
unset SLURM_LOCALID
unset SLURM_NTASKS
unset SLURM_NPROCS

source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/etc/profile.d/conda.sh
conda activate voformer
export WANDB_API_KEY="de2b136779280d18cb6d59d1a23248b5010833f8"

# Generate wandb run name based on parameters
WANDB_RUN_NAME="nv_vocab${VOCAB_SIZE}_img${INPUT_IMAGE_SIZE}_bs${BATCH_SIZE}_n${NUM_INPUT_FRAMES}"
OUTPUT_DIR="./output/${WANDB_RUN_NAME}"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    train.py \
    --nusc_data_path /lustre/fsw/portfolios/nvr/users/ymingli/datasets/vlm_visual_odom/NuScenes/train_test \
    --batch_size ${BATCH_SIZE} \
    --vocab_size ${VOCAB_SIZE} \
    --input_image_size ${INPUT_IMAGE_SIZE} \
    ${NUM_INPUT_FRAMES:+--num_input_frames ${NUM_INPUT_FRAMES}} \
    ${NUM_INPUT_POSES:+--num_input_poses ${NUM_INPUT_POSES}} \
    --wandb_project visual-odometry \
    --wandb_run_name ${WANDB_RUN_NAME} \
    --output_dir "${OUTPUT_DIR}" \
    --save_steps 1000 \
    --eval_steps 1000 \
    ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"} 
