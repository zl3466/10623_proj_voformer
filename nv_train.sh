#!/bin/bash
# Set default values
BATCH_SIZE=8
VOCAB_SIZE=100
INPUT_IMAGE_SIZE=256

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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch_size BATCH_SIZE    Batch size per device (default: 8)"
            echo "  --vocab_size VOCAB_SIZE    Vocabulary size for pose tokenization (default: 100)"
            echo "  --img_size IMG_SIZE        Input image size in pixels (default: 256)"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use all defaults"
            echo "  $0 --batch_size 4                     # Custom batch size"
            echo "  $0 --batch_size 4 --vocab_size 200    # Custom batch and vocab"
            echo "  $0 --batch_size 4 --vocab_size 100 --img_size 256  # All custom"
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
WANDB_RUN_NAME="nv_vocab${VOCAB_SIZE}_img${INPUT_IMAGE_SIZE}_bs${BATCH_SIZE}"
OUTPUT_DIR="./output/${WANDB_RUN_NAME}"

echo "Starting training with:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Vocab size: ${VOCAB_SIZE}"
echo "  Input image size: ${INPUT_IMAGE_SIZE}"
echo "  Wandb run name: ${WANDB_RUN_NAME}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""

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
    --wandb_project visual-odometry \
    --wandb_run_name ${WANDB_RUN_NAME} \
    --output_dir "${OUTPUT_DIR}" \
    --save_steps 1000 \
    --eval_steps 1000 
