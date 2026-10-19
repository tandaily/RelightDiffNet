export MODEL_ID="/path/to/stable-diffusion-v1-5"
export OUTPUT_DIR="/path/to/relightdiffnet_output"
export RMBG_MODEL_PATH="/path/to/RMBG-2.0"


CUDA_VISIBLE_DEVICES=5 python train_relight_diffnet.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name="/path/to/train_syn_object.jsonl" \
  --object_dataset_name="/path/to/object_train.jsonl" \
  --train_batch_size=12 --gradient_checkpointing \
  --max_train_steps=30000 \
  --checkpointing_steps=2000 --checkpoints_total_limit=10 \
  --validation_epochs=1 \
  --learning_rate=1e-05 --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_syn_portrait_image_url="/path/to/synthetic_portrait_test.jsonl" \
  --val_object_image_url="/path/to/object_test.jsonl" \
  --val_real_portrait_image_url="/path/to/real_portrait_test.jsonl" \
  --validation_prompt="" \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --resume_from_checkpoint=latest \
  --dataloader_num_workers=4 \
  --use_8bit_adam \
