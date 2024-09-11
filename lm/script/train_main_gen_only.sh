accelerate launch  --config_file accelerate_config.yaml \
     lm/main_gen_only.py \
    --dataset webqsp_0107 \
     --lr 1e-5 \
     --weight_decay 1e-7 \
     --max_epochs 100  \
     --patience 10 \
     --train_batch_size 16 \
     --eval_batch_size 16 \
     --test_batch_size 16 \
     --lr_decay_type cosine_warmup \
     --model_name Salesforce/codet5-large \
     --optimizer adamw \
