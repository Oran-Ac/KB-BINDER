accelerate launch  --config_file accelerate_config.yaml \
    lm/main_gen_llm.py \
    --dataset webqsp_0107 \
    --add_feature True \
    --feature_type replaced \
    --use_s_expression True \
    --test_batch_size 1 \
    --model_name meta-llama/Llama-3.2-3B \
    --few_shot 10