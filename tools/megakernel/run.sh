#!/bin/sh
TLLM_LOG_LEVEL="WARNING" \
export CUDA_VISIBLE_DEVICES=1
python3 run.py \
    --input_file test.json \
    --batch_size 1 \
    --max_input_len 160 \
    --max_output_len 32 \
    --num_beams 1 \
    --repetition_penalty 1.1 \
    --engine_dir ./mk_aigctitle_int4 \
    --tokenizer_dir qwen3 \
    --no_add_special_tokens \
    --input_id_format 2 \
    --sort_example 0 \
    --cuda_graph_mode --output_avglength