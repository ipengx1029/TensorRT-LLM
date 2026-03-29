#!/bin/sh

set -x
source ../hack/env_bashrc

python3 megakernels/convert_qwen.py \
        --model_dir=qwen3_1p7b_int4_gptq \
        --qformat=int4_sync_g128 \
        --output_dir=./mk_int4 \
        --max_batch_size=4