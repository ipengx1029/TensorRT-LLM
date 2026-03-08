import argparse
import json
import os
import safetensors
import tensorrt_llm
import torch
import tensorrt as trt
from tensorrt_llm._utils import  numpy_to_torch
from tensorrt_llm.models import Qwen3MegaKernel
from tensorrt_llm import Builder, Parameter
from tensorrt_llm.network import net_guard
from transformers import Qwen3Config
import sys
sys.path.append('../3rdparty/tk/megakernels')
from megakernels.model_types import ExtraModelConfig
from megakernels.qwen3 import Qwen3ForCausalLM
from megakernels.model_types import ExtraModelConfig
from megakernels.scheduler import (
    assign_to_sms,
    tensorize_instructions,
)
from megakernels.dispatch import (
    make_schedule_builder,
)

def get_engine_name(rank):
    return 'rank{}.engine'.format(rank)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--max_batch_size', type=int, default=2)
    parser.add_argument('--max_input_len', type=int, default=100)
    parser.add_argument('--max_output_len', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='./mk_engine')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dtype',
                        type=str,
                        default='bfloat16',
                        choices=['bfloat16'])
    return parser.parse_args()

def load_weights_from_hf(model_dir, extra_config):
    model = Qwen3ForCausalLM.from_pretrained(
        model_dir, extra_config=extra_config
    )
    print("model loaded")
    schedule_builder = make_schedule_builder('latency')
    print("get schedule_builder")
    schedule = schedule_builder.build(model)
    print("schedule built")
    assigned_to_sms = assign_to_sms(
        'rr', schedule=schedule
    )
    print("assigned to sms")
    tensorize_instructions(schedule.globs, assigned_to_sms)
    print("get instructions")

    weights = {
        'qkv_proj_weights':schedule.globs.qkv_proj_weights,
        'o_proj_weights':schedule.globs.o_proj_weights,
        'attn_ln_weights':schedule.globs.attn_ln_weights,
        'mlp_ln_weights':schedule.globs.mlp_ln_weights,
        'up_proj_weights':schedule.globs.up_proj_weights,
        'gate_proj_weights':schedule.globs.gate_proj_weights,
        'down_proj_weights':schedule.globs.down_proj_weights,
        'lm_head_norm_weights':schedule.globs.lm_head_norm_weights,
        'lm_head_weights':schedule.globs.lm_head_weights,
        'rope_cos':schedule.globs.rope_cos,
        'rope_sin':schedule.globs.rope_sin,
        'barriers':schedule.globs.barriers,
        'instructions':schedule.globs.instructions,
        'timings':schedule.globs.timings,
        'embeddings': model.model.embed_tokens.embed_tokens.weight,
        'q_norm_weights':schedule.globs.q_norm_weights,
        'k_norm_weights':schedule.globs.k_norm_weights,
    }

    return weights

def convert2model(model, weights):
    print("loading weight to trt model")
    model.qkv_proj_weights.value = weights['qkv_proj_weights'].to('cuda')
    model.o_proj_weights.value = weights['o_proj_weights'].to('cuda')
    model.attn_ln_weights.value = weights['attn_ln_weights'].to('cuda')
    model.mlp_ln_weights.value = weights['mlp_ln_weights'].to('cuda')
    model.up_proj_weights.value = weights['up_proj_weights'].to('cuda')
    model.gate_proj_weights.value = weights['gate_proj_weights'].to('cuda')
    model.down_proj_weights.value = weights['down_proj_weights'].to('cuda')
    model.lm_head_norm_weights.value = weights['lm_head_norm_weights'].to('cuda')
    model.lm_head_weights.value = weights['lm_head_weights'].to('cuda')
    model.rope_cos.value = weights['rope_cos'].to('cuda')
    model.rope_sin.value = weights['rope_sin'].to('cuda')
    model.q_norm_weights.value = weights['q_norm_weights'].to('cuda')
    model.k_norm_weights.value = weights['k_norm_weights'].to('cuda')
    model.barriers = Parameter(value=weights['barriers'].to('cuda'), shape=weights['barriers'].shape)
    model.instructions = Parameter(value=weights['instructions'].to('cuda'), shape=weights['instructions'].shape)
    model.timings = Parameter(value=weights['timings'].to('cuda'), shape=weights['timings'].shape)
    model.vocab_embedding.weight.value = weights['embeddings'].to('cuda')
    print("loaded weight to trt model")

def save_checkpoint(model, output_dir, save_config=True):
    rank = 0
    weights = {
        name: numpy_to_torch(param.raw_value) 
        for name, param in model.named_parameters()
    }
    from safetensors.torch import save_file
    save_file(weights, os.path.join(output_dir, f'rank{rank}.safetensors'))
    if save_config:
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(model.config.to_dict(), f, indent=4)

def from_huggin_face(args):
    config : Qwen3Config = Qwen3Config.from_pretrained(args.model_dir)
    max_seq_len = args.max_input_len + args.max_output_len 
    extra_config = ExtraModelConfig(
        interleave_rope=True,
        max_len_override = max_seq_len,
        max_batch_size = args.max_batch_size,
    )
    model = Qwen3MegaKernel(config)
    weights = load_weights_from_hf(args.model_dir, extra_config)
    convert2model(model, weights)
    return model

def main():
    args = parse_arguments()
    print("args=", args)

    model = from_huggin_face(args)

    print("model.dtype=", model.dtype)
    print("type of model.dtype=", type(model.dtype))
    tensorrt_llm.logger.set_level("info")
    builder = Builder()
    builder_config = builder.create_builder_config(  # 需要仔细研究
        precision=model.dtype,
        strongly_typed=True,
        int8=False,
        fp8=False,
    )
    print("init builder finished")
    network = builder.create_network()
    #network.plugin_config.debug_print = True
    network.plugin_config.remove_input_padding = True
    network.plugin_config.megakernel_model = True
    network.plugin_config.set_lookup_plugin("bfloat16")
    with net_guard(network):
        # 在net_guard上下文中初始化Tensor
        network.set_named_parameters(model.named_parameters())
        model_kwargs = model.prepare_inputs(
            args.max_batch_size, args.max_input_len, args.max_output_len)
        print("model_kwargs=", model_kwargs)
        # 使用mkLlamaPlugin，传递所有参数
        model(**model_kwargs)
        # 标记输出
        print("mark output finished")

    with net_guard(network):
        network.to_dot(f'rank{args.rank}.dot')

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    assert engine is not None, 'Failed to build engine.'
    engine_name = get_engine_name(args.rank)
    engine_file = os.path.join(args.output_dir, engine_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(engine_file, 'wb') as f:
        f.write(engine)
    builder_config.engine_name = engine_name
    builder_config.precision = "bfloat16"
    builder.save_config(builder_config,
                        os.path.join(args.output_dir, 'config.json'))
    print(f"save model engine to {args.output_dir} success")


if __name__ == '__main__':
    main()
