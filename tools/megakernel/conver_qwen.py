import argparse
import json
import os
import safetensors
import tensorrt_llm
import torch
import tensorrt as trt
from tensorrt_llm._utils import  numpy_to_torch
from tensorrt_llm.models import Qwen3MegaKernel, Qwen2MegaKernel
from tensorrt_llm import Builder, Parameter
from tensorrt_llm.network import net_guard
from transformers import Qwen3Config, Qwen2Config
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.version import __version__
import sys
sys.path.append('../3rdparty/tk/megakernels')
from megakernels.model_types import ExtraModelConfig
from megakernels.qwen3 import Qwen3ForCausalLM
from megakernels.qwen2 import Qwen2ForCausalLM
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
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--max_input_len', type=int, default=160)
    parser.add_argument('--max_output_len', type=int, default=80)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./mk_engine')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16'])
    parser.add_argument('--qformat', type=str, default='bfloat16', 
                        choices=['bfloat16', 'int4_sync_g128'])
    parser.add_argument('--qwen_version', type=str, default='qwen3', choices=['qwen2', 'qwen3'])
    return parser.parse_args()

def load_weights_from_hf(model_dir, extra_config, qwen_version):
    # choose model from config
    if qwen_version == 'qwen2':
        model = Qwen2ForCausalLM.from_pretrained(
            model_dir, extra_config=extra_config
        )
    elif qwen_version == 'qwen3':
        model = Qwen3ForCausalLM.from_pretrained(
            model_dir, extra_config=extra_config
        )
    print("model loaded")
    schedule_builder = make_schedule_builder('latency')
    print("get schedule_builder")
    schedule = schedule_builder.build(model, qformat=extra_config.qformat)
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
    }
    if qwen_version == 'qwen3':
        weights.update({
            'q_norm_weights':schedule.globs.q_norm_weights,
            'k_norm_weights':schedule.globs.k_norm_weights,
        })
    else:
        weights.update({
            'qkv_proj_bias':schedule.globs.qkv_proj_bias,
        })
    # add quant scales if they exist
    if extra_config.qformat == "int4_sync_g128":
        weights["qkv_proj_scales"] = schedule.globs.qkv_proj_scales
        weights["o_proj_scales"] = schedule.globs.o_proj_scales
        weights["up_proj_scales"] = schedule.globs.up_proj_scales
        weights["gate_proj_scales"] = schedule.globs.gate_proj_scales
        weights["down_proj_scales"] = schedule.globs.down_proj_scales

    return weights

def convert2model(model, weights, extra_config):
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
    model.barriers = Parameter(value=weights['barriers'].to('cuda'), shape=weights['barriers'].shape)
    model.instructions = Parameter(value=weights['instructions'].to('cuda'), shape=weights['instructions'].shape)
    model.timings = Parameter(value=weights['timings'].to('cuda'), shape=weights['timings'].shape)
    model.vocab_embedding.weight.value = weights['embeddings'].to('cuda')
    if isinstance(model, Qwen2MegaKernel):
        model.qkv_proj_bias.value = weights['qkv_proj_bias'].to('cuda')
    if isinstance(model, Qwen3MegaKernel):
        model.q_norm_weights.value = weights['q_norm_weights'].to('cuda')
        model.k_norm_weights.value = weights['k_norm_weights'].to('cuda')
    if extra_config.qformat == "int4_sync_g128":
        print("load quant weight params")
        model.qkv_proj_scales.value = weights["qkv_proj_scales"].to('cuda')
        model.o_proj_scales.value = weights["o_proj_scales"].to('cuda')
        model.up_proj_scales.value = weights["up_proj_scales"].to('cuda')
        model.gate_proj_scales.value = weights["gate_proj_scales"].to('cuda')
        model.down_proj_scales.value = weights["down_proj_scales"].to('cuda')
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
    if args.qwen_version == 'qwen2':
        config : Qwen2Config = Qwen2Config.from_pretrained(args.model_dir)
        config.head_dim = config.hidden_size // config.num_attention_heads
    elif args.qwen_version == 'qwen3':
        config : Qwen3Config = Qwen3Config.from_pretrained(args.model_dir)
    max_seq_len = args.max_input_len + args.max_output_len 
    extra_config = ExtraModelConfig(
        interleave_rope=False,
        max_len_override = max_seq_len,
        max_batch_size = args.max_batch_size,
        qformat=args.qformat
    )
    #TODO: choose model from config
    if args.qwen_version == 'qwen2':
        model = Qwen2MegaKernel(config, args.qformat)
    elif args.qwen_version == 'qwen3':
        model = Qwen3MegaKernel(config, args.qformat)
    weights = load_weights_from_hf(args.model_dir, extra_config, args.qwen_version)
    convert2model(model, weights, extra_config)
    return model

def main():
    args = parse_arguments()
    print("args=", args)

    model = from_huggin_face(args)

    print("model.dtype=", model.dtype)
    print("type of model.dtype=", type(model.dtype))
    tensorrt_llm.logger.set_level("info")
    builder = Builder()
    max_seq_len = args.max_input_len + args.max_output_len
    builder_config = builder.create_builder_config(  # 需要仔细研究
        precision=model.dtype,
        strongly_typed=True,
        int8=False,
        fp8=False,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        max_beam_width=args.max_beam_width,
        max_num_tokens=max_seq_len,
        opt_num_tokens=max_seq_len // 2,
        remove_input_padding=True,
    )
    print("init builder finished")
    network = builder.create_network()
    #network.plugin_config.debug_print = True
    network.plugin_config.remove_input_padding = True
    network.plugin_config.megakernel_model = True
    network.plugin_config.paged_kv_cache = False
    network.plugin_config.set_lookup_plugin("bfloat16")
    with net_guard(network):
        # 在net_guard上下文中初始化Tensor
        network.set_named_parameters(model.named_parameters())
        model_kwargs = model.prepare_inputs(
            args.max_batch_size, args.max_input_len, args.max_output_len, args.max_beam_width)
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

    dict_config = builder_config.to_dict()
    build_config = BuildConfig.from_dict(
        dict_config["builder_config"], 
        plugin_config=builder_config.plugin_config)
    hf_config = model.config
    config = {
        'version': __version__,
        'pretrained_config': {
            'architecture': 'Qwen3ForCausalLM',
            'dtype': "bfloat16",
            'logits_dtype': 'float32',
            'num_hidden_layers': hf_config.num_hidden_layers,
            'num_attention_heads': hf_config.num_attention_heads,
            'num_key_value_heads': hf_config.num_key_value_heads,
            'hidden_size': hf_config.hidden_size,
            'intermediate_size': hf_config.intermediate_size,
            'vocab_size': hf_config.vocab_size,
            'head_dim': hf_config.head_dim,
            'embedding_vocab_size': hf_config.vocab_size,
            'max_position_embeddings': hf_config.max_position_embeddings,
            'hidden_act': hf_config.hidden_act,
            'initializer_range': hf_config.initializer_range,
            'quantization': {
                'quant_algo': None,
                'kv_cache_quant_algo': None,
                'exclude_modules': ['lm_head', 'nlu_head', 'context_feature_model']
            },
            'mapping': {
                'world_size': 1,
                'tp_size': 1,
                'pp_size': 1,
            },
            'rms_norm_eps': hf_config.rms_norm_eps,
            'rope_theta': hf_config.rope_theta,
            'nlu_heads': getattr(hf_config, "nlu_heads", []),
            'context_features_size': getattr(hf_config, "context_features_size", 0),
        },
        "build_config" : build_config.to_dict()
    }
    print("config=", config)
    with open(os.path.join(args.output_dir, 'config.json'), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"save model engine to {args.output_dir} success")


if __name__ == '__main__':
    main()
