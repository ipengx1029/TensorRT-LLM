""""
    qwen mk models
"""
from collections import OrderedDict
from pathlib import Path
import tensorrt as trt
from transformers import Qwen3Config
from tensorrt_llm.functional import mk_plugin, MKGlobals, MKScalesParams, Tensor, debug_print, cast
from tensorrt_llm.module import Module
from ...layers import Embedding
from ...parameter import Parameter

class Qwen3MegaKernel(Module):
    """ qwen model """
    def __init__(self, config: Qwen3Config, quant_type: str):
        """ init """
        super().__init__()
        self.vocab_embedding = Embedding(
            config.vocab_size, config.hidden_size, dtype=config.torch_dtype
        )
        self.config = config
        self.quant_type = quant_type
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.dtype = config.torch_dtype
        self.num_hidden_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.intermediate_size = config.intermediate_size
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self._init_weights()

    def _init_weights(self):
        """init weights"""
        print(f"self.dtype: {self.dtype} quant_type {self.quant_type}")
        if self.quant_type == "int4_sync_g128":
            num_elem_weight = 4
        else:
            num_elem_weight = 1
        qkv_hidden = (self.hidden_size // self.num_attention_heads) * (
            self.num_attention_heads + self.num_key_value_heads * 2
        )
        self.qkv_proj_weights = Parameter(
            shape=[self.num_hidden_layers, 
                   qkv_hidden, 
                   self.hidden_size // num_elem_weight],
            dtype=trt.bfloat16,
        )
        self.o_proj_weights = Parameter(
            shape=[self.num_hidden_layers, 
                   self.hidden_size, 
                   self.hidden_size // num_elem_weight],
            dtype=trt.bfloat16,
        )
        self.attn_ln_weights = Parameter(
            shape=[self.num_hidden_layers, self.hidden_size],
            dtype=trt.bfloat16,
        )
        self.mlp_ln_weights = Parameter(
            shape=[self.num_hidden_layers, self.hidden_size],
            dtype=trt.bfloat16,
        )
        self.up_proj_weights = Parameter(
            shape=[
                self.num_hidden_layers,
                self.intermediate_size,
                self.hidden_size // num_elem_weight,
            ],  # 28, 6144, 2048
            dtype=trt.bfloat16,
        )
        self.gate_proj_weights = Parameter(
            shape=[self.num_hidden_layers, 
                   self.intermediate_size, 
                   self.hidden_size // num_elem_weight],
            dtype=trt.bfloat16,
        )
        self.down_proj_weights = Parameter(
            shape=[self.num_hidden_layers, 
                   self.hidden_size, 
                   self.intermediate_size // num_elem_weight],
            dtype=trt.bfloat16,
        )
        self.lm_head_norm_weights = Parameter(
            shape=[self.hidden_size],
            dtype=trt.bfloat16,
        )
        self.lm_head_weights = Parameter(
            shape=[self.vocab_size, self.hidden_size],
            dtype=trt.bfloat16,
        )
        self.rope_cos = Parameter(
            shape=[self.max_position_embeddings, self.head_dim], dtype=trt.float32
        )
        self.rope_sin = Parameter(
            shape=[self.max_position_embeddings, self.head_dim], dtype=trt.float32
        )
        self.q_norm_weights = Parameter(
            shape=[self.num_hidden_layers, self.head_dim],
            dtype=trt.bfloat16,
        )
        self.k_norm_weights = Parameter(
            shape=[self.num_hidden_layers, self.head_dim],
            dtype=trt.bfloat16,
        )
        # [56, 129, 32]
        self.instructions = None
        # [28, 10, 32]
        self.barriers = None
        # [56, 129, 128]
        self.timings = None

        if self.quant_type == "int4_sync_g128":
            group_size = 128
            self.qkv_proj_scales = Parameter(
                shape=[self.num_hidden_layers, 
                       qkv_hidden, 
                       self.hidden_size // group_size],
                dtype=trt.bfloat16,
            ) #torch.Size([28, 4096, 16]) torch.bfloat16
            self.o_proj_scales = Parameter(
                shape=[self.num_hidden_layers, 
                       self.hidden_size, 
                       self.hidden_size // group_size],
                dtype=trt.bfloat16,
            ) #torch.Size([28, 2048, 16]) torch.bfloat16
            self.up_proj_scales = Parameter(
                shape=[
                    self.num_hidden_layers,
                    self.intermediate_size,
                    self.hidden_size // group_size,
                ], 
                dtype=trt.bfloat16,
            ) #torch.Size([28, 6144, 16]) torch.bfloat16
            self.gate_proj_scales = Parameter(
                shape=[self.num_hidden_layers, 
                       self.intermediate_size, 
                       self.hidden_size // group_size],
                dtype=trt.bfloat16,
            ) #torch.Size([28, 6144, 16]) torch.bfloat16
            self.down_proj_scales = Parameter(
                shape=[self.num_hidden_layers, 
                       self.hidden_size, 
                       self.intermediate_size // group_size],
                dtype=trt.bfloat16,
            ) ## torch.Size([28, 2048, 48]) torch.bfloat16

    def prepare_inputs(self, max_batch_size, max_input_len, max_output_len):
        """ prepare build inputs """
        # cache kv
        bs_range = [[1, (max_batch_size + 1) // 2, max_batch_size]]
        max_seq_len = max_input_len + max_output_len
        bs_max_input_len = max_batch_size * max_input_len
        input_range = [[1, (bs_max_input_len + 1) // 2, bs_max_input_len]]
        seq_range = [[1, (max_seq_len + 1) // 2, max_seq_len]]

        model_kwargs = {}
        model_kwargs["input_ids"] = Tensor(
            name="input_ids",
            shape=[-1],
            dtype=trt.int32,
            dim_range=OrderedDict(
                [
                    ("max_input", input_range),
                ]
            ),
        )
        model_kwargs["input_lengths"] = Tensor(
            name="input_lengths",
            shape=[-1],
            dtype=trt.int32,
            dim_range=OrderedDict(
                [
                    ("max_batch_size", bs_range),
                ]
            ),
        )
        bs_num_layers = self.num_hidden_layers * max_batch_size
        kv_dim_range = OrderedDict(
            [
                ("nlayers_mbatchsize", [bs_num_layers]),
                ("max_seq_len", seq_range),
                ("num_key_value_heads", [self.num_key_value_heads]),
                ("head_dim", [self.head_dim]),
            ]
        )
        kv_shape = [bs_num_layers, -1, self.num_key_value_heads, self.head_dim]
        model_kwargs["k_cache"] = Tensor(
            name="k_cache",
            shape=kv_shape,
            dtype=trt.bfloat16,
            dim_range=kv_dim_range,
        )
        model_kwargs["v_cache"] = Tensor(
            name="v_cache", shape=kv_shape, dtype=trt.bfloat16, dim_range=kv_dim_range
        )
        model_kwargs["bs_params"] = Tensor(
            name="bs_params",
            shape=[-1, 3],
            dtype=trt.int32,
            dim_range=OrderedDict(
                [("max_batch_size", bs_range), ("bs_params_width", [3])]
            ),
        )
        return model_kwargs

    def init_globals(self, hidden_states, input_lengths, **kwargs):
        """init globals"""
        assert self.instructions is not None, "instructions is not null"
        globs = MKGlobals(
            # hidden_states
            hidden_states,
            # input_lengths
            input_lengths,
            self.qkv_proj_weights.value,
            self.attn_ln_weights.value,
            self.o_proj_weights.value,
            self.mlp_ln_weights.value,
            self.up_proj_weights.value,
            self.gate_proj_weights.value,
            self.down_proj_weights.value,
            self.lm_head_norm_weights.value,
            self.lm_head_weights.value,
            # not stacked for each layer
            self.rope_cos.value,
            self.rope_sin.value,
            # qk norm weights
            self.q_norm_weights.value,
            self.k_norm_weights.value,
            # instructions
            self.instructions.value,
            self.timings.value,
            self.barriers.value,
            # kvcache
            kwargs.get("k_cache"),
            kwargs.get("v_cache"),
            # batch size
            kwargs.get("bs_params"),
            None
        )
        if self.quant_type == "int4_sync_g128":
            # add scales params
            globs.scales_params = MKScalesParams(
                self.qkv_proj_scales.value,
                self.o_proj_scales.value,
                self.up_proj_scales.value,
                self.gate_proj_scales.value,
                self.down_proj_scales.value,
            )
        return globs

    def forward(self, input_ids: Tensor, input_lengths: Tensor, **kwargs):
        """forward"""
        #input_ids = debug_print("input_ids", input_ids, False)
        hidden_states = self.vocab_embedding(input_ids)
        #hidden_states = debug_print("hidden_states", hidden_states, False)
        print("hidden_states=", hidden_states)
        #hidden_states.mark_output("hidden_states", hidden_states.dtype)
        globals = self.init_globals(hidden_states, input_lengths, **kwargs)
        output = mk_plugin(
            1,  # qwen model
            globals,
            self.num_attention_heads,
            self.vocab_size,
            self.intermediate_size,
            self.head_dim,
            self.num_hidden_layers,
            self.num_key_value_heads,
            self.hidden_size,
        )
        logits = cast(output, dtype=trt.float32)
        logits.mark_output("logits", trt.float32)
        print("logits=", logits)
        return logits
