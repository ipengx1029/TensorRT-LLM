#ifndef KITTENS_HOPPER
#include "base.h"
#include "qwen.h"
#include "mkutil.cuh"
#include "mk_api.cuh"
#include "qwen2-quant/qwen2.cuh"

namespace mk {
template<int sms_count, int max_share_memory>
GPTQQwen2ModelInfer<sms_count, max_share_memory>::GPTQQwen2ModelInfer() {
    mAttnScale = 1.0f / sqrt(QWEN2_1P5B_HEAD_DIM);
    mRmsNormEps = 1e-06;
}
template<int sms_count, int max_share_memory>
void GPTQQwen2ModelInfer<sms_count, max_share_memory>::infer(
    MKGlobals* g, cudaStream_t stream) {
     // qwen2 1p5b
    using config = megakernel::default_config<max_share_memory, false, 12>;
    typedef quant_globals_t<config, -1, sms_count> qwen2_encoder_globals;
    typedef quant_globals_t<config, 1,  sms_count> qwen2_decoder_globals;
    if (g->encoder) { // encoder
        auto globs = make_global_configs<qwen2_encoder_globals>(g);
        mk_api::mk_func_execute(globs, stream);
    } else { // decoder
        if (g->batch_size > 1) {
            auto globs = make_global_configs<qwen2_encoder_globals>(g);
            mk_api::mk_func_execute(globs, stream);
        } else {
            auto globs = make_global_configs<qwen2_decoder_globals>(g);
            mk_api::mk_func_execute(globs, stream);
        }
    }
}
template<int sms_count, int max_share_memory>
template<typename TGlobal>
TGlobal GPTQQwen2ModelInfer<sms_count, max_share_memory>::make_global_configs(MKGlobals* glob) {
    auto &globals = *glob;
    using bar_layout = typename TGlobal::barriers;
    using ins_layout = typename TGlobal::instruction_layout;
    using time_layout = typename TGlobal::timing_layout;
    using weight_layout = typename TGlobal::weights_t;
    using lm_head_weight_layout = typename TGlobal::lm_head_weight_t;
    using norm_layout = typename TGlobal::norm_weights_t;
    using weight_big_layout = typename TGlobal::weights_big_indim_t;
    using kvcache_layout = typename TGlobal::kv_cache_t;
    using rope_layout = typename TGlobal::rope_table_t;
    using act_layout = typename TGlobal::activations_t;
    using act_silu_layout = typename TGlobal::activations_big_indim_t;
    using logits_layout = typename TGlobal::logits_t;
    using qk_norm_layout = typename TGlobal::qk_norm_weights_t;
    using bs_param_layout = typename TGlobal::batch_vec_t;
    using zeros_or_scale_layout = typename TGlobal::zeros_or_scale_t;
    using zeros_or_scale_big_indim_layout = typename TGlobal::zeros_or_scale_big_indim_t;
    using bias_layout = typename TGlobal::qkv_bias_t;
    constexpr bool only_cols = !TGlobal::encoder;
    return TGlobal(
        convert2gl<bar_layout>(globals.Bar),                // Bar torch::zeros({16, 10, 48}, torch::kInt32)
        convert2gl<ins_layout>(globals.instructions),       // torch::zeros({56, 68, 32}, torch::kInt32
        convert2gl<time_layout>(globals.timings),           // torch::zeros({56, 68, 128}, torch::kInt32)

        convert2gl<weight_layout>(globals.qkv_weights),     // torch::zeros({16, 3072, 2048/4}, torch::kBFloat16)
        convert2gl<zeros_or_scale_layout>(globals.qkv_proj_scales),
        convert2gl<bias_layout>(globals.qkv_bias),          
        convert2gl<norm_layout>(globals.attn_norm_weights),    // torch::zeros({16, 2048}, torch::kBFloat16)
        convert2gl<weight_layout>(globals.o_weights),       // torch::zeros({16, 2048, 2048/4}, torch::kBFloat16)
        convert2gl<zeros_or_scale_layout>(globals.o_proj_scales), 
        convert2gl<norm_layout>(globals.mlp_norm_weights),     // torch::zeros({16, 2048}, torch::kBFloat16)
        convert2gl<weight_layout>(globals.up_weights),      // torch::zeros({16, 8192, 2048/4}, torch::kBFloat16)
        convert2gl<zeros_or_scale_layout>(globals.up_proj_scales), 
        convert2gl<weight_layout>(globals.gate_weights),    // torch::zeros({16, 8192, 2048/4}, torch::kBFloat16)
        convert2gl<zeros_or_scale_layout>(globals.gate_proj_scales), 
        convert2gl<weight_big_layout>(globals.down_weights), // torch::zeros({16, 2048, 8192/4}, torch::kBFloat16)
        convert2gl<zeros_or_scale_big_indim_layout>(globals.down_proj_scales), 
        convert2gl<norm_layout>(globals.lm_head_norm_weights), // torch::zeros({2048}, torch::kBFloat16)
        convert2gl<lm_head_weight_layout>(globals.lm_head_weights),   // torch::zeros({128256, 2048}, torch::kBFloat16)
        // kv cache
        convert2gl_array<kvcache_layout, QWEN2_1P5B_NUM_LAYERS>(globals.kv_caches),
        // other buffers
        convert2gl<rope_layout>(globals.rope_cos),  // torch::zeros({131072, 64}, torch::kFloat32)
        convert2gl<rope_layout>(globals.rope_sin),  // torch::zeros({131072, 64}, torch::kFloat32)
        // activation buffers
        convert2gl<act_layout, only_cols>(globals.hidden_states),            // hidden_states
        // activation rms norm temp result
        convert2gl<act_layout, only_cols>(globals.rms_norm_states),         // rms_norm_states
        convert2gl<act_layout, only_cols>(globals.q_post_rope),          // q_post_rope
        convert2gl<act_layout, only_cols>(globals.attn_out),                // attn_out
        convert2gl<act_silu_layout, only_cols>(globals.silu_out),                // silu_out
        convert2gl<logits_layout, only_cols>(globals.logits),                  // logits
        globals.pos_id,
        mAttnScale,
        mRmsNormEps,
        globals.tokens_num,
        // convert2gl<qk_norm_layout>(globals.q_norm_weights),  // attn_lse_intermediates
        // convert2gl<qk_norm_layout>(globals.k_norm_weights),  // attn_out_intermediates
        convert2gl<bs_param_layout>(globals.bs_params)
    );
}
template class GPTQQwen2ModelInfer<56,  164000>; // A30
template class GPTQQwen2ModelInfer<72,  100000>; // A10
template class GPTQQwen2ModelInfer<92,  100000>; // L20 
template class GPTQQwen2ModelInfer<108, 164000>; // A100
template class GPTQQwen2ModelInfer<128,  100000>; //4090
}
#endif