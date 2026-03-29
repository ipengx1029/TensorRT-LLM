#pragma once
#include "NvInfer.h"
#include <memory>
namespace mk {
struct MKTensor {
    void* ptr;
    const nvinfer1::Dims *dim;
};
struct MKGLTensor {
    void* ptr;
    nvinfer1::Dims dim;
};
struct MKGlobals {
    MKTensor Bar;
    MKTensor instructions;
    MKTensor timings;

    // 模型权重
    MKTensor qkv_weights;
    MKTensor attn_norm_weights;
    MKTensor o_weights;
    MKTensor mlp_norm_weights;
    MKTensor up_weights;
    MKTensor gate_weights;
    MKTensor down_weights;
    MKTensor lm_head_norm_weights;
    MKTensor lm_head_weights;

    // KV缓存
    std::vector<MKTensor> kv_caches;

    // Rope表
    MKTensor rope_cos;
    MKTensor rope_sin;

    // 激活缓冲区
    MKTensor hidden_states;
    // plugin inner ptr
    MKGLTensor rms_norm_states;
    MKGLTensor q_post_rope;
    MKGLTensor attn_out;
    MKGLTensor silu_out;
    MKTensor logits;

    MKTensor q_norm_weights;
    MKTensor k_norm_weights;

    MKTensor bs_params;
    // pos id
    unsigned int pos_id;
    unsigned int tokens_num;
    // batch size
    bool encoder;
    int batch_size;

    // gptq scales
    MKTensor qkv_proj_scales;
    MKTensor o_proj_scales;
    MKTensor up_proj_scales;
    MKTensor gate_proj_scales;
    MKTensor down_proj_scales;
};
template<typename T>
inline size_t zero_mk_tensor(MKTensor &tensor, cudaStream_t stream) {
    size_t numel = 1;
    for (int i = 0; i < tensor.dim->nbDims; ++i) {
        numel *= tensor.dim->d[i];
    }
    cudaMemsetAsync(tensor.ptr, 0, sizeof(T) * numel, stream);
    return numel;
}
// model infer
class ModelInfer {
public:
    virtual ~ModelInfer() {}
    virtual void infer(MKGlobals* g, cudaStream_t stream) = 0;
};
// get model infer
extern std::shared_ptr<ModelInfer> get_model_infer(
    const int model_type, const int quant_type, const int sms_count);
} // namespace mk