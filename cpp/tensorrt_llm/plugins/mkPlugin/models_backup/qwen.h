#pragma once
namespace mk {
// qwen normal
template<int sms_count, int max_share_memory>
class QwenModelInfer : public ModelInfer {
public:
    QwenModelInfer();
    void infer(MKGlobals* g, cudaStream_t stream);
private:
    template<typename TGlobal>
    TGlobal make_global_configs(MKGlobals* glob);
private:
    float mAttnScale;
    float mRmsNormEps;
};
// qwen gpt quant
template<int sms_count, int max_share_memory>
class GPTQQwenModelInfer : public ModelInfer {
public:
    GPTQQwenModelInfer();
    void infer(MKGlobals* g, cudaStream_t stream);
private:
    template<typename TGlobal>
    TGlobal make_global_configs(MKGlobals* glob);
private:
    float mAttnScale;
    float mRmsNormEps;
};
}