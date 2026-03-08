#include "mkPlugin.h"
#include "NvInfer.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "../common/debug_utils.h"
using namespace nvinfer1;
namespace tensorrt_llm {
namespace plugins {

const char *MK_PLUGIN_NAME = "MKPlugin";
const char *MK_PLUGIN_VERSION = "1";

// cuda device property
static cudaDeviceProp &get_device_prop() {
    static cudaDeviceProp prop;
    static bool init = false;
    if (!init) {
        cudaGetDeviceProperties(&prop, 0);
        init = true;
    }
    return prop;
}
// get count of streaming multiprocessors (SM)
static inline int get_sms_count(void) {
    auto &prop = get_device_prop();
    return prop.multiProcessorCount;
}

MkPlugin::MkPlugin(int model_type, int numHeads, int vocabSize,
                   int intermediateSize, int headDim, int numHiddenLayers,
                   int numKeyvalueHeads, int hiddenSize)
    : mModelType(model_type), mNumHeads(numHeads), mVocabSize(vocabSize),
      mIntermediateSize(intermediateSize), mHeadDim(headDim),
      mNumHiddenLayers(numHiddenLayers), mNumKeyvalueHeads(numKeyvalueHeads),
      mHiddenSize(hiddenSize) {
    int sms_count = get_sms_count();
    mModelInfer = mk::get_model_infer(model_type, sms_count);
    TLLM_CHECK_WITH_INFO(mModelInfer != nullptr, 
        "MKPLugin get model infer nullptr error");
    TLLM_LOG_WARNING("initialize mkplugin sms_count=%d, model_type=%d", 
        sms_count, model_type);
    mNumInputs = 0;
    mParams.pos_id = 0;
    mParams.tokens_num = 0;
    mNextPosId = 0;
}

MkPlugin::MkPlugin(const void *data, size_t length) {
    char const *d = reinterpret_cast<char const *>(data);
    auto const *a = d;
    read(d, mModelType);
    read(d, mNumHeads);
    read(d, mVocabSize);
    read(d, mIntermediateSize);
    read(d, mHeadDim);
    read(d, mNumHiddenLayers);
    read(d, mNumKeyvalueHeads);
    read(d, mHiddenSize);
    TLLM_CHECK_WITH_INFO(
        d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int)length, (int)(d - a));
}
const char *MkPlugin::getPluginType() const noexcept { return MK_PLUGIN_NAME; }
const char *MkPlugin::getPluginVersion() const noexcept {
    return MK_PLUGIN_VERSION;
}
int MkPlugin::getNbOutputs() const noexcept { return 1; }
int MkPlugin::initialize() noexcept {
    return 0;
}
void MkPlugin::terminate() noexcept {}

size_t MkPlugin::getSerializationSize() const noexcept {
    return sizeof(mModelType) + sizeof(mNumHeads) + sizeof(mVocabSize) +
           sizeof(mIntermediateSize) + sizeof(mHeadDim) +
           sizeof(mNumHiddenLayers) + sizeof(mNumKeyvalueHeads) +
           sizeof(mHiddenSize);
}

void MkPlugin::serialize(void *buffer) const noexcept {
    char *d = static_cast<char *>(buffer);
    char *a = d;
    write(d, mModelType);
    write(d, mNumHeads);
    write(d, mVocabSize);
    write(d, mIntermediateSize);
    write(d, mHeadDim);
    write(d, mNumHiddenLayers);
    write(d, mNumKeyvalueHeads);
    write(d, mHiddenSize);
    assert(d == a + getSerializationSize());
}
void MkPlugin::destroy() noexcept {
    delete this;
}

IPluginV2DynamicExt *MkPlugin::clone() const noexcept {
    return new MkPlugin(mModelType, mNumHeads, mVocabSize, mIntermediateSize,
                        mHeadDim, mNumHiddenLayers, mNumKeyvalueHeads,
                        mHiddenSize);
}

void MkPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

const char *MkPlugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

DataType MkPlugin::getOutputDataType(int index, const DataType *inputTypes,
                                     int nbInputs) const noexcept {
    return inputTypes[0];
}

DimsExprs MkPlugin::getOutputDimensions(int outputIndex,
                                        const DimsExprs *inputs, int nbInputs,
                                        IExprBuilder &exprBuilder) noexcept {
    DimsExprs out_logits;
    out_logits.nbDims = 2;
    out_logits.d[0] = inputs[0].d[0]; // hidden_states [batch_size, hidden_dim]
    out_logits.d[1] = exprBuilder.constant(mVocabSize);
    return out_logits;
}
bool MkPlugin::supportsFormatCombination(int pos, const PluginTensorDesc *inOut,
                                         int nbInputs, int nbOutputs) noexcept {
    if (pos == 1 || pos == 2 || pos == 3 || pos == 4 || pos == 18) {
        // input_lengths, barrier, instructs, timeing, params
        return (inOut[pos].format == TensorFormat::kLINEAR && 
            inOut[pos].type == nvinfer1::DataType::kINT32);
    } else if (pos == 14 || pos == 15) { // rope_cos, rope_sin float32
        return (inOut[pos].format == TensorFormat::kLINEAR &&
            inOut[pos].type == nvinfer1::DataType::kFLOAT);
    }
    return (inOut[pos].format == TensorFormat::kLINEAR && 
        inOut[pos].type == nvinfer1::DataType::kBF16);
}
void MkPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs,
                               const DynamicPluginTensorDesc *out,
                               int nbOutputs) noexcept {
    mNumInputs = nbInputs;
}

size_t MkPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs,
                                  const PluginTensorDesc *outputs,
                                  int nbOutputs) const noexcept {
    size_t workspace_size = 0;
    // rms_norm_states, post_ln_rope_q, attn_out, silu_out
    int max_batch_size = inputs[0].dims.d[0];
    size_t byte_size = tensorrt_llm::runtime::BufferDataType(inputs[0].type).getSize();
    workspace_size = max_batch_size * (mHiddenSize * 3 + mIntermediateSize) * byte_size;
    printf("getWorkspaceSize max_batch_size=%d, workspace_size=%lu\n", max_batch_size, workspace_size);
    // alloc workspace size                            
    return workspace_size;
}
// set mk templ gl tensor
void MkPlugin::set_mk_gl_tensor(
    const int token_nums, const nvinfer1::PluginTensorDesc *inputs, void *workspace) {
    int8_t* ptr = reinterpret_cast<int8_t*>(workspace);
    size_t byte_size = tensorrt_llm::runtime::BufferDataType(inputs[0].type).getSize();
    size_t temp_byte_size = token_nums * mHiddenSize * byte_size;
    // rms_norm_states [bs, hidden_dim]
    mParams.rms_norm_states.dim.nbDims = 2;
    mParams.rms_norm_states.dim.d[0] = token_nums;
    mParams.rms_norm_states.dim.d[1] = mHiddenSize;
    mParams.rms_norm_states.ptr = (void *)ptr;
    ptr += temp_byte_size;
    // post_ln_rope_q [bs, hidden_dim]
    mParams.q_post_rope.dim.nbDims = 2;
    mParams.q_post_rope.dim.d[0] = token_nums;
    mParams.q_post_rope.dim.d[1] = mHiddenSize;
    mParams.q_post_rope.ptr = (void *)ptr;
    ptr += temp_byte_size;
    // attn_out [bs, hidden_dim]
    mParams.attn_out.dim.nbDims = 2;
    mParams.attn_out.dim.d[0] = token_nums;
    mParams.attn_out.dim.d[1] = mHiddenSize;
    mParams.attn_out.ptr = (void *)ptr;
    ptr += temp_byte_size;
    // silu_out [bs, inner_hidden_dim]
    mParams.silu_out.dim.nbDims = 2;
    mParams.silu_out.dim.d[0] = token_nums;
    mParams.silu_out.dim.d[1] = mIntermediateSize;
    mParams.silu_out.ptr = (void *)ptr;
}
int MkPlugin::enqueue(const PluginTensorDesc *inputDesc,
                      const PluginTensorDesc *outputDesc,
                      const void *const *inputs, void *const *outputs,
                      void *workspace, cudaStream_t stream) noexcept {
    // host buffer
    int batch_size = inputDesc[1].dims.d[0];
    int *input_lengths = (int *)inputs[1];
    // pos id
    int tokens_num = 0;
    for (int bs = 0; bs < batch_size; ++bs) {
        tokens_num += input_lengths[bs];
    }
    // auto add pos id
    if (tokens_num == batch_size) { // decoder
        mParams.encoder = false;
        mParams.pos_id = mNextPosId;
        mNextPosId += input_lengths[0];
    } else { // encoder
        mParams.encoder = true;
        mParams.pos_id = 0;
        mNextPosId = input_lengths[0];
    }
    mParams.batch_size = batch_size;
    mParams.tokens_num = tokens_num;
    // set mk gl temp buffer
    set_mk_gl_tensor(tokens_num, inputDesc, workspace);
    // printf("batch size=%d, tokens_num=%d, next_pos_id=%d, input ptr=%lu\n", 
    //     batch_size, tokens_num, mNextPosId, (uint64_t)inputs[0]);

    mParams.hidden_states = {(void *)inputs[0], &inputDesc[0].dims};
    mParams.instructions = {(void *)inputs[2], &inputDesc[2].dims};
    mParams.timings = {(void *)inputs[3], &inputDesc[3].dims};
    mParams.Bar = {(void *)inputs[4], &inputDesc[4].dims};
    // reset barrier
    mk::zero_mk_tensor<int>(mParams.Bar, stream);

    // 模型权重
    mParams.qkv_weights = {(void *)inputs[5], &inputDesc[5].dims};
    mParams.attn_norm_weights = {(void *)inputs[6], &inputDesc[6].dims};
    mParams.o_weights = {(void *)inputs[7], &inputDesc[7].dims};
    mParams.mlp_norm_weights = {(void *)inputs[8], &inputDesc[8].dims};
    mParams.up_weights = {(void *)inputs[9], &inputDesc[9].dims};
    mParams.gate_weights = {(void *)inputs[10], &inputDesc[10].dims};
    mParams.down_weights = {(void *)inputs[11], &inputDesc[11].dims};
    mParams.lm_head_norm_weights = {(void *)inputs[12], &inputDesc[12].dims};
    mParams.lm_head_weights = {(void *)inputs[13], &inputDesc[13].dims};
    // Rope表
    mParams.rope_cos = {(void *)inputs[14], &inputDesc[14].dims};
    mParams.rope_sin = {(void *)inputs[15], &inputDesc[15].dims};

    // KV缓存
    mParams.k_cache = {(void *)inputs[16], &inputDesc[16].dims};
    mParams.v_cache = {(void *)inputs[17], &inputDesc[17].dims};
    mParams.bs_params = {(void *)inputs[18], &inputDesc[18].dims};

    // Qwen qkv norm
    if (mModelType == 1) {
        TLLM_CHECK_WITH_INFO(mNumInputs == 21, "MKPLugin qwen model need 21 inputs");
        mParams.q_norm_weights = {(void *)inputs[19], &inputDesc[19].dims};
        mParams.k_norm_weights = {(void *)inputs[20], &inputDesc[20].dims};
    }
    mParams.logits = {(void *)outputs[0], &outputDesc[0].dims};
    // model infer
    mModelInfer->infer(&mParams, stream);

    return 0;
}

::std::vector<PluginField> MkPluginCreator::mPluginAttributes{
    {"model_type", nullptr, PluginFieldType::kINT32, 1},
    {"num_heads", nullptr, PluginFieldType::kINT32, 1},
    {"vocab_size", nullptr, PluginFieldType::kINT32, 1},
    {"intermediate_size", nullptr, PluginFieldType::kINT32, 1},
    {"head_dim", nullptr, PluginFieldType::kINT32, 1},
    {"num_hidden_layers", nullptr, PluginFieldType::kINT32, 1},
    {"num_keyvalue_heads", nullptr, PluginFieldType::kINT32, 1},
    {"hidden_size", nullptr, PluginFieldType::kINT32, 1},
};

PluginFieldCollection MkPluginCreator::mFC{
    static_cast<int>(mPluginAttributes.size()), mPluginAttributes.data()};

MkPluginCreator::MkPluginCreator() {}

const char *MkPluginCreator::getPluginName() const noexcept {
    return MK_PLUGIN_NAME;
}

const char *MkPluginCreator::getPluginVersion() const noexcept {
    return MK_PLUGIN_VERSION;
}

const PluginFieldCollection *MkPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2 *
MkPluginCreator::createPlugin(const char *name,
                              const PluginFieldCollection *fc) noexcept {
    int model_type, numHeads = 0, headSize = 0, vocabSize = 0,
                    intermediateSize = 0;
    int headDim = 0, numHiddenLayers = 0, numKeyvalueHeads = 0, hiddenSize = 0;
    for (int i = 0; i < fc->nbFields; ++i) {
        ::std::string fname(fc->fields[i].name);
        if (fname == "model_type") {
            model_type = *static_cast<const int *>(fc->fields[i].data);
        }
        if (fname == "num_heads") {
            numHeads = *static_cast<const int *>(fc->fields[i].data);
        }
        if (fname == "vocab_size") {
            vocabSize = *static_cast<const int *>(fc->fields[i].data);
        }
        if (fname == "intermediate_size") {
            intermediateSize = *static_cast<const int *>(fc->fields[i].data);
        }
        if (fname == "head_dim") {
            headDim = *static_cast<const int *>(fc->fields[i].data);
        }
        if (fname == "num_hidden_layers") {
            numHiddenLayers = *static_cast<const int *>(fc->fields[i].data);
        }
        if (fname == "num_keyvalue_heads") {
            numKeyvalueHeads = *static_cast<const int *>(fc->fields[i].data);
        }
        if (fname == "hidden_size") {
            hiddenSize = *static_cast<const int *>(fc->fields[i].data);
        }
    }
    return new MkPlugin(model_type, numHeads, vocabSize, intermediateSize,
                        headDim, numHiddenLayers, numKeyvalueHeads, hiddenSize);
}

IPluginV2 *MkPluginCreator::deserializePlugin(const char *name,
                                              const void *serialData,
                                              size_t serialLength) noexcept {
    return new MkPlugin(serialData, serialLength);
}

void MkPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

const char *MkPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

// 07.21: 必须注册宏
REGISTER_TENSORRT_PLUGIN(MkPluginCreator);

} // namespace plugins
} // namespace tensorrt_llm
