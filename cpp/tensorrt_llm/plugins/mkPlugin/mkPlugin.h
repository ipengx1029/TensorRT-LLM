#pragma once

#include "NvInfer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "models/base.h"

namespace tensorrt_llm {
namespace plugins {
class MkPlugin : public BasePlugin {
public:
    MkPlugin(int model_type, int quant_type, int numHeads, int vocabSize, int intermediateSize,
             int headDim, int numHiddenLayers, int numKeyvalueHeads,
             int hiddenSize);
    MkPlugin(const void *data, size_t length);
    ~MkPlugin() override = default;

    const char *getPluginType() const noexcept override;
    const char *getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void *buffer) const noexcept override;
    void destroy() noexcept override;
    nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
    void setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *getPluginNamespace() const noexcept override;

    nvinfer1::DataType getOutputDataType(int index,
                                         const nvinfer1::DataType *inputTypes,
                                         int nbInputs) const noexcept override;

    nvinfer1::DimsExprs
    getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                        int nbInputs,
                        nvinfer1::IExprBuilder &exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos,
                                   const nvinfer1::PluginTensorDesc *inOut,
                                   int nbInputs,
                                   int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                         int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc *out,
                         int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                            int nbInputs,
                            const nvinfer1::PluginTensorDesc *outputs,
                            int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                const nvinfer1::PluginTensorDesc *outputDesc,
                const void *const *inputs, void *const *outputs,
                void *workspace, cudaStream_t stream) noexcept override;
  private:
    // set mk templ gl tensor
    void set_mk_gl_tensor(const int token_nums, 
        const nvinfer1::PluginTensorDesc *input, void *workspace);
    // update gptq quant tensor
    void update_gptq_gl_tensor(const int start_idx, 
        const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs);

  private:
    int mModelType;
    int mQuantType;
    int mNumHeads;
    int mVocabSize;
    int mIntermediateSize;
    int mHeadDim;
    int mNumHiddenLayers;
    int mNumKeyvalueHeads;
    int mHiddenSize;
    std::string mNamespace;
    std::shared_ptr<mk::ModelInfer> mModelInfer;
    mk::MKGlobals mParams;
    int mNumInputs;
    int mNextPosId;
};

class MkPluginCreator : public BaseCreator {
  public:
    MkPluginCreator();
    ~MkPluginCreator() override = default;

    const char *getPluginName() const noexcept override;
    const char *getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection *getFieldNames() noexcept override;
    nvinfer1::IPluginV2 *
    createPlugin(const char *name,
                 const nvinfer1::PluginFieldCollection *fc) noexcept override;
    nvinfer1::IPluginV2 *
    deserializePlugin(const char *name, const void *serialData,
                      size_t serialLength) noexcept override;
    void setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *getPluginNamespace() const noexcept override;

  private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace tensorrt_llm
