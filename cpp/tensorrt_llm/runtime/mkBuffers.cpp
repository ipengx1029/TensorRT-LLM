#include "tensorrt_llm/runtime/mkBuffers.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/runtime/runtimeBuffers.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/positionIdManager.h"
#include <cstdlib>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

MKBuffers::MKBuffers() {
    k_cache = nullptr;
    v_cache = nullptr;
    logits = nullptr;
    bs_params = nullptr;
    input_lengths = nullptr;
}

MKBuffers::MKBuffers(TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig) {
    auto& manager = runtime.getBufferManager();
    nvinfer1::DataType modelDtype;
    // TODO: get modelDtype from modelConfig or engine
    modelDtype = nvinfer1::DataType::kBF16;
    k_cache = manager.emptyTensor(MemoryType::kGPU, modelDtype);
    v_cache = manager.emptyTensor(MemoryType::kGPU, modelDtype);
    logits = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
    bs_params = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    input_lengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
}

void MKBuffers::reshape(GenerationConfig const& generationConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig) {
    generation_config_ = generationConfig;
    auto const batchSize = generationConfig.batchSize;
    auto const inputLengthSum = generationConfig.inputLengthSum;
    
    TLLM_LOG_TRACE("input batchSize: %d", batchSize);
    TLLM_LOG_TRACE("input inputLengthSum: %d", inputLengthSum);
    if (inputLengthSum == 0) {
        return;
    }
    
    const auto max_tokens = modelConfig.getMaxSequenceLen();
    const auto num_layer = modelConfig.getNbAttentionLayers();
    const auto num_kv_head = modelConfig.getNbKvHeads();
    const auto head_size = modelConfig.getSizePerHead();
    const auto hidden_size = modelConfig.getHiddenSize();
    int const intermediate_size = modelConfig.getMlpHiddenSize();
    const auto vocabSize = modelConfig.getVocabSize();

    TLLM_LOG_TRACE("num_layer: %d", num_layer);
    TLLM_LOG_TRACE("num_kv_head: %d", num_kv_head);
    TLLM_LOG_TRACE("head_size: %d", head_size);
    TLLM_LOG_TRACE("hidden_size: %d", hidden_size);
    TLLM_LOG_TRACE("intermediate_size: %d", intermediate_size);
    TLLM_LOG_TRACE("vocabSize: %d", vocabSize);

    auto const kvCacheReserve = ITensor::makeShape(
        {num_layer * batchSize, max_tokens, num_kv_head, head_size}
    );

    k_cache->reshape(kvCacheReserve);
    v_cache->reshape(kvCacheReserve);
    logits->reshape(ITensor::makeShape({inputLengthSum, vocabSize}));
    bs_params->reshape(ITensor::makeShape({batchSize, 3}));
    input_lengths->reshape(ITensor::makeShape({batchSize}));
}

void MKBuffers::reshapeKvTensors(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxBlocksPerSeq, runtime::TllmRuntime const& runtime) {
    // TODO not used yet
}

void MKBuffers::setKvPoolPointers(KvCacheManager const* kvCacheManager) {
    // TODO not used yet
}

void MKBuffers::reset(BufferManager& manager)
{
    seq_len_ = 0;
}

/*
This function slices mkBuffers according to real batch size and num of tokens.
In mk plugin when establishing globals struct, 
global tensors will be set to the exact same shape as mkBuffers
so we need to slice mkBuffers in advance
*/
MKBuffers MKBuffers::sliceTo(GenerationConfig const& generationConfig, ModelConfig const& modelConfig, SizeType32 offset, SizeType32 batchSize) {    
    MKBuffers buffers;
    buffers.k_cache = ITensor::slice(k_cache, offset * modelConfig.getNbAttentionLayers(), batchSize * modelConfig.getNbAttentionLayers());
    buffers.v_cache = ITensor::slice(v_cache, offset * modelConfig.getNbAttentionLayers(), batchSize * modelConfig.getNbAttentionLayers());

    SizeType32 maxInputOffset = generationConfig.accumulatedInputLength[offset];
    SizeType32 maxInputStep = generationConfig.accumulatedInputLength[offset + batchSize] - maxInputOffset;

    buffers.logits = ITensor::slice(logits, maxInputOffset, maxInputStep);
    buffers.bs_params = ITensor::slice(bs_params, offset, batchSize);
    buffers.input_lengths = ITensor::slice(input_lengths, offset, batchSize);
    return buffers;
}

void MKBuffers::prepareContextStep(RuntimeBuffers* runtimeBuffers, TensorPtr const& inputIds, TokenIdType padId, BufferManager& manager, 
KvCacheManager const* KvCacheManager, SizeType32 firstBatchSlotIdx, 
ModelConfig const& modelConfig, WorldConfig const& worldConfig) {
    seq_len_ = make_bs_param(manager, *(runtimeBuffers->contextLengthsHost));
}

void MKBuffers::postContextStep(RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBuffers, 
BufferManager& manager, ModelConfig const& modelConfig, WorldConfig const& worldConfig) {
    kernels::gatherLastTokenLogits(*(runtimeBuffers->logits), *logits, *(runtimeBuffers->lastTokenIds), manager.getStream());
}

void MKBuffers::prepareNextStep(RuntimeBuffers* runtimeBuffers, SizeType32 step, BufferManager& manager,
    KvCacheManager* kvCacheManager, SizeType32 firstBatchSlotIdx, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig) {
    const int vocabSize = modelConfig.getVocabSize();
    const int batchSize = generation_config_.batchSize;
    int new_token_num = 1;
    int* input_lengths_ptr = (int*)input_lengths->data();
    for (int i = 0; i < batchSize; ++i) {
        input_lengths_ptr[i] = 1;
    }

    if (batchSize > 1) {
        new_token_num = update_bs_param(manager, *(runtimeBuffers->contextLengthsHost), step);
    }

    seq_len_ += new_token_num;
    runtimeBuffers->logits->reshape(ITensor::makeShape({new_token_num, 1, vocabSize})); // TODO support beamsearch
    logits = ITensor::view(runtimeBuffers->logits, ITensor::makeShape({new_token_num, vocabSize}));
}

void MKBuffers::getRuntimeBuffers(RuntimeBuffers const* runtimeBuffers, TensorMap& inputBuffers, TensorMap& outputBuffers,
    SizeType32 step, TensorPtr const& inputIds, ModelConfig const& modelConfig, WorldConfig const& worldConfig) const {
    inputBuffers.clear();
    outputBuffers.clear();

    outputBuffers.insert_or_assign("logits", ITensor::view(logits));

    inputBuffers.insert_or_assign("input_ids", inputIds);
    inputBuffers.insert_or_assign("k_cache", k_cache);
    inputBuffers.insert_or_assign("v_cache", v_cache);
    inputBuffers.insert_or_assign("bs_params", bs_params);
    inputBuffers.insert_or_assign("input_lengths", input_lengths);
}

int MKBuffers::make_bs_param(BufferManager& manager, ITensor &input_lengths_host) {
    const int batch_size = generation_config_.batchSize;
    std::vector<int> bs_params_host;
    bs_params_host.resize(3*batch_size);
    int offset = 0;
    int idx = 0;
    auto input_lengths_buffer = BufferRange<SizeType32>(input_lengths_host);
    int* input_lengths_ptr = (int*)input_lengths->data();
    for (int i = 0; i < batch_size; ++i) {
        int len = input_lengths_buffer[i];
        bs_params_host[idx++] = offset;
        bs_params_host[idx++] = len;
        bs_params_host[idx++] = 0;
        input_lengths_ptr[i] = len;
        offset += len;
    }
    bs_params = manager.copyFrom(
        bs_params_host.data(), ITensor::makeShape({batch_size, 3}), nvinfer1::DataType::kINT32, MemoryType::kGPU);
    return offset;
}

int MKBuffers::update_bs_param(BufferManager& manager, ITensor &input_lengths_host, SizeType32 new_token_num) {
    const int batch_size = generation_config_.batchSize;
    std::vector<int> bs_params_host;
    bs_params_host.resize(3*batch_size);
    int idx = 0;
    auto input_lengths_buffer = BufferRange<SizeType32>(input_lengths_host);
    for (int i = 0; i < batch_size; ++i) {
        int len = input_lengths_buffer[i];
        bs_params_host[idx++] = i;
        bs_params_host[idx++] = 1;
        bs_params_host[idx++] = len + new_token_num;
    }
    bs_params = manager.copyFrom(
        bs_params_host.data(), ITensor::makeShape({batch_size, 3}), nvinfer1::DataType::kINT32, MemoryType::kGPU);
    return batch_size;
}