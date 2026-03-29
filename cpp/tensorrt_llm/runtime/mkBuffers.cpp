#include "tensorrt_llm/runtime/mkBuffers.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/runtime/runtimeBuffers.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/positionIdManager.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"
#include <cstdlib>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

MKBuffers::MKBuffers() {
    presentKeysVals.clear();
    k_cache = nullptr;
    v_cache = nullptr;
    logits_from_mk_buffer = nullptr;
    logits_ptr_in_use = nullptr;
    bs_params = nullptr;
    input_lengths = nullptr;
    bs_host_params = nullptr;
}

MKBuffers::MKBuffers(TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig, bool createKV) {
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& manager = runtime.getBufferManager();

    if (createKV) {
        nvinfer1::DataType modelDtype = nvinfer1::DataType::kBF16;
        auto const localNbLayers = modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism());
        presentKeysVals = utils::createBufferVector(runtime, localNbLayers, MemoryType::kGPU, modelDtype);
    }

    logits_from_mk_buffer = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
    logits_ptr_in_use = logits_from_mk_buffer;
    bs_params = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    input_lengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    bs_host_params = manager.emptyTensor(MemoryType::kPINNED, nvinfer1::DataType::kINT32);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

MKBuffers::MKBuffers(TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
    : MKBuffers(runtime, modelConfig, worldConfig, true) {
}

MKBuffers::MKBuffers(TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig, std::vector<TensorPtr>& pastKV)
    : MKBuffers(runtime, modelConfig, worldConfig, false) {
    presentKeysVals = pastKV;
}

void MKBuffers::reshape(GenerationConfig const& generationConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig) {
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    generation_config_ = generationConfig;
    auto const batchSize = generationConfig.batchSize;
    auto const inputLengthSum = generationConfig.inputLengthSum;
    auto const maxAttentionWindow = generationConfig.maxAttentionWindow;
    
    TLLM_LOG_TRACE("input batchSize: %d", batchSize);
    TLLM_LOG_TRACE("input inputLengthSum: %d", inputLengthSum);
    if (inputLengthSum == 0) {
        return;
    }
    const auto vocabSize = modelConfig.getVocabSize();

    auto const kvCacheReserve = ITensor::makeShape(
        {batchSize, 2, modelConfig.getNbKvHeads(), maxAttentionWindow, modelConfig.getSizePerHead()});
    utils::reshapeBufferVector(presentKeysVals, kvCacheReserve);
    logits_from_mk_buffer->reshape(ITensor::makeShape({inputLengthSum, vocabSize}));
    logits_ptr_in_use = logits_from_mk_buffer;
    bs_params->reshape(ITensor::makeShape({batchSize, 3}));
    bs_host_params->reshape(ITensor::makeShape({batchSize, 3}));
    input_lengths->reshape(ITensor::makeShape({batchSize + 1}));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
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
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    MKBuffers buffers;
    buffers.presentKeysVals = utils::sliceBufferVector(presentKeysVals, offset, batchSize);

    SizeType32 maxInputOffset = generationConfig.accumulatedInputLength[offset];
    SizeType32 maxInputStep = generationConfig.accumulatedInputLength[offset + batchSize] - maxInputOffset;

    buffers.logits_ptr_in_use = ITensor::slice(logits_ptr_in_use, maxInputOffset, maxInputStep);
    buffers.bs_params = ITensor::slice(bs_params, offset, batchSize);
    buffers.input_lengths = ITensor::slice(input_lengths, offset, batchSize + 1);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return buffers;
}

void MKBuffers::prepareContextStep(RuntimeBuffers* runtimeBuffers, TensorPtr const& inputIds, TokenIdType padId, BufferManager& manager, 
KvCacheManager const* KvCacheManager, SizeType32 firstBatchSlotIdx, 
ModelConfig const& modelConfig, WorldConfig const& worldConfig) {
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    seq_len_ = make_bs_param(manager, *(runtimeBuffers->contextLengthsHost));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void MKBuffers::postContextStep(RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBuffers, 
BufferManager& manager, ModelConfig const& modelConfig, WorldConfig const& worldConfig) {
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    kernels::gatherLastTokenLogits(*(runtimeBuffers->logits), *logits_ptr_in_use, *(runtimeBuffers->lastTokenIds), manager.getStream());
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void MKBuffers::prepareNextStep(RuntimeBuffers* runtimeBuffers, SizeType32 step, BufferManager& manager,
    KvCacheManager* kvCacheManager, SizeType32 firstBatchSlotIdx, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig) {
    TLLM_LOG_TRACE("%s start, step: %d", __PRETTY_FUNCTION__, step);
    const int vocabSize = modelConfig.getVocabSize();
    const int batchSize = generation_config_.batchSize;
    int new_token_num = 1;
    int* input_lengths_ptr = (int*)input_lengths->data();
    for (int i = 0; i < batchSize; ++i) {
        input_lengths_ptr[i] = 1;
    }
    if (seq_len_ == 0) {
        // trt encoder + mk decoder
        seq_len_ = BufferRange<SizeType32>(*(runtimeBuffers->contextLengthsHost))[0];
    }
    // add seq offset for next tokens
    input_lengths_ptr[batchSize] = seq_len_;

    if (batchSize > 1) {
        new_token_num = update_bs_param(manager, *(runtimeBuffers->contextLengthsHost), step);
    }

    seq_len_ += new_token_num;
    runtimeBuffers->logits->reshape(ITensor::makeShape({new_token_num, 1, vocabSize})); // TODO support beamsearch
    logits_ptr_in_use = ITensor::view(runtimeBuffers->logits, ITensor::makeShape({new_token_num, vocabSize}));
    TLLM_LOG_TRACE("%s stop, step: %d, batch: %d, seq_len: %d", __PRETTY_FUNCTION__, step, batchSize, seq_len_);
}

void MKBuffers::getRuntimeBuffers(RuntimeBuffers const* runtimeBuffers, TensorMap& inputBuffers, TensorMap& outputBuffers,
    SizeType32 step, TensorPtr const& inputIds, ModelConfig const& modelConfig, WorldConfig const& worldConfig) const {
    TLLM_LOG_TRACE("%s start, step: %d", __PRETTY_FUNCTION__, step);
    inputBuffers.clear();
    outputBuffers.clear();

    outputBuffers.insert_or_assign("logits", ITensor::view(logits_ptr_in_use));

    auto const localNbLayers = modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism());
    auto const firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;
    auto const& layerTypes = modelConfig.getLayerTypes();
    utils::insertTensorVector(inputBuffers, "past_key_value_", presentKeysVals, firstLayerId, layerTypes,
        ModelConfig::LayerType::kATTENTION);
    inputBuffers.insert_or_assign("input_ids", inputIds);
    inputBuffers.insert_or_assign("bs_params", bs_params);
    inputBuffers.insert_or_assign("input_lengths", input_lengths);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

int MKBuffers::make_bs_param(BufferManager& manager, ITensor &input_lengths_host) {
    const int batch_size = generation_config_.batchSize;
    int* bs_params_host = bufferCast<int>(*bs_host_params);
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
    bs_params->reshape(ITensor::makeShape({batch_size, 3}));
    manager.copy(bs_params_host, *bs_params, MemoryType::kGPU);
    return offset;
}

int MKBuffers::update_bs_param(BufferManager& manager, ITensor &input_lengths_host, SizeType32 new_token_num) {
    const int batch_size = generation_config_.batchSize;
    int* bs_params_host = bufferCast<int>(*bs_host_params);
    int idx = 0;
    auto input_lengths_buffer = BufferRange<SizeType32>(input_lengths_host);
    for (int i = 0; i < batch_size; ++i) {
        int len = input_lengths_buffer[i];
        bs_params_host[idx++] = i;
        bs_params_host[idx++] = 1;
        bs_params_host[idx++] = len + new_token_num;
    }
    bs_params->reshape(ITensor::makeShape({batch_size, 3}));
    manager.copy(bs_params_host, *bs_params, MemoryType::kGPU);
    return batch_size;
}