/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/runtime/runtimeBuffers.h"

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

#include <algorithm>
namespace tensorrt_llm::common {
    extern bool getEnvGptSessionEnableDebugPrint();
}
using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

void RuntimeBuffers::clear()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    contextLengthsHost = nullptr;
    contextLengthsDevice = nullptr;

    logits = nullptr;
    sequenceLengths = nullptr;
    lastTokenIds = nullptr;
    requestTypes = nullptr;

    cacheIndirectionDecoderInput = nullptr;
    cacheIndirectionDecoderOutput = nullptr;

    cumLogProbs = nullptr;
    logProbs = nullptr;

    nluScores = nullptr;

    contextFeatures = nullptr;

    hiddenStates = nullptr;

    allocated = false;
    isMedusa = false;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::clearTensorMaps()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    for (auto& buffer : inputBuffers)
        buffer.clear();
    for (auto& buffer : outputBuffers)
        buffer.clear();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::addEngine(TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig) {
    bool megaKernelModel = modelConfig.isMegaKernelModel();
    if (megaKernelModel) {
        mkBuffers.emplace(runtime, modelConfig, worldConfig, transformerBuffers->presentKeysVals);
        mkBuffers.tempClear();
    }
}

void RuntimeBuffers::switchBuffers() {
    if (mkBuffers.hasCurrent()) {
        mkBuffers.tempClear();
        transformerBuffers.restore();
        TLLM_LOG_TRACE("switchBuffers: from mkBuffers to transformerBuffers");
    } else if (transformerBuffers.hasCurrent()) {
        transformerBuffers.tempClear();
        mkBuffers.restore();
        TLLM_LOG_TRACE("switchBuffers: from transformerBuffers to mkBuffers");
    } else {
        TLLM_THROW("No buffers to switch");
    }
}

void RuntimeBuffers::create(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, 
                            TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
                            std::optional<runtime::MedusaModule::MedusaChoices> const& medusaChoices)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& manager = runtime.getBufferManager();
    auto const& engine = runtime.getEngine();

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto const logitsType = engine.getTensorDataType("logits");
        logits = manager.emptyTensor(MemoryType::kGPU, logitsType);
        originalLogitsPtr = logits;

        allGenerationLogits = manager.emptyTensor(MemoryType::kGPU, logitsType);
        if (modelConfig.computeGenerationLogits())
        {
            cacheGenerationFragmentPointerDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT64);
            cacheGenerationFragmentPointerHost = manager.emptyTensor(MemoryType::kPINNED, nvinfer1::DataType::kINT64);

            generationLogitsFragments = std::make_shared<std::vector<TensorPtr>>();
        }

        if (modelConfig.ContextFeaturesSize()) {
            contextFeatures = manager.emptyTensor(MemoryType::kGPU, logitsType);
        }
    }

    lastTokenIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    bool transformerBased = modelConfig.isTransformerBased();
    bool rnnBased = modelConfig.isRnnBased();
    bool megaKernelModel = modelConfig.isMegaKernelModel();

    contextLengthsHost = manager.emptyTensor(MemoryType::kPINNED, nvinfer1::DataType::kINT32);
    if (transformerBased && !megaKernelModel)
    {
        if (modelConfig.useGptAttentionPlugin())
        {
            requestTypes = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
        }
        transformerBuffers.emplace(runtime, modelConfig, worldConfig);
    }
    if (rnnBased && !megaKernelModel)
    {
        requestTypes = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
        rnnStateBuffers.emplace(runtime, modelConfig, worldConfig);
    }
    if (megaKernelModel) {
        mkBuffers.emplace(runtime, modelConfig, worldConfig);
    }

    cacheIndirectionDecoderInput = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    cacheIndirectionDecoderOutput = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    nbFinished = BufferManager::pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

    if (worldConfig.isPipelineParallel())
    {
        hiddenStates = manager.emptyTensor(MemoryType::kGPU, modelConfig.getDataType());
    }
    isMedusa = modelConfig.useMedusa();
    if (isMedusa) {
        medusaBuffers.emplace();
        medusaBuffers->create(maxBatchSize, maxBeamWidth, medusaChoices, manager, modelConfig, worldConfig, runtime);
        medusaInputTokens = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        acceptedTokensLengthHost = manager.emptyTensor(MemoryType::kPINNED, nvinfer1::DataType::kINT32);
        medusaSequenceLengths = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        medusaContextLengths = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::initFromInput(ITensor const& inputIds, TensorPtr const& inputLengths, bool inputPacked,
    SizeType32 beamWidth, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
    BufferManager& manager)
{
    if (isMedusa) {
        medusaContextLengths->reshape(inputLengths->getShape());
        manager.copy(*inputLengths, *medusaContextLengths);
        contextLengthsDevice = medusaContextLengths;
    } else {
        contextLengthsDevice = inputLengths;
    }
    contextLengthsHost->reshape(inputLengths->getShape());
    manager.copy(*contextLengthsDevice, *contextLengthsHost);
    manager.getStream().synchronize(); // wait for context lengths to be copied to host

    generationConfig = GenerationConfig::fromInput(
        inputIds, *contextLengthsHost, inputPacked, beamWidth, maxAttentionWindow, sinkTokenLength, maxSequenceLength);
}

void RuntimeBuffers::reshape(ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    auto const maxInputLength = generationConfig.maxInputLength;
    auto const maxAttentionWindow = generationConfig.maxAttentionWindow;
    auto const maxSeqLength = generationConfig.maxSeqLength;
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    if (worldConfig.isLastPipelineParallelRank())
    {
        if (modelConfig.computeContextLogits())
        {
            if (!modelConfig.computeGenerationLogits())
            {
                // If only enable computeContextLogits, also need to have a generation buffer to store the last token of
                // context
                allGenerationLogits->reshape(ITensor::makeShape({1, batchSize, beamWidth, vocabSizePadded}));
            }
        }
        else
        {
            // If only gather generation logits
            if (modelConfig.computeGenerationLogits())
            {
                logits = originalLogitsPtr; // logits point to original buffer
            }
            logits->reshape(ITensor::makeShape({batchSize, 1, vocabSizePadded}));
        }

        if (modelConfig.computeGenerationLogits())
        {
            allGenerationLogits->reshape(
                ITensor::makeShape({(maxSeqLength - maxInputLength), batchSize, beamWidth, vocabSizePadded}));

            cacheGenerationFragmentPointerDevice->reshape(
                ITensor::makeShape({batchSize, (maxSeqLength - maxInputLength)}));
            cacheGenerationFragmentPointerHost->reshape(
                ITensor::makeShape({batchSize, (maxSeqLength - maxInputLength)}));
        }

        if (modelConfig.ContextFeaturesSize()) {
            contextFeatures->reshape(ITensor::makeShape({batchSize, beamWidth, modelConfig.ContextFeaturesSize()}));
        }
    }

    lastTokenIds->reshape(ITensor::makeShape({batchSize}));

    if (transformerBuffers)
    {
        if (modelConfig.useGptAttentionPlugin())
        {
            requestTypes->reshape(ITensor::makeShape({batchSize}));
        }
        transformerBuffers->reshape(generationConfig, modelConfig, worldConfig);
    }

    if (rnnStateBuffers)
    {
        requestTypes->reshape(ITensor::makeShape({batchSize}));
        rnnStateBuffers->reshape(generationConfig, modelConfig, worldConfig);
    }

    if (mkBuffers) 
    {
        mkBuffers->reshape(generationConfig, modelConfig, worldConfig);
    }

    auto const cacheIndirShape = ITensor::makeShape({batchSize, beamWidth, maxAttentionWindow});
    cacheIndirectionDecoderInput->reshape(cacheIndirShape);
    cacheIndirectionDecoderOutput->reshape(cacheIndirShape);

    if (worldConfig.isPipelineParallel())
    {
        // reserve max size
        auto const maxNumTokens = std::max(beamWidth, maxInputLength);
        auto const hiddenSize = modelConfig.getHiddenSize() * worldConfig.getTensorParallelism();
        auto const hiddenStatesShape = ITensor::makeShape(
            {batchSize, maxNumTokens, hiddenSize}); // reserve space in traditional [bs, seq_len, hidden_state] way.
        hiddenStates->reshape(hiddenStatesShape);
    }

    if (isMedusa) {
        auto const maxDecoderLen = modelConfig.getMaxTokensPerStep();
        // medusaBuffers->reshape(0, 0, maxDecoderLen);
        medusaInputTokens->reshape(ITensor::makeShape({batchSize, maxDecoderLen}));
        acceptedTokensLengthHost->reshape(ITensor::makeShape({batchSize}));
        medusaContextLengths->reshape(ITensor::makeShape({batchSize}));
        medusaSequenceLengths->reshape(ITensor::makeShape({batchSize}));
        if (transformerBuffers && !modelConfig.usePagedKvCache()) {
            auto &presentKeysVals = transformerBuffers->presentKeysVals;
            size_t numLayers = presentKeysVals.size();
            pastKeyValuePtrList.resize(numLayers);
            for (size_t i = 0; i < numLayers; ++i) {
                pastKeyValuePtrList[i] = static_cast<int8_t*>(presentKeysVals[i]->data());
            }
        }
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
        auto const medusaHeads = modelConfig.getMedusaModule()->medusaHeads();
        medusaBuffers->medusaLogitsDevice->reshape(
            ITensor::makeShape({medusaHeads, batchSize, maxDecoderLen, vocabSizePadded}));
    }
    allocated = true;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::reset(BufferManager& manager)
{
    clearTensorMaps();
    manager.setZero(*cacheIndirectionDecoderInput);
    manager.setZero(*cacheIndirectionDecoderOutput);

    if (transformerBuffers)
    {
        transformerBuffers->reset(manager);
    }

    if (rnnStateBuffers)
    {
        rnnStateBuffers->reset(manager);
    }

    if (mkBuffers) 
    {
        mkBuffers->reset(manager);
    }
}

std::vector<RuntimeBuffers> RuntimeBuffers::split(
    SizeType32 contextBatchSize, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    std::vector<RuntimeBuffers> bufferSlices;
    auto const generationBatchSize = generationConfig.batchSize;
    bufferSlices.reserve(tc::ceilDiv(generationBatchSize, contextBatchSize));
    if (contextBatchSize >= generationBatchSize)
    {
        bufferSlices.emplace_back(*this);
    }
    else
    {
        for (auto offset = 0; offset < generationBatchSize; offset += contextBatchSize)
        {
            auto const batchSize = std::min(contextBatchSize, generationBatchSize - offset);
            auto& buffers = bufferSlices.emplace_back();
            buffers.generationConfig = generationConfig;
            buffers.generationConfig.batchSize = batchSize;

            buffers.contextLengthsHost = ITensor::slice(contextLengthsHost, offset, batchSize);
            buffers.contextLengthsDevice = ITensor::slice(contextLengthsDevice, offset, batchSize);

            if (worldConfig.isLastPipelineParallelRank() && !modelConfig.computeContextLogits())
            {
                buffers.logits = ITensor::slice(logits, offset, batchSize);
            }

            buffers.lastTokenIds = ITensor::slice(lastTokenIds, offset, batchSize);

            if (transformerBuffers)
            {
                buffers.transformerBuffers
                    = transformerBuffers->sliceTo(generationConfig, modelConfig, offset, batchSize);
            }

            if (rnnStateBuffers)
            {
                buffers.rnnStateBuffers = rnnStateBuffers->sliceTo(offset, batchSize);
            }

            if (mkBuffers) {
                buffers.mkBuffers = mkBuffers->sliceTo(generationConfig, modelConfig, offset, batchSize);
            }

            if (requestTypes != nullptr)
            {
                buffers.requestTypes = ITensor::slice(requestTypes, offset, batchSize);
            }
            if (worldConfig.isPipelineParallel())
            {
                TLLM_CHECK_WITH_INFO(hiddenStates->getShape().nbDims == 3,
                    "Invalid shape for hiddenStates."); // Expect hiddens states shape to be [bs, seq_len, hidden_size]
                // at generation buffer split stage.
                buffers.hiddenStates = ITensor::slice(hiddenStates, offset, batchSize);
            }

            buffers.cacheIndirectionDecoderOutput = ITensor::slice(cacheIndirectionDecoderOutput, offset, batchSize);

            if (modelConfig.usePromptTuning())
            {
                auto const& ptuningEnabled = promptTuningParams.promptTuningEnabled;
                buffers.promptTuningParams.promptTuningEnabled
                    = std::vector<bool>(ptuningEnabled.begin() + offset, ptuningEnabled.begin() + offset + batchSize);

                buffers.promptTuningParams.tasks = ITensor::slice(promptTuningParams.tasks, offset, batchSize);
            }
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return bufferSlices;
}

void RuntimeBuffers::gatherLastTokenLogits(
    BufferManager& manager, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(modelConfig.computeContextLogits(),
        "Gather last token logits is only necessary when context logits are computed");

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
        TensorPtr tiledTensor = ITensor::slice(allGenerationLogits, 0, 1);
        tiledTensor->squeeze(0);
        kernels::gatherLastTokenLogits(*tiledTensor, *logits, *lastTokenIds, manager.getStream());
        manager.getStream().synchronize();

        std::swap(logits, tiledTensor);
        if (modelConfig.usePackedInput())
        {
            tiledTensor->reshape(
                ITensor::makeShape({generationConfig.inputLengthSum, vocabSizePadded})); // [packedSize, vocabSize]
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::postContextStep(std::vector<RuntimeBuffers> const& contextBuffers, BufferManager& manager,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    auto const maxDecoderLen = modelConfig.getMaxTokensPerStep();
    if (transformerBuffers.hasCurrent())
    {
        transformerBuffers->postContextStep(this, contextBuffers, manager, modelConfig, worldConfig);
    }
    if (rnnStateBuffers)
    {
        rnnStateBuffers->postContextStep(this, contextBuffers, manager, modelConfig, worldConfig);
    }

    if (mkBuffers.hasCurrent())
    {
        mkBuffers->postContextStep(this, contextBuffers, manager, modelConfig, worldConfig);
    }

    // use output lengths after context step
    manager.copy(*contextLengthsDevice, *outputLengths);
    sequenceLengths = ITensor::view(outputLengths);
    sequenceLengths->reshape(ITensor::makeShape({batchSize * beamWidth}));
    // no need to copy data in lastTokenIds because it is overwritten in prepareNextStep
    lastTokenIds->reshape(ITensor::makeShape({batchSize * beamWidth * maxDecoderLen}));

    if (modelConfig.usePromptTuning())
    {
        std::vector<SizeType32> reqBeamWidths(batchSize, beamWidth);
        //// Note: reqPromptLenghts won't be used
        std::vector<SizeType32> reqPromptLengths;
        // Copy the generationInput tasks to host
        promptTuningTasksHost = manager.copyFrom(*promptTuningParams.tasks, MemoryType::kPINNED);
        // Update the promptTuningParams tasks tensor
        promptTuningParams.fillTasksTensor(promptTuningTasksHost, batchSize, 0, reqBeamWidths, reqPromptLengths,
            manager, modelConfig.usePackedInput());
    }
    // medusa decoder step
    if (isMedusa) {
        // medusaBuffers->reshape(0, batchSize, maxDecoderLen);
        int32_t* contextHostPtr = bufferCast<int32_t>(*contextLengthsHost);
        for (SizeType32 i = 0; i < batchSize * beamWidth; ++i) {
            contextHostPtr[i] += (maxDecoderLen - 1);
        }
        manager.copy(*contextLengthsHost, *contextLengthsDevice);
        manager.copy(*contextLengthsDevice, *medusaSequenceLengths);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::prepareContextStep(TensorPtr const& inputIds, TokenIdType const padId, BufferManager& manager,
    batch_manager::kv_cache_manager::KVCacheManager const* kvCacheManager, SizeType32 firstBatchSlotIdx,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& stream = manager.getStream();

    // use context lengths only in context step
    sequenceLengths = contextLengthsDevice;

    if (transformerBuffers)
    {
        transformerBuffers->prepareContextStep(
            this, inputIds, padId, manager, kvCacheManager, firstBatchSlotIdx, modelConfig, worldConfig);
    }

    if (rnnStateBuffers)
    {
        rnnStateBuffers->prepareContextStep(this, manager);
    }

    if (mkBuffers) 
    {
        mkBuffers->prepareContextStep(
            this, inputIds, padId, manager, kvCacheManager, firstBatchSlotIdx, modelConfig, worldConfig);
    }

    if (modelConfig.usePackedInput())
    {
        kernels::invokeInclusiveSum(*lastTokenIds, *contextLengthsDevice, manager, stream);
    }
    else
    {
        manager.copy(*contextLengthsDevice, *lastTokenIds);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

RuntimeBuffers::TensorPtr RuntimeBuffers::prepareNextStep(SizeType32 const step, BufferManager& manager,
    batch_manager::kv_cache_manager::KVCacheManager* kvCacheManager, SizeType32 firstBatchSlotIdx,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig, bool nlu_exec)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& stream = manager.getStream();
    SizeType32 const batchSize = generationConfig.batchSize;
    SizeType32 const beamWidth = generationConfig.beamWidth;
    auto const maxDecoderLen = modelConfig.getMaxTokensPerStep();
    auto const inputShape = [&modelConfig, batchSize, beamWidth, maxDecoderLen]()
    {
        if (modelConfig.usePackedInput())
        {
            // batch in last dim
            return ITensor::makeShape({batchSize * beamWidth * maxDecoderLen});
        }
        else
        {
            // batch in first dim
            return ITensor::makeShape({batchSize * beamWidth, maxDecoderLen});
        }
    }();
    if (transformerBuffers.hasCurrent()) {
        transformerBuffers->prepareNextStep(
            this, step, manager, kvCacheManager, firstBatchSlotIdx, modelConfig, worldConfig);
    }

    if (mkBuffers.hasCurrent()) {
        mkBuffers->prepareNextStep(
            this, step, manager, kvCacheManager, firstBatchSlotIdx, modelConfig, worldConfig);
    }
    kernels::invokeFill(*lastTokenIds, 1, stream);
    if (modelConfig.usePackedInput()) {
        kernels::invokeInclusiveSum(*lastTokenIds, *lastTokenIds, manager, stream);
    }
    // reshape medusa logits
    if (isMedusa) {
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
        auto const medusaHeads = modelConfig.getMedusaModule()->medusaHeads();
        medusaBuffers->medusaLogitsDevice->reshape(
            ITensor::makeShape({medusaHeads, batchSize, maxDecoderLen, vocabSizePadded}));
        logits->reshape(ITensor::makeShape({batchSize, maxDecoderLen, vocabSizePadded}));
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);

        return ITensor::view(medusaInputTokens, inputShape);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    auto output_ptr = newTokens ? ITensor::view(newTokens, inputShape) : TensorPtr{};
    if (nlu_exec) {
        TLLM_LOG_TRACE("force input to be %d", endId);
        kernels::invokeFill(*output_ptr, endId, stream);
        if (transformerBuffers) {
            kernels::invokeUpdateNluPositionIdsGenerationPhaseGlm(
                *transformerBuffers->positionIds, 
                *contextLengthsDevice, 
                *sequenceLengths, 
                batchSize, 
                beamWidth, 
                modelConfig.usePackedInput(), 
                stream);
        }
    }
    return output_ptr;
}

void RuntimeBuffers::getRuntimeBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, SizeType32 const step,
    TensorPtr const& inputIds, TensorPtr const& commPtrs, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    inputBuffers.clear();
    outputBuffers.clear();

    if (transformerBuffers.hasCurrent())
    {
        transformerBuffers->getRuntimeBuffers(
            this, inputBuffers, outputBuffers, step, inputIds, modelConfig, worldConfig);
    }
    if (mkBuffers.hasCurrent())
    {
        mkBuffers->getRuntimeBuffers(
            this, inputBuffers, outputBuffers, step, inputIds, modelConfig, worldConfig);
    }

    if (rnnStateBuffers)
    {
        rnnStateBuffers->getRuntimeBuffers(this, inputBuffers, outputBuffers, step, inputIds, modelConfig, worldConfig);
    }

    if (modelConfig.useCustomAllReduce() && worldConfig.isTensorParallel())
    {
        inputBuffers.insert_or_assign("all_reduce_workspace", commPtrs);
    }

    if (modelConfig.usePromptTuning())
    {
        inputBuffers.insert_or_assign("prompt_embedding_table", promptTuningParams.embeddingTable);
        inputBuffers.insert_or_assign("tasks", promptTuningParams.tasks);
        inputBuffers.insert_or_assign("prompt_vocab_size", promptTuningParams.vocabSize);
    }
    // medusa buffers
    if (medusaBuffers)
    {
        medusaBuffers->insertInputTensors(inputBuffers, outputBuffers, worldConfig, generationConfig.batchSize);
    }
    if (tc::getEnvGptSessionEnableDebugPrint()) {
        std::cout << "step=" << step << "====================================================" << std::endl;
        if (step == 0 && transformerBuffers) {
            std::cout << "inputIds=" << *inputIds << std::endl;
            std::cout << "positionIds=" << *transformerBuffers->positionIds << std::endl;
        }
    }
    // utils::printTensorMap(std::cerr, inputBuffers);
    // utils::printTensorMap(std::cerr, outputBuffers);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

// update medusa new tokens
void RuntimeBuffers::updateMedusaNewTokens(BufferManager& manager, TensorPtr const &accTokensLen, 
    TensorPtr const &newTokens, TensorPtr const &nextDraftTokens) {
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& stream = manager.getStream();
    manager.copy(*accTokensLen, *acceptedTokensLengthHost);
    // update position ids
    transformerBuffers->updatePositionIds(manager, accTokensLen);
    // nextDraftTokens=shape: (1, 18), newAllTokens=shape: (19, 1), medusaInputTokens=shape: (1, 19)
    kernels::invokeUpdateMedusaTokenIds(*medusaInputTokens, *newTokens, *accTokensLen, *nextDraftTokens, stream);
    // update medusa sequence lengths
    kernels::invokeTensorAdd(*medusaSequenceLengths, *medusaSequenceLengths, *accTokensLen, stream);
    stream.synchronize();
    // update total accepted tokens length
    transformerBuffers->updateAcceptedTokensLength(this, acceptedTokensLengthHost);
    // debug print
    if (tc::getEnvGptSessionEnableDebugPrint()) {
        std::cout << "accTokensLen=" << *accTokensLen << std::endl;
        std::cout << "nextDraftTokens=" << *nextDraftTokens << std::endl;
        std::cout << "medusaInputIds=" << *medusaInputTokens << std::endl;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
void RuntimeBuffers::printAcceptedTokensLength(void) {
    if (!isMedusa) {
        return;
    }
    transformerBuffers->printAcceptedTokensLength();
}
