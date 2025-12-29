#pragma once

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/generationConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"

// TODO: use KVCacheManager
namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class KVCacheManager;
}

namespace tensorrt_llm::runtime
{
class RuntimeBuffers;

class MKBuffers
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using KvCacheManager = batch_manager::kv_cache_manager::KVCacheManager;
    using TensorMap = StringPtrMap<ITensor>;

    MKBuffers();

    MKBuffers(
        TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);
    
    void reshape(
        GenerationConfig const& generationConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void reshapeKvTensors(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxBlocksPerSeq, runtime::TllmRuntime const& runtime);

    void setKvPoolPointers(KvCacheManager const* kvCacheManager);

    void reset(BufferManager& manager);

    MKBuffers sliceTo(GenerationConfig const& generationConfig, ModelConfig const& modelConfig, SizeType32 offset, SizeType32 batchSize);

    void prepareContextStep(RuntimeBuffers* runtimeBuffers, TensorPtr const& inputIds, TokenIdType padId, BufferManager& manager, 
    KvCacheManager const* KvCacheManager, SizeType32 firstBatchSlotIdx, 
    ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void postContextStep(RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBuffers, 
    BufferManager& manager, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void prepareNextStep(RuntimeBuffers* runtimeBuffers, SizeType32 step, BufferManager& manager,
        KvCacheManager* kvCacheManager, SizeType32 firstBatchSlotIdx, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig);

    void getRuntimeBuffers(RuntimeBuffers const* runtimeBuffers, TensorMap& inputBuffers, TensorMap& outputBuffers,
        SizeType32 step, TensorPtr const& inputIds, ModelConfig const& modelConfig, WorldConfig const& worldConfig) const;

protected:
    int make_bs_param(BufferManager& manager, ITensor &input_lengths_host);
    int update_bs_param(BufferManager& manager, ITensor &input_lengths_host, SizeType32 new_token_num);
public:
    // engine
    // TODO: use buffer from kvCacheManager 
    TensorPtr k_cache;
    TensorPtr v_cache;

    TensorPtr logits; 
    TensorPtr bs_params;
    TensorPtr input_lengths;

    SizeType32 seq_len_{0};
    GenerationConfig generation_config_;
};
} // namespace tensorrt_llm::runtime