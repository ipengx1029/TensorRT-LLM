#include <iostream>
#include "tensorrt_llm/api/tensorrt_runtime_api.h"

using namespace tensorrt_llm::api;

template<typename T>
class TensorImpl : public LlmTensor {
public:
    void* data() override {
        return buffer.data();
    }

    void reshape(const nvinfer1::Dims &s) override {
        std::vector<long> _tmp_shape;
        size_t tmp_volume = 1;
        for (int i = 0; i < s.nbDims; ++i) {
            _tmp_shape.push_back(s.d[i]);
            tmp_volume *= s.d[i];
        }
        if (_volume < tmp_volume) {
            buffer.resize(tmp_volume);
        }
        _volume = tmp_volume;
        shape = _tmp_shape;
    }

    void reshape(const std::vector<long> &s) {
        size_t tmp_volume = 1;
        for (auto &v : s) {
            tmp_volume *= v;
        }
        if (_volume < tmp_volume) {
            buffer.resize(tmp_volume);
        }
        shape = s;
        _volume = tmp_volume;
    }

public:
    std::vector<T> buffer;
    size_t _volume{0};
};

bool run_test_gptsession(const std::shared_ptr<TRTExecutor> &trt_executor) {
    auto config = trt_executor->get_config();
    std::vector<std::shared_ptr<LlmTensor>> input_tensors;
    input_tensors.resize(2);
      
    uint32_t batch_size = 2;
    uint32_t max_input_length = 4;
    // set input tensors

    auto input_ids = std::make_shared<TensorImpl<int32_t>>();
    if (config.input_packed) {
        input_ids->reshape(std::vector<long>({8}));
    } else {
        input_ids->reshape(std::vector<long>({batch_size, max_input_length}));
    }
    input_ids->name = "input_ids";
    input_ids->buffer = {99526,  46944,  45861, 101108,
                         99526,  46944,  45861, 101108};
    input_ids->dtype = nvinfer1::DataType::kINT32;           
    input_tensors[0] = input_ids;

    auto input_lengths = std::make_shared<TensorImpl<int32_t>>();
    input_lengths->reshape(std::vector<long>({batch_size}));
    input_lengths->name = "input_lengths";
    input_lengths->buffer = {4, 4};
    input_lengths->dtype = nvinfer1::DataType::kINT32;
    input_tensors[1] = input_lengths;

    // set output tensors
    std::vector<std::shared_ptr<LlmTensor>> output_tensors;
    
    {
        auto tmp_tensor = std::make_shared<TensorImpl<int32_t>>();
        tmp_tensor->reshape(std::vector<long>({batch_size, config.max_beam_size, config.max_input_len + config.max_dec_len}));
        tmp_tensor->name = "ids";
        tmp_tensor->dtype = nvinfer1::DataType::kINT32;
        output_tensors.emplace_back(tmp_tensor);
    }

    // nlu scores
    if (config.nlu_scores_size > 0) {
        auto tmp_tensor = std::make_shared<TensorImpl<float>>();
        tmp_tensor->reshape(std::vector<long>({batch_size, config.max_beam_size, config.nlu_scores_size}));
        tmp_tensor->name = "nlu_scores";
        tmp_tensor->dtype = nvinfer1::DataType::kFLOAT;
        output_tensors.emplace_back(tmp_tensor);
    }
    // decoder scores
    if (config.cum_log_probs > 0) {
        auto tmp_tensor = std::make_shared<TensorImpl<float>>();
        tmp_tensor->reshape(std::vector<long>({batch_size, config.max_beam_size}));
        tmp_tensor->name = "cum_log_probs";
        tmp_tensor->dtype = nvinfer1::DataType::kFLOAT;
        output_tensors.emplace_back(tmp_tensor);
    }
    // context feature
    if (config.context_fea_size > 0) {
        auto tmp_tensor = std::make_shared<TensorImpl<float>>();
        tmp_tensor->reshape(std::vector<long>({batch_size, config.context_fea_size}));
        tmp_tensor->name = "context_features";
        tmp_tensor->dtype = nvinfer1::DataType::kFLOAT;
        output_tensors.emplace_back(tmp_tensor);
    }

    //run generate
    int ret = trt_executor->run(input_tensors, &output_tensors);
    if (ret != 0) {
        std::cout << "TrtLlmGptSessionCore run failed, ret: " << ret << std::endl;
        return false;
    }
    std::cerr << "ids: " << std::endl;
    for (int i = 0; i < 30; ++i) {
        std::cerr << std::dynamic_pointer_cast<TensorImpl<int32_t>>(output_tensors[0])->buffer[i] << ", ";
    }
    std::cerr << std::endl;
    std::cout << "Execute GptSession sussess, out_ids: " << std::dynamic_pointer_cast<TensorImpl<int32_t>>(output_tensors[0])->buffer[max_input_length] << std::endl;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: ./gptApiTest <engine_path> [<GPT_SESSION case_config_file>]";
        return -1;
    }
    auto engine_type = static_cast<tensorrt_llm::api::EngineType>(2);
    tensorrt_llm::api::Config config;
    config.engine_type = engine_type;
    config.engine_path = argv[1];
    config.log_level = "debug";
    config.config_file = (engine_type == tensorrt_llm::api::EngineType::GPT_SESSION) ? argv[2] : ""; 

    auto trt_executor = tensorrt_llm::api::create_trt_executor(config);
    int ret = trt_executor->init(config);
    if (ret != 0) {
        std::cout << "failed to init trt-llm predictor, ret: " << ret << std::endl;
        return -1;
    }
    // test api runner
    if (engine_type == tensorrt_llm::api::EngineType::GPT_SESSION) {
        run_test_gptsession(trt_executor);
    }
    return 0;
}