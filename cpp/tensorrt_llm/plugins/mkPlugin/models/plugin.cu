#include "base.h"
#include "qwen.h"
namespace mk {
std::shared_ptr<ModelInfer> get_model_infer(
    const int model_type, const int quant_type, const int sms_count) {
    std::shared_ptr<ModelInfer> infer = nullptr;
    if (model_type == 1) {
        if (quant_type == 0) {
            switch(sms_count) {
#ifndef KITTENS_HOPPER
                case 56: // A30
                    infer = std::make_shared<QwenModelInfer<56, 164000>>();
                    break;
                case 72: // A10
                    infer = std::make_shared<QwenModelInfer<72, 100000>>();
                    break;
                case 92: // L20
                    infer = std::make_shared<QwenModelInfer<92, 100000>>();
                    break;
                case 108: // A100
                    infer = std::make_shared<QwenModelInfer<108, 164000>>();
                    break;
                case 128: // 4090
                    infer = std::make_shared<QwenModelInfer<128, 100000>>();
                    break;
#else
                case 132: // H100
                    infer = std::make_shared<QwenModelInfer<132, 227000>>();
                    break;
#endif
                default:
                    return nullptr;
            }
        } else if (quant_type == 1) {
#ifndef KITTENS_HOPPER
            switch(sms_count) {
                case 56: // A30
                    infer = std::make_shared<GPTQQwenModelInfer<56, 164000>>();
                    break;
                case 72: // A10
                    infer = std::make_shared<GPTQQwenModelInfer<72, 100000>>();
                    break;
                case 92: // L20
                    infer = std::make_shared<GPTQQwenModelInfer<92, 100000>>();
                    break;
                case 108: // A100
                    infer = std::make_shared<GPTQQwenModelInfer<108, 164000>>();
                    break;
                case 128: // 4090
                    infer = std::make_shared<GPTQQwenModelInfer<128, 100000>>();
                    break;
                default:
                    return nullptr;
            }
#endif
        }
    } 
    return infer;
}
}