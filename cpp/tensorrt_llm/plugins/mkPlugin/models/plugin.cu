#include "base.h"
#include "qwen.h"
namespace mk {
std::shared_ptr<ModelInfer> get_model_infer(const int model_type, const int sms_count) {
    std::shared_ptr<ModelInfer> infer = nullptr;
    if (model_type == 1) {
        switch(sms_count) {
            case 56: // A30
                infer = std::make_shared<QwenModelInfer<56, 164000>>();
                break;
            case 72: // A10
                infer = std::make_shared<QwenModelInfer<72, 100000>>();
                break;
            case 92: // L20
                infer = std::make_shared<QwenModelInfer<92, 100000>>();
                break;
            default:
                return nullptr;
        }
    } 
    return infer;
}
}