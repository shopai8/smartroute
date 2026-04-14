// 文件名: MethodSelector.h

#pragma once 

#include <string>
#include <vector>
#include <memory> 

#include "../third_party/onnxruntime-linux-x64-1.16.3/include/onnxruntime_cxx_api.h"

class MethodSelector {
public:
    MethodSelector(const std::string& model_path);

    // 返回值从 bool 改为 float，以支持多分类和回归模型
    float predict(const std::vector<float>& features);

private:
    // 移除 Ort::Env _env; 改为使用静态方法获取全局单例
    Ort::Session _session{nullptr};
    Ort::AllocatorWithDefaultOptions _allocator;
    
    std::vector<const char*> _input_node_names;
    std::vector<const char*> _output_node_names;
    std::vector<int64_t> _input_node_dims;

    std::string _input_name_str;
    std::string _output_name_str;

    // 获取全局共享的 ONNX Runtime 环境
    static Ort::Env& get_shared_env();
};