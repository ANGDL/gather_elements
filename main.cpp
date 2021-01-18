#include "torch/torch.h"
#include "gather_elements.cuh"
#include "NvInfer.h"

#include <string>
#include <vector>
#include <iostream>
#include <dlfcn.h>

using namespace std;

class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char *msg) override {
        // suppress info-level messages
        if (severity == Severity::kINFO) return;

        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "UNKNOWN: ";
                break;
        }
        std::cerr << msg << std::endl;
    }
};

nvinfer1::ICudaEngine *
load_trt_engine(const std::string plan_file, nvinfer1::ILogger &logger) {
    // reading the model in memory
    std::cout << "Loading TRT Engine..." << std::endl;
    std::stringstream trt_model_stream;
    trt_model_stream.seekg(0, std::stringstream::beg);
    std::ifstream cache(plan_file);
    assert(cache.good());
    trt_model_stream << cache.rdbuf();
    cache.close();

    // calculating model size
    trt_model_stream.seekg(0, std::ios::end);
    const int model_size = trt_model_stream.tellg();
    trt_model_stream.seekg(0, std::ios::beg);
    void *model_mem = malloc(model_size);
    trt_model_stream.read((char *) model_mem, model_size);

    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine *engine
            = runtime->deserializeCudaEngine(model_mem, model_size);
    free(model_mem);
    runtime->destroy();
    std::cout << "Loading Complete!" << std::endl;

    return engine;
}

inline void loadLibrary(const std::string& path)
{
#ifdef _MSC_VER
    void* handle = LoadLibrary(path.c_str());
#else
    void* handle = dlopen(path.c_str(), RTLD_LAZY);
#endif
    if (handle == nullptr)
    {
#ifdef _MSC_VER
        gLogError << "Could not load plugin library: " << path << std::endl;
#else
        std::cout << "Could not load plugin library: " << path << ", due to: " << dlerror() << std::endl;
        exit(-1);
#endif
    }
}

int main() {
    std::vector<std::string> ld_libs = {
            "/home/ang/TensorRT-7.0.0.11/lib/libnvinfer.so",
            "/home/ang/TensorRT-7.0.0.11/lib/libnvinfer_plugin.so",
            "/home/ang/TensorRT-7.0.0.11/lib/libnvparsers.so",
            "/home/ang/onnx-tensorrt/build/libnvonnxparser.so"
    };

    for(auto& c : ld_libs){
        loadLibrary(c);
    }

    cudaSetDevice(0);

    torch::Tensor b = torch::rand({2, 10, 10}, {torch::kFloat32}).cuda();
    torch::Tensor index = torch::tensor({0, 1, 1, 0, 2, 1, 3, 2}).reshape({ 1, 2, -1}).to(torch::kInt64).cuda();
//    torch::Tensor index = torch::randint_like(b, 0, 1).to(torch::kInt64).cuda();

    torch::Tensor index32 = torch::tensor({0, 1, 1, 0, 2, 1, 3, 2}, {torch::kInt32}).reshape({1, 2, -1}).cuda();

    torch::Tensor output = torch::zeros_like(index).to(torch::kFloat32).cuda();
    torch::Tensor trt_output = torch::zeros_like(index32).to(torch::kFloat32).cuda();

    auto inputs = std::vector<void*>({b.data_ptr(), index32.data_ptr()});
    auto outputs = std::vector<void*>({output.data_ptr()});

    auto trt_bindings = std::vector<void*>({b.data_ptr(), index32.data_ptr(), trt_output.data_ptr()});

    auto logger = Logger();
    std::string plane_file = "/home/ang/onnx-tensorrt/build/gather.trt";
    auto engine = load_trt_engine(plane_file, logger);
    auto trt_context = engine->createExecutionContext();

    assert(engine->getNbBindings() == 3);

    unsigned int* workspace = nullptr;
    size_t workspace_size = sizeof(unsigned int) * (index.dim() * 4 + index.dim() * index.element_size());
    gpuErrchk(cudaMalloc(&workspace, workspace_size));

    std::vector<unsigned int> tensor_dims{b.sizes().begin(), b.sizes().end()};
    std::vector<unsigned int> index_dims{index32.sizes().begin(), index32.sizes().end()};

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for(int i = 0; i != 1; ++i){
        auto res = torch::gather(b, 1, index);

        gather_elements(
                inputs.data(), outputs.data(), 1, index32.dim(),
                tensor_dims.data(), index_dims.data(),
                workspace, stream);

        trt_context->executeV2(trt_bindings.data());

        std::cout << (res - output).sum() << "\n " << (res - trt_output).sum() << std::endl;

        std::cout << "b ========" << std::endl;
        std::cout << b << std::endl;

        std::cout << "res ========" << std::endl;
        std::cout << res << std::endl;
        std::cout << "cuda output ========" << std::endl;
        std::cout << output << std::endl;
        std::cout << "trt output ========" << std::endl;
        std::cout << trt_output << std::endl;

    }

    cudaFree(workspace);
    cudaStreamDestroy(stream);

    return 0;
}
