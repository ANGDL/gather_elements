#include <iostream>
#include "torch/torch.h"
#include "gather_elements.cuh"
#include <vector>

using namespace std;

int main() {
    cudaSetDevice(0);

    torch::Tensor b = torch::rand({1, 2, 2}).cuda();
    torch::Tensor index = torch::tensor({0, 1, 1, 1}).reshape({1, 2, 2}).to(torch::kInt64).cuda();

    torch::Tensor output = torch::zeros_like(index).to(torch::kFloat32).cuda();

    auto inputs = std::vector<void*>({b.data_ptr(), index.data_ptr()});
    auto outputs = std::vector<void*>({output.data_ptr()});

    for(int i = 0; i != 3; ++i){
        auto res = torch::gather(b, 1, index);


        gather_elements(
                inputs.data(), outputs.data(), 1,
                b.size(0), b.size(1), b.size(2),
                index.size(0), index.size(1), index.size(2));

//
//        std::cout << res << std::endl;
//        std::cout << output << std::endl;

        std::cout << (res - output).sum() << std::endl;

    }

    return 0;
}
