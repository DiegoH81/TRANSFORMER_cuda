
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

#include "tensor.cuh"
#include "transformer.cuh"
#include "data_loader.cuh"


int main()
{
    Data train_data(28 * 28);
    train_data.load_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

    Data evaluation_data(28 * 28);
    evaluation_data.load_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
    
    std::vector<float> all_images_data, all_expected_outputs;
    train_data.get_image(0, all_images_data, all_expected_outputs);
    Tensor input_total(all_images_data), expected_total(all_expected_outputs);

    std::cout << "Data loaded, " << train_data.n_samples << "\n";

    Transformer testin(1, 2);

    auto ans = testin.forward(&input_total);
    
    auto ans_CPU = ans.get_data_CPU();

    for (int i = 0; i < 10; i++)
        std::cout << ans_CPU[i] << " ";
    std::cout << "\n";

    return 0;
}