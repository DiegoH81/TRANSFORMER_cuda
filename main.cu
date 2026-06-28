
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_loader.cuh"
#include "Transformer.cuh"


int main()
{
    Data train_data(28 * 28);
    train_data.load_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

    Data evaluation_data(28 * 28);
    evaluation_data.load_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

    std::cout << "INPUT DATA\n";
    std::cout << "- image size: " << train_data.image_size << "\n";
    std::cout << "- number of images: " << train_data.n_samples << "\n";

    Transformer testin(train_data, 0.1f);

    auto data = testin.forward();

    testin.backward();
    testin.zero_grad();
    

    //data.print();
    auto cpu_data = data.get_data_CPU();
    std::cout << "DATA  SIZE: " << cpu_data.size() << "\n";
    return 0;
}