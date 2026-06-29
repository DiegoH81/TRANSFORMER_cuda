#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_loader.cuh"
#include "Transformer.cuh"
#include "layer_norm.cuh"

void test_layernorm()
{
    // --- CONFIGURACION ---
    int dim_model = 4;   // pequeno para verificar a mano
    int seq_len = 1;
    int n_images = 1;

    // Input conocido: [2, 4, 4, 2]
    // mean = (2+4+4+2)/4 = 3.0
    // var  = ((2-3)^2 + (4-3)^2 + (4-3)^2 + (2-3)^2) / 4
    //      = (1 + 1 + 1 + 1) / 4 = 1.0
    // std  = sqrt(1.0 + 1e-5) ≈ 1.0
    // x_hat = [-1, 1, 1, -1]
    // con gamma=1, beta=0 → output = [-1, 1, 1, -1]

    std::vector<float> input_data = { 2.0f, 4.0f, 4.0f, 2.0f };
    std::vector<float> expected = { -1.0f, 1.0f, 1.0f, -1.0f };

    // Crear tensor input
    Tensor input(input_data);

    // Crear LayerNorm
    LayerNorm ln(dim_model, seq_len, n_images, &input);

    // Forward
    ln.forward();

    // Bajar resultado a CPU
    auto result = ln.output.get_data_CPU();

    // Verificar
    std::cout << "=== TEST LAYER NORM FORWARD ===\n";
    bool passed = true;
    for (int i = 0; i < dim_model; i++)
    {
        float diff = std::abs(result[i] - expected[i]);
        std::cout << "output[" << i << "] = " << result[i]
            << "  esperado = " << expected[i]
            << "  diff = " << diff;

        if (diff < 1e-4f)
            std::cout << "  OK\n";
        else
        {
            std::cout << "  FALLO\n";
            passed = false;
        }
    }

    if (passed)
        std::cout << "\nTEST PASSED\n";
    else
        std::cout << "\nTEST FAILED\n";
}

int main()
{
    Data train_data(28 * 28);
    train_data.load_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

    Data evaluation_data(28 * 28);
    evaluation_data.load_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

    std::cout << "INPUT DATA\n";
    std::cout << "- image size: " << train_data.image_size << "\n";
    std::cout << "- number of images: " << train_data.n_samples << "\n";


    int batch_size = 128;

    Transformer testin(batch_size, 0.1f);

    //for (int i = 0; i < train_data.n_samples; i += batch_size)
    {
        std::vector<float> batch_images, batch_labels;
        train_data.get_batch(0, batch_size, batch_images, batch_labels);

        testin.set_batch(batch_images);

        auto output = testin.forward();
        testin.backward();
        testin.zero_grad();
        
        //data.print();
        auto cpu_data = output.get_data_CPU();
        std::cout << "DATA  SIZE: " << cpu_data.size() << "\n";
    }

    

    //test_layernorm();
    return 0;
}