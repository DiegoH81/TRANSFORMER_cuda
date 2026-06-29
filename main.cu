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

    Transformer testin(batch_size, 0.001f);

    float total_loss = 0.0f;
    int total_correct = 0;
    int max_epochs = 15;

    for (int epoch = 1; epoch <= max_epochs; epoch++)
    {
        total_correct = 0;

        std::cout << "EPOCH " << epoch << "/" << max_epochs << "\n";

        for (int i = 0; i < train_data.n_samples; i += batch_size)
        {
            std::vector<float> batch_images, batch_labels;
            std::vector<int> batch_labels_int;


            batch_labels_int = train_data.get_batch_labels_int(i, batch_size);
            train_data.get_batch(i, batch_size, batch_images, batch_labels);

            testin.set_batch(batch_images);

            auto output = testin.forward();

            Tensor expected(batch_labels);
            testin.backward(expected);
            testin.zero_grad();

            // Accuracy testing
            auto preds = testin.class_head.predictions();
            for (int j = 0; j < batch_labels_int.size(); j++)
                if (preds[j] == batch_labels_int[j])
                    total_correct++;

            // Print cada 100 batches
            if ((i / batch_size) % 100 == 0)
            {
                float partial_acc = (float)total_correct / (i + batch_size) * 100.0f;
                std::cout << "  batch " << i / batch_size << "/" << train_data.n_samples / batch_size << "  acc parcial: " << partial_acc << "%\n";
            }
        }
    }

    float train_acc = (float)total_correct / train_data.n_samples * 100.0f;
    std::cout << " - Train acc: " << train_acc << "%\n";


    std::cout << "\nEVALUATION\n";
    int eval_correct = 0;
    for (int i = 0; i < evaluation_data.n_samples; i += batch_size)
    {
        std::vector<float> batch_images, batch_labels_onehot;
        std::vector<int> batch_labels_int;
        evaluation_data.get_batch(i, batch_size, batch_images, batch_labels_onehot);
        batch_labels_int = evaluation_data.get_batch_labels_int(i, batch_size);

        testin.set_batch(batch_images);
        testin.forward();

        auto preds = testin.class_head.predictions();
        for (int j = 0; j < batch_labels_int.size(); j++)
            if (preds[j] == batch_labels_int[j])
                eval_correct++;
    }

    float eval_acc = (float)eval_correct / evaluation_data.n_samples * 100.0f;
    std::cout << " - Eval acc: " << eval_acc << "%\n";


    //test_layernorm();
    return 0;
}