
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <algorithm>
#include <random>

#include "tensor.cuh"
#include "transformer.cuh"
#include "data_loader.cuh"


int main()
{
    Data train_data(28 * 28);
    train_data.load_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

    Data evaluation_data(28 * 28);
    evaluation_data.load_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
    

    int batch_size = 128;
    float learning_rate = 0.005f;
    float accuracy_train = 0.0f, accuracy_eval = 0.0f;

    Transformer* model = nullptr;// (batch_size, 3, learning_rate);
    bool can_work = false;

    Tensor input_data, exected_data;

    while (true)
    {
        int option = 0;

        std::cout << "-------------Menu-------------\n";
        std::cout << "1. Load weights\n";
        std::cout << "2. Train\n";
        std::cout << "3. Test\n";
        std::cout << "4. Info\n";
        std::cout << "Enter option: ";
        std::cin >> option;
        std::cout << "------------------------------\n\n";

        switch (option)
        {
        case 1:
        {
            if (!model)
                model = new Transformer(batch_size, 3, learning_rate);

            model->load("TRANSFORMER_data.txt", accuracy_train, accuracy_eval);
            can_work = true;
            std::cout << "Transformer data loaded\n";
            break;
        }
        case 2:
        {
            model = new Transformer(batch_size, 3, learning_rate);
            can_work = true;

            int num_epochs = 5;

            int total_case = 0;
            std::cout << "Transformer training started\n";
            for (int epoch = 0; epoch < num_epochs; epoch++)
            {
                int total_correct = 0;
                std::cout << "Epoch: " << epoch + 1 << " / " << num_epochs + 1 << "\n";


                for (int i = 0; i + batch_size < train_data.n_samples; i += batch_size)
                {
                    std::vector<float> batch_images, batch_labels;
                    std::vector<int> batch_labels_int;

                    batch_labels_int = train_data.get_batch_labels_int(i, batch_size);
                    train_data.get_batch(i, batch_size, batch_images, batch_labels);

                    input_data.load(batch_images);
                    exected_data.load(batch_labels);

                    auto output = model->forward(&input_data);
                    model->update_weights(exected_data);
                    model->zero_grad();

                    // accuracy testing
                    auto preds = model->get_prediction();
                    for (int j = 0; j < batch_labels_int.size(); j++)
                        if (preds[j] == batch_labels_int[j])
                        {
                            total_correct++;
                            total_case++;
                        }

                    // Print info
                    if ((i / batch_size) % 100 == 0)
                    {
                        float partial_acc = (float)total_correct / (i + batch_size) * 100.0f;
                        std::cout << "  batch " << i / batch_size << "/" << train_data.n_samples / batch_size << "  acc parcial: " << partial_acc << "%\n";

                    }
                }

            }

            accuracy_train = (float)total_case / train_data.n_samples * 100.0f;
            std::cout << " - Training accuracy: " << accuracy_train << "%\n";

            break;
        }
        case 3:
        {
            if (!can_work)
                break;

            // Evaluation
            int eval_correct = 0;
            for (int i = 0; i + batch_size < evaluation_data.n_samples; i += batch_size)
            {
                std::vector<float> batch_images, batch_labels;
                std::vector<int> batch_labels_int;
                evaluation_data.get_batch(i, batch_size, batch_images, batch_labels);
                batch_labels_int = evaluation_data.get_batch_labels_int(i, batch_size);

                input_data.load(batch_images);
                exected_data.load(batch_labels);

                model->forward(&input_data);

                auto preds = model->get_prediction();

                for (int j = 0; j < batch_labels_int.size(); j++)
                    if (preds[j] == batch_labels_int[j])
                        eval_correct++;
            }

            accuracy_eval = (float)eval_correct / evaluation_data.n_samples * 100.0f;
            std::cout << " - Evaluation accuracy: " << accuracy_eval << "%\n";
        }
        case 4:
        {
            std::cout << "Transformer Info\n";
            std::cout << "- Training accuracy: " << accuracy_train << "\n";
            std::cout << "- Evaluation accuracy: " << accuracy_eval << "\n";
            break;
        }
        }
    }

    return 0;
}