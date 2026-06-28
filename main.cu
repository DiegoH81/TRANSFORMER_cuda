
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "tensor.cuh"
#include "layer.cuh"
#include "activation_function.cuh"
#include "MLP.cuh"
#include "data_loader.cuh"

int main()
{
    Data train_data(28 * 28);
    train_data.load_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

    Data evaluation_data(28 * 28);
    evaluation_data.load_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");


    float accuracy_train = 0.0f, accuracy_eval = 0.0f;
    MLP* mlp_MNIST = nullptr;

    while (true)
    {
        std::cout << "-------------Menu-------------\n";
        std::cout << "1. Train MLP\n";
        std::cout << "2. Load weights\n";
        std::cout << "3. Test MLP\n";
        std::cout << "4. Show info\n";
        std::cout << "5. Exit\n";
        std::cout << "Enter option: ";
        std::string option;
        std::getline(std::cin, option);
        std::cout << "------------------------------\n\n";
        
        if (option == "1")
        {
            if (mlp_MNIST)
                delete mlp_MNIST;

            mlp_MNIST = new MLP({ LayerInfo(train_data.image_size, 128, ActivationType::ReLu),
                                  LayerInfo(128, 64, ActivationType::ReLu),
                                  LayerInfo(64, 10, ActivationType::None) }, 0.1f);

            std::cout << "Enter number of epochs to train: ";
            std::string number_of_epochs;
            std::getline(std::cin, number_of_epochs);

            // Data to tensor
            std::vector<float> all_images_data, all_expected_outputs;
            train_data.get_all(all_images_data, all_expected_outputs);
            Tensor input_total(all_images_data), expected_total(all_expected_outputs);


            mlp_MNIST->enable_traininig(train_data.n_samples);

            for (int i = 0; i < std::stoi(number_of_epochs); i++)
            {
                mlp_MNIST->evaluate(input_total);
                mlp_MNIST->update_weights(expected_total);
                mlp_MNIST->zero_grad();
            }

            // Evaluation:
            mlp_MNIST->disable_training();
            accuracy_train = mlp_MNIST->accuracy(train_data, 500);
            accuracy_eval = mlp_MNIST->accuracy(evaluation_data, 500);
            std::cout << "Train: " << accuracy_train << "%" << " | Eval: " << accuracy_eval << "%\n\n";
            std::cout << "------------------------------\n\n";
        }
        else if (option == "2")
        {
            if (mlp_MNIST)
                delete mlp_MNIST;

            mlp_MNIST = load("MLP_data.txt", accuracy_train, accuracy_eval);
            std::cout << "MLP loaded\n\n";
            std::cout << "------------------------------\n\n";
        }
        else if (option == "3")
        {
            if (!mlp_MNIST)
            {
                std::cout << "Unloaded MLP\n\n";
                continue;
            }

            mlp_MNIST->disable_training();
            mlp_MNIST->accuracy_random(train_data, 10);
            std::cout << "------------------------------\n\n";
        }
        else if (option == "4")
        {
            if (!mlp_MNIST)
            {
                std::cout << "Unloaded MLP\n\n";
                continue;
            }
            mlp_MNIST->print_info(accuracy_train, accuracy_eval);
            std::cout << "------------------------------\n\n";
        }
        else if (option == "5")
            break;
    }
    return 0;
}