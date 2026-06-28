#ifndef MLP_H
#define MLP_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <random>

#include "layer.cuh"
#include "activation_function.cuh"
#include "data_loader.cuh"


class LayerInfo
{
public:
	size_t in_size, out_size;
	ActivationType activation;

	LayerInfo(size_t in, size_t out, ActivationType in_activation):
		in_size(in), out_size(out), activation(in_activation)
	{ }
};

class MLP
{
public:
	std::vector<Layer*> layers;

	MLP(std::vector<LayerInfo> in_config, float in_learning_rate):
		MLP_config(in_config), learning_rate(in_learning_rate), batch_size(1)
	{
		Tensor* previous_tensor = nullptr;

		for (auto& info : MLP_config)
		{
			info.in_size;
			info.out_size;

			Layer* to_push = new Layer(info.in_size, info.out_size, 1, info.activation, previous_tensor);
			
			layers.push_back(to_push);
			previous_tensor = &to_push->output;
		}
	}

	void print_info(float accuracy_training = 0.0f, float accuracy_eval = 0.0f)
	{
		std::cout << "MLP INFO\n";
		std::cout << "Accuracy training: " << accuracy_training << "\n";
		std::cout << "Accuracy evaluation: " << accuracy_eval << "\n";

		for (auto& info : MLP_config)
		{
			std::cout << "IN: " << info.in_size << " OUT: " << info.out_size << " Activation: ";
			if (info.activation == ActivationType::None)
				std::cout << "None";
			else if (info.activation == ActivationType::ReLu)
				std::cout << "ReLu";
			std::cout << "\n";
		}
	}

	void print_batch_info()
	{
		for (auto& layer : layers)
			if (layer)
			{
				std::cout << "In: " << layer->input_size << " Out: " << layer->output_size << "\n";
				std::cout << "Batch size: " << layer->batch_size << "\n";
			}
	}

	void save(const std::string& path, float accuracy_training = 0.0f, float accuracy_eval = 0.0f)
	{
		std::ofstream file(path, std::ios::out);

		file << accuracy_training << "\n";
		file << accuracy_eval << "\n";
		file << learning_rate << "\n";
		size_t n_layers = MLP_config.size();
		file << n_layers << "\n";

		for (auto& info : MLP_config)
		{
			file << info.in_size << " " << info.out_size << " ";
			if (info.activation == ActivationType::None)
				file << "None";
			else if (info.activation == ActivationType::ReLu)
				file << "ReLu";

			file << "\n";
		}


		for (auto& layer : layers)
		{
			auto w = layer->weights.get_data_CPU();

			for (float val : w)
				file << val << " ";

			file << "\n";


			auto b = layer->bias.get_data_CPU();
			for (float val : b)
				file << val << " ";
			file << "\n";
		}

		file.close();
	}

	Tensor evaluate(Tensor& input)
	{
		if (MLP_config.empty() || (input.size != layers[0]->batch_size * layers[0]->input_size))
			return Tensor({ 0.0f });

		layers[0]->previous_layer = &input;

		for (auto& layer : layers)
		{
			if (layer)
				layer->forward();
		}

		return layers.back()->output;
	}

	void update_weights(Tensor& expected)
	{
		for (int i = MLP_config.size() - 1; i >= 0; i--)
		{
			auto current_layer = layers[i];
			
			if (i == MLP_config.size() - 1)
				current_layer->compute_error_last(expected);
	
			current_layer->apply_derivative();
			current_layer->update_weights(learning_rate);
			current_layer->compute_error_intermediate();			
		}
	}

	void zero_grad()
	{
		for (auto& layer : layers)
		{
			if (layer)
			{
				layer->output.zero_grad();
				if (layer->previous_layer)
					layer->previous_layer->zero_grad();
			}
		}
	}

	void enable_traininig(size_t in_batch_size)
	{
		batch_size = in_batch_size;
		for (auto& layer : layers)
		{
			if (layer)
			{
				layer->batch_size = batch_size;
				layer->output.set_size(layer->output_size * batch_size);
			}
		}
	}

	void disable_training()
	{
		for (auto& layer : layers)
		{
			if (layer)
			{
				layer->batch_size = 1;
				layer->output.set_size(layer->output_size);
			}
		}
		batch_size = 1;
	}

	float accuracy(Data& data, int n_samples)
	{
		int correct = 0;

		for (int i = 0; i < n_samples; i++)
		{
			std::vector<float> image_data, expected_val;
			data.get_image(i, image_data, expected_val);

			Tensor input(image_data);
			auto result = evaluate(input);

			auto output_cpu = result.get_data_CPU();

			int predicted = 0;
			float max_val = -999.0f;

			for (int i = 0; i < output_cpu.size(); i++)
			{
				if (output_cpu[i] >= max_val)
				{
					max_val = output_cpu[i];
					predicted = i;
				}
			}

			int expected = 0;
			for (int i = 0; i < expected_val.size(); i++)
			{
				if (expected_val[i] == 1.0f)
				{
					expected = i;
					break;
				}
			}


			if (predicted == expected)
				correct++;
		}

		return (float)correct / n_samples * 100.0f;
	}

	void accuracy_random(Data& data, int n_samples)
	{
		int correct = 0;

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, data.n_samples - 1);

		for (int i = 0; i < n_samples; i++)
		{
			int random_index = dis(gen);

			std::vector<float> image_data, expected_val;
			data.get_image(random_index, image_data, expected_val);

			Tensor input(image_data);
			auto result = evaluate(input);

			auto output_cpu = result.get_data_CPU();

			int predicted = 0;
			float max_val = -999.0f;

			for (int i = 0; i < output_cpu.size(); i++)
			{
				if (output_cpu[i] >= max_val)
				{
					max_val = output_cpu[i];
					predicted = i;
				}
			}

			int expected = 0;
			for (int i = 0; i < expected_val.size(); i++)
			{
				if (expected_val[i] == 1.0f)
				{
					expected = i;
					break;
				}
			}


			if (predicted == expected)
				correct++;

			
			print_ascii_number(image_data);
			std::cout << "Expected: " << expected << " - Predicted: " << predicted << "\n\n";
			
		}

		std::cout << "Accuracy percentage of " << n_samples << " samples: " <<  (float)correct / n_samples * 100.0f << "%\n\n";
	}

private:
	std::vector<LayerInfo> MLP_config;
	float learning_rate;
	int batch_size;
};

MLP* load(const std::string& path, float& accuracy_train, float& accuracy_eval)
{
	std::ifstream file(path);
	file >> accuracy_train;

	file >> accuracy_eval;

	float learning_rate;
	file >> learning_rate;

	size_t n_layers;
	file >> n_layers;

	std::vector<LayerInfo> config;
	for (size_t i = 0; i < n_layers; i++)
	{
		size_t in, out;
		std::string act;
		file >> in >> out >> act;

		ActivationType activation = ActivationType::None;
		if (act == "ReLu")
			activation = ActivationType::ReLu;
		
		config.push_back(LayerInfo(in, out, activation));
	}

	MLP* mlp = new MLP(config, learning_rate);

	for (auto& layer : mlp->layers)
	{
		std::vector<float> w(layer->weights.size);
		for (auto& v : w)
			file >> v;
		layer->weights.load(w);

		std::vector<float> b(layer->bias.size);
		for (auto& v : b)
			file >> v;
		layer->bias.load(b);
	}

	return mlp;
}

#endif
