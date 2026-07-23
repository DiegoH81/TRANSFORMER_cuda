#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include <vector>
#include <string>
#include <fstream>

#include "patch_embedding.cuh"
#include "CLS_token.cuh"
#include "positional_encoding.cuh"
#include "encoder_block.cuh"
#include "classification.cuh"

class Transformer
{
public:
	size_t n_batches;
	float learning_rate;
	Transformer(size_t in_batch, size_t num_encoders_blocks, float in_lr = 0.1f) :
		n_batches(in_batch),
		learning_rate(in_lr),
		patch(28 * 28, 7, 64, n_batches),
		CLS(patch.n_patches,
			patch.linear_dim,
			&patch.linear.output,
			n_batches),
		positional(CLS.n_patches + 1,
			CLS.patch_dim,
			&CLS.output,
			n_batches),
		classification_head(positional.n_patches, positional. patch_dim, nullptr, n_batches)
	{
		Tensor* current_input = &positional.output;
		for (int i = 0; i < num_encoders_blocks; i++)
		{
			EncoderBlock* to_push = new EncoderBlock(positional.n_patches, positional.patch_dim, current_input, n_batches);
			encoder_blocks.push_back(to_push);

			current_input = &to_push->output;
		}
		
		classification_head.previous = current_input;
	}

	~Transformer()
	{
		for (auto* block : encoder_blocks)
			delete block;
	}

	Tensor forward(Tensor* in_data)
	{
		patch.forward(in_data);
		CLS.forward();
		positional.forward();
		
		int counter = 0;
		for (auto& encoder : encoder_blocks)
		{
			encoder->forward();
			counter++;
		}

		classification_head.forward();

		return classification_head.classification_layer.output;
	}

	void zero_grad()
	{
		classification_head.zero_grad();

		for (auto& encoder : encoder_blocks)
			encoder->zero_grad();

		positional.zero_grad();
		CLS.zero_grad();
		patch.zero_grad();
	}

	void update_weights(Tensor& expected)
	{
		classification_head.backward(expected, learning_rate);

		for (auto rev_it = encoder_blocks.rbegin(); rev_it != encoder_blocks.rend(); rev_it++)
			(*rev_it)->backward(learning_rate);

		positional.backward(learning_rate);
		CLS.backward(learning_rate);
		patch.backward(learning_rate);
	}

	std::vector<int> get_prediction()
	{
		auto cpu = classification_head.classification_layer.output.get_data_CPU();
		std::vector<int> preds(n_batches);

		for (int img = 0; img < n_batches; img++)
		{
			int best = 0;
			float best_val = cpu[img * 10];
			for (int c = 1; c < 10; c++)
			{
				//std::cout << "CPU: " << cpu[img * 10 + c] << "\n";
				if (cpu[img * 10 + c] > best_val)
				{
					best_val = cpu[img * 10 + c];
					best = c;
				}
			}
			preds[img] = best;
		}
		return preds;
	}

	void save(const std::string& path, float accuracy_training = 0.0f, float accuracy_eval = 0.0f)
	{
		std::ofstream file(path, std::ios::out);

		file << accuracy_training << "\n";
		file << accuracy_eval << "\n";
		file << learning_rate << "\n";
		file << patch.batch_size << "\n";
		file << encoder_blocks.size() << "\n";

		patch.save_weights(file);
		CLS.save_weights(file);
		positional.save_weights(file);

		for (auto& encoder : encoder_blocks)
			encoder->save_weights(file);

		classification_head.save_weights(file);

		file.close();
	}

	void load(const std::string& path, float& out_accuracy_training, float& out_accuracy_eval)
	{
		std::ifstream file(path, std::ios::in);

		size_t saved_batch_size = 0, saved_num_blocks = 0;
		float saved_lr = 0.0f;

		file >> out_accuracy_training;
		file >> out_accuracy_eval;
		file >> saved_lr;
		file >> saved_batch_size;
		file >> saved_num_blocks;

		if (saved_batch_size != n_batches || saved_num_blocks != encoder_blocks.size())
		{
			std::cout << "Different parameters detected for saved transformer and already built transformer.\n";
			return;
		}

		learning_rate = saved_lr;

		patch.load_weights(file);
		CLS.load_weights(file);
		positional.load_weights(file);

		for (auto& encoder : encoder_blocks)
			encoder->load_weights(file);

		classification_head.load_weights(file);

		file.close();
	}

	PatchEmbedding patch;
	CLS_token CLS;
	PositionalEncoding positional;
	Classification classification_head;
	
	std::vector<EncoderBlock*> encoder_blocks;

};


#endif