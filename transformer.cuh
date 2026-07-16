#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include <vector>

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

		for (auto& encoder : encoder_blocks)
			encoder->forward();
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
private:
	PatchEmbedding patch;
	CLS_token CLS;
	PositionalEncoding positional;
	Classification classification_head;
	
	std::vector<EncoderBlock*> encoder_blocks;

};

#endif