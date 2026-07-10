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
	Transformer(size_t in_batch, size_t num_encoders_blocks) :
		n_batches(in_batch),
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

	void forward(Tensor* in_data)
	{
		patch.forward(in_data);
		CLS.forward();
		positional.forward();

		for (auto& encoder : encoder_blocks)
			encoder->forward();
		classification_head.forward();
	}

private:
	PatchEmbedding patch;
	CLS_token CLS;
	PositionalEncoding positional;
	Classification classification_head;
	
	std::vector<EncoderBlock*> encoder_blocks;

	size_t n_batches;
};

#endif