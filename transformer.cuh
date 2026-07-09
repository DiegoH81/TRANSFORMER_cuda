#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include "patch_embedding.cuh"
#include "CLS_token.cuh"
#include "positional_encoding.cuh"

class Transformer
{
public:
	Transformer(size_t in_batch) :
		n_batches(in_batch),
		patch(28 * 28, 7, 64, n_batches),
		CLS(patch.n_patches,
			patch.linear_dim,
			&patch.linear.output,
			n_batches),
		positional(CLS.n_patches + 1,
				   CLS.patch_dim,
				   &CLS.output,
				   n_batches)
	{

	}

	void forward(Tensor* in_data)
	{
		patch.forward(in_data);
		CLS.forward();
		positional.forward();
	}

private:
	PatchEmbedding patch;
	CLS_token CLS;
	PositionalEncoding positional;

	size_t n_batches;
};

#endif