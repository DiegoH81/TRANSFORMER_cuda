#ifndef TRANSFORMER_H
#define TRANSFORMER_HA

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CLS_token.cuh"
#include "PositionalEncoding.cuh"
#include "PatchEmbedding.cuh"

#include "data_loader.cuh"

class Transformer
{
public:
    Transformer(Data& in_training_data, float in_learning_rate):
        training_data(in_training_data),
        patch_embedding(training_data),
        CLS(patch_embedding.dim_model, patch_embedding.num_patches, patch_embedding.n_images),
        position_encoding(CLS.num_patches + 1, CLS.dim_model, CLS.n_images),

        learning_rate(in_learning_rate)
    { }

    Tensor forward()
    {
        patch_embedding.forward();
        //std::cout << "PatchEmb output size: " << patch_embedding.projection->output.size << "\n";


        CLS.previous = &patch_embedding.projection->output;
        CLS.forward();
        //std::cout << "CLS output size: " << CLS.output.size << "\n";

        position_encoding.previous = &CLS.output;
        position_encoding.forward();
        //std::cout << "PosEnc output size: " << position_encoding.previous->size << "\n";

        return *position_encoding.previous; // Temporal for testing only
                                            // LA ULTIMA SALIDA ES IN PLACE, es decir la data esta en:
                                            // position_encoding.previous
    }

    void zero_grad()
    {
        patch_embedding.zero_grad();
        CLS.zero_grad();
        position_encoding.zero_grad();
    }

    void backward()
    {
        position_encoding.backward();
        position_encoding.update_weights(learning_rate);

        CLS.backward();
        CLS.update_weights(learning_rate);

        patch_embedding.backward(learning_rate); // Update weights is inside backward here
    }
private:
    Data& training_data;
    float learning_rate;

    PatchEmbedding patch_embedding;
    CLSToken CLS;
    PositionalEncoding position_encoding;
};

#endif