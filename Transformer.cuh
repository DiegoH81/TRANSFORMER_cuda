#ifndef TRANSFORMER_H
#define TRANSFORMER_HA

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CLS_token.cuh"
#include "PositionalEncoding.cuh"
#include "PatchEmbedding.cuh"

#include "EncoderBlock.cuh"
#include "ClassificationHead.cuh"



class Transformer
{
public:
    float learning_rate;

    PatchEmbedding patch_embedding;
    CLSToken CLS;
    PositionalEncoding position_encoding;
    ClassificationHead class_head;

    std::vector<EncoderBlock> encoders;


    Transformer(int in_batch_size, float in_learning_rate, int n_encoder_blocks = 2):
        patch_embedding(in_batch_size),
        CLS(patch_embedding.dim_model, patch_embedding.num_patches, patch_embedding.n_images),
        position_encoding(CLS.num_patches + 1, CLS.dim_model, CLS.n_images),
        class_head(in_batch_size, position_encoding.dim_model, 10),
        learning_rate(in_learning_rate)
    {
        for (int i = 0; i < n_encoder_blocks; i++)
            encoders.push_back(EncoderBlock(position_encoding.n_images, position_encoding.sequence_len,
                                            position_encoding.dim_model, 4, in_learning_rate));
        
    }

    Tensor Transformer::forward()
    {
        patch_embedding.forward();

        //std::cout << "\n===== PATCH EMBEDDING =====\n";
        //patch_embedding.patches_tensor.print(500);


        CLS.previous = &patch_embedding.projection->output;
        CLS.forward();

        //std::cout << "\n===== CLS =====\n";
        //CLS.output.print(5);

        position_encoding.previous = &CLS.output;
        position_encoding.forward();

        //std::cout << "\n===== POSITION ENCODING =====\n";
        //CLS.output.print(5);

        //std::cout << "\n===== ENCODER =====\n";
        encoders[0].previous = position_encoding.previous;
        for (int i = 1; i < encoders.size(); i++)
            encoders[i].previous = &encoders[i - 1].output;

        for (auto& enc : encoders)
            enc.forward();

        //std::cout << "\n===== CLASSIFICATION HEAD =====\n";
        class_head.previous = &encoders.back().output;
        class_head.forward(position_encoding.sequence_len);

        return class_head.output;
    }

    void set_batch(std::vector<float>& batch_images)
    {
        patch_embedding.set_batch(batch_images);
    }

    void zero_grad()
    {
        patch_embedding.zero_grad();
        CLS.zero_grad();
        position_encoding.zero_grad();

        for (auto& enc : encoders)
            enc.zero_grad();

        class_head.zero_grad();
    }

    void backward(Tensor& expected)
    {
        class_head.backward(expected, learning_rate, position_encoding.sequence_len);

        for (int i = encoders.size() - 1; i >= 0; i--)
            encoders[i].backward(learning_rate);

        position_encoding.backward();
        position_encoding.update_weights(learning_rate);

        CLS.backward();
        CLS.update_weights(learning_rate);

        patch_embedding.backward(learning_rate); // Update weights is inside backward here


    }
};

#endif
