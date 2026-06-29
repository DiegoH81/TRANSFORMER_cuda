#ifndef PATCH_EMBEDDING_H
#define PATCH_EMBEDDING_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "tensor.cuh"
#include "layer.cuh"

#include "data_loader.cuh"

#include <vector>

__global__
void extract_patches( float* images, float* patches, int n_images)
{
    int img_idx = blockIdx.x;
    int patch_idx = blockIdx.y;

    int pixel_idx = threadIdx.x;

    if (img_idx >= n_images)
        return;

    // Patch position in 4x4 grid
    int patch_row = patch_idx / 4;
    int patch_col = patch_idx % 4;

    // Pixel position in 7x7 grid
    int pixel_row = pixel_idx / 7;
    int pixel_col = pixel_idx % 7;

    // Real coordinate
    int img_row = patch_row * 7 + pixel_row;
    int img_col = patch_col * 7 + pixel_col;

    int src = img_idx * 784 + img_row * 28 + img_col;
    int dst = img_idx * (16 * 49) + patch_idx * 49 + pixel_idx;

    patches[dst] = images[src];
}
class PatchEmbedding
{
public:
    Layer* projection;
    Tensor images_tensor, patches_tensor;
    Tensor* previous;

    int n_images, num_patches, patch_size, dim_model;

    PatchEmbedding(int in_n_images):
        images_tensor(), patches_tensor(), projection(nullptr),
        num_patches(16), patch_size(49)
    {
        n_images = in_n_images;

        patches_tensor.set_size(n_images * num_patches * patch_size);

        previous = &patches_tensor; // Previous = The data image but re-ordered

        dim_model = 64;
        projection = new Layer(patch_size, dim_model, n_images * num_patches, ActivationType::None, &patches_tensor);
    }
    
    void set_batch(std::vector<float>& batch_images)
    {
        images_tensor.load(batch_images);
    }

    void forward()
    {
        dim3 grid(n_images, 16);
        int threads = 49;

        extract_patches << <grid, threads >> > ( images_tensor.data, patches_tensor.data, n_images );
        cudaDeviceSynchronize();

        projection->forward();
    }

    void backward(float learning_rate)
    {
        projection->compute_error_intermediate();
        projection->update_weights(learning_rate);
    }

    void zero_grad()
    {
        projection->zero_grad();
    }
};

#endif