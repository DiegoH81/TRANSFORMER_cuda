#ifndef CLASSIFICATION_HEAD_H
#define CLASSIFICATION_HEAD_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tensor.cuh"
#include "layer.cuh"


__global__
void extract_cls(float* encoder_out, float* cls_out, int seq_len, int dim_model, int n_images)
{
    int img = blockIdx.x;
    int d = threadIdx.x;
    if (img >= n_images || d >= dim_model)
        return;

    cls_out[img * dim_model + d] = encoder_out[img * seq_len * dim_model + d];
}

__global__
void extract_cls_backward(float* d_cls, float* d_encoder_out, int seq_len, int dim_model, int n_images)
{
    int img = blockIdx.x;
    int d = threadIdx.x;
    if (img >= n_images || d >= dim_model)
        return;

    //d_encoder_out[img * seq_len * dim_model + d] = d_cls[img * dim_model + d];
    atomicAdd(&d_encoder_out[img * seq_len * dim_model + d], d_cls[img * dim_model + d]);
}

__global__
void softmax_forward(float* x, float* out, int n_classes, int n_images)
{
    int img = blockIdx.x * blockDim.x + threadIdx.x;
    if (img >= n_images)
        return;

    float* row = x + img * n_classes;
    float* outrow = out + img * n_classes;

    float maxval = row[0];
    for (int i = 1; i < n_classes; i++)
        maxval = fmaxf(maxval, row[i]);

    float sum = 0.0f;
    for (int i = 0; i < n_classes; i++)
        sum += expf(row[i] - maxval);

    for (int i = 0; i < n_classes; i++)
        outrow[i] = expf(row[i] - maxval) / sum;
}

class ClassificationHead
{
public:
    int n_images, dim_model, n_classes;

    Tensor cls_tokens;
    Tensor logits;
    Tensor output;

    Layer linear;
    Tensor* previous;

    ClassificationHead(int in_n_images, int in_dim_model, int in_n_classes = 10)
        : n_images(in_n_images), dim_model(in_dim_model), n_classes(in_n_classes),
          linear(in_dim_model, in_n_classes, in_n_images, ActivationType::None, nullptr)
    {
        cls_tokens.set_size(n_images * dim_model);
        logits.set_size(n_images * n_classes);
        output.set_size(n_images * n_classes);
    }

    void forward(int seq_len)
    {
        dim3 grid(n_images);
        int  threads = dim_model;
        extract_cls << <grid, threads >> > (previous->data, cls_tokens.data, seq_len, dim_model, n_images);
        cudaDeviceSynchronize();

        // Linear
        linear.previous_layer = &cls_tokens;
        linear.forward();

        // Soft_Max
        int thr = 256;
        int blk = (n_images + thr - 1) / thr;
        softmax_forward << <blk, thr >> > (linear.output.data, output.data, n_classes, n_images);
        cudaDeviceSynchronize();
    }

    
    void backward(Tensor& expected, float learning_rate, int seq_len)
    {
        linear.compute_error_last(expected);

        linear.apply_derivative();
        linear.update_weights(learning_rate);
        linear.compute_error_intermediate();

     
        dim3 grid(n_images);
        int  threads = dim_model;

        extract_cls_backward << <grid, threads >> > (cls_tokens.gradient, previous->gradient, seq_len, dim_model, n_images);
        cudaDeviceSynchronize();
    }

    void zero_grad()
    {
        cls_tokens.zero_grad();
        logits.zero_grad();
        output.zero_grad();
        linear.zero_grad();
    }

  
    std::vector<int> predictions()
    {
        auto cpu = output.get_data_CPU();
        std::vector<int> preds(n_images);

        for (int img = 0; img < n_images; img++)
        {
            int best = 0;
            float best_val = cpu[img * n_classes];
            for (int c = 1; c < n_classes; c++)
            {
                if (cpu[img * n_classes + c] > best_val)
                {
                    best_val = cpu[img * n_classes + c];
                    best = c;
                }
            }
            preds[img] = best;
        }
        return preds;
    }

    float accuracy(std::vector<int>& labels)
    {
        auto preds = predictions();
        int correct = 0;
        for (int i = 0; i < n_images; i++)
            if (preds[i] == labels[i])
                correct++;

        return (float)correct / n_images * 100.0f;
    }
};

#endif