#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Tensor
{
public:
    float* data, * gradient;
    size_t size;

    Tensor() :
        data(nullptr), gradient(nullptr), size(0)
    {
    }

    Tensor(const Tensor& other) : size(other.size)
    {
        init_array();
        cudaMemcpy(data, other.data, size * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(gradient, other.gradient, size * sizeof(float), cudaMemcpyDefault);
    }

    Tensor& operator=(const Tensor& other)
    {
        if (this == &other)
            return *this;

        size = other.size;
        free_memory();
        init_array();
        cudaMemcpy(data, other.data, size * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(gradient, other.gradient, size * sizeof(float), cudaMemcpyDefault);

        return *this;
    }

    Tensor(const std::vector<float>& in_data) :
        size(in_data.size())
    {
        load(in_data);
    }

    ~Tensor()
    {
        //free_memory();
    }

    void set_size(size_t in_size)
    {
        size = in_size;

        free_memory();
        init_array();

        reset_array(data);
        reset_array(gradient);
    }

    void set_random(float limit)
    {
        std::vector<float> h_data(size);
        static std::mt19937 gen(std::random_device{}());

        std::uniform_real_distribution<float> dist(-limit, limit);

        for (auto& v : h_data)
            v = dist(gen);

        cudaMemcpy(data, h_data.data(), size * sizeof(float), cudaMemcpyDefault);
    }

    void load(const std::vector<float>& in_data)
    {
        size = in_data.size();

        free_memory();
        init_array();

        cudaMemcpy(data, in_data.data(), size * sizeof(float), cudaMemcpyDefault);
        reset_array(gradient);
    }

    void print()
    {
        auto CPU_vector = get_data_CPU();
        std::cout << "T { ";

        size_t vec_size = CPU_vector.size();
        for (size_t i = 0; i < vec_size; i++)
        {
            std::cout << CPU_vector[i];

            if (i != vec_size - 1)
                std::cout << ", ";
        }

        std::cout << " }\n";
    }

    void zero_grad()
    {
        reset_array(gradient);
    }

    std::vector<float> get_data_CPU()
    {
        std::vector<float> to_return(size);

        cudaMemcpy(to_return.data(), data, size * sizeof(float), cudaMemcpyDefault);

        return to_return;
    }
private:
    void init_array()
    {
        cudaMalloc(&data, size * sizeof(float));
        cudaMalloc(&gradient, size * sizeof(float));
    }

    void free_memory()
    {
        if (data)
            cudaFree(data);

        if (gradient)
            cudaFree(gradient);
    }

    void reset_array(float* in_array)
    {
        //if (in_array)
        cudaMemset(in_array, 0, size * sizeof(float));
    }
};

#endif
