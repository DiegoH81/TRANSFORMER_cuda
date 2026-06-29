#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

__global__
void normalize(uint8_t* images, int n_samples, int img_size)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample >= n_samples)
        return;

    for (int p = 0; p < img_size; p++)
        images[sample * img_size + p] = images[sample * img_size + p] > 120 ? 1 : 0;
}

class Data
{
public:
    int n_samples, image_size;
    uint8_t* images, *expected_output;

    Data(int in_size) : n_samples(0), image_size(in_size)
    {
    }

    ~Data()
    {
        if (images)
            cudaFreeHost(images);
        if (expected_output)
            cudaFreeHost(expected_output);
    }

    void load_data(std::string image_path, std::string labels_path)
    {
        std::ifstream labels_file(labels_path, std::ios::binary);
        readBE32(labels_file);
        n_samples = readBE32(labels_file);
        
        // Reserve memory
        size_t total_bytes = n_samples * image_size;
        cudaMallocHost(&images, total_bytes);
        cudaMallocHost(&expected_output, n_samples);

        labels_file.read((char*)expected_output, n_samples);

        std::ifstream images_file(image_path, std::ios::binary);
        readBE32(images_file);
        readBE32(images_file);
        readBE32(images_file);
        readBE32(images_file);


        for (int s = 0; s < n_samples; s++)
            images_file.read((char*)&images[s * image_size], image_size);


        uint8_t* d_images;

        cudaMalloc(&d_images, total_bytes);
        cudaMemcpy(d_images, images, total_bytes, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (n_samples + threads - 1) / threads;

        normalize << <blocks, threads >> > (d_images, n_samples, image_size);
        cudaDeviceSynchronize();

        cudaMemcpy(images, d_images, total_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_images);
    }

    void get_image(int index, std::vector<float>& out_data, std::vector<float>& out_expected)
    {
        if (index >= n_samples)
            return;

        out_data.clear();
        out_data.reserve(image_size);

        int start_index = index * image_size;

        for (int i = 0; i < image_size; i++)
            out_data.push_back(images[start_index + i]);

        out_expected.resize(10, 0);
        out_expected[expected_output[index]] = 1;
    }

    void get_all(std::vector<float>& out_data, std::vector<float>& expected)
    {
        out_data.clear();
        out_data.reserve(image_size);

        expected.reserve(n_samples);
        out_data.reserve(n_samples * image_size);

        for (int i = 0; i < n_samples; i++)
        {
            std::vector<float> to_push(10, 0);
            to_push[expected_output[i]] = 1;

            expected.insert(expected.end(), to_push.begin(), to_push.end());

            int start_idx = i * image_size;
            for (int j = 0; j < image_size; j++)
                out_data.push_back(images[start_idx + j]);
        }
    }

    void get_batch(int start_index, int batch_size, std::vector<float>& out_data, std::vector<float>& out_expected)
    {
        int count = std::min(batch_size, n_samples - start_index);

        out_data.clear();
        out_expected.clear();
        out_data.reserve(count * image_size);
        out_expected.reserve(count * 10);

        for (int i = 0; i < count; i++)
        {
            // Label one-hot
            std::vector<float> to_push(10, 0);
            to_push[expected_output[start_index + i]] = 1;
            out_expected.insert(out_expected.end(), to_push.begin(), to_push.end());

            // Imagen
            int start_idx = (start_index + i) * image_size;
            for (int j = 0; j < image_size; j++)
                out_data.push_back((float)images[start_idx + j]);
        }
    }

    std::vector<int> get_batch_labels_int(int start_index, int batch_size)
    {
        int count = std::min(batch_size, n_samples - start_index);
        std::vector<int> labels(count);
        for (int i = 0; i < count; i++)
            labels[i] = expected_output[start_index + i];
        return labels;
    }


private:
    uint32_t readBE32(std::ifstream& file)
    {
        uint8_t value[4];
        file.read(reinterpret_cast<char*>(value), 4);
        return (value[0] << 24) | (value[1] << 16) | (value[2] << 8) | value[3];
    }
};

void print_ascii_number(std::vector<float>& num)
{
    for (int y = 0; y < 28; y++)
    {
        for (int x = 0; x < 28; x++)
        {
            int p = y * 28 + x;
            char c = (num[p] < 1) ? '#' : ' ';
            std::cout << c;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

#endif
