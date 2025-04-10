#pragma once
#include <cuda_runtime.h>
#include <string>
#include <iostream>

namespace cuda_matrix
{
    // Helper function to allocate memory on device with optional copy from host
    template <typename T>
    T* device_alloc(T* h_input, int rows, int cols)
    {
        T* d_mat;
        cudaMalloc((void**)&d_mat, rows * cols * sizeof(T));
        if (h_input != nullptr)
        {
            // If h_input is not null, copy the data from host to device
            cudaMemcpy(d_mat, h_input, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
        }
        return d_mat;
    }

    // Helper function to free device memory
    template <typename T>
    void device_free(T* d_input)
    {
        if (d_input != nullptr) {
            cudaFree(d_input);
            d_input = nullptr; // Set to null to avoid dangling pointer
        }
    }


    template <typename T>
    void to(T* h_mat, T* d_mat, int rows, int cols, std::string device)
    {
        if (device == "cpu" || device == "host")
        {
            // If the device is CPU then copy from device to host
            cudaMemcpy(h_mat, d_mat, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);
            
        }
        else if (device == "gpu" || device == "device")
        {
            // If the device is GPU then copy from host to device
            cudaMemcpy(d_mat, h_mat, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
        }
        else
        {
            fprintf(stderr, "Error: Invalid device specified. Use 'cpu' or 'gpu'.\n");
            return;
        }

    }

    void cuda_config(int block_size_x, int block_size_y, int device_id);
    int cuda_device_count();
    void cuda_synchronize();
    void set_block_size(int block_size_x, int block_size_y);
    void m_copy(const float* d_src, float* d_dest, int rows, int cols);
    void m_copy_row(const float* d_src, float* d_dest, int src_row, int dest_row, int rows1, int rows2, int cols);
    void m_sum(const float* d_input, float* d_output, int rows, int cols, int axis);
    void m_argmax(const float* d_input, int* d_output, int rows, int cols, int axis);
    void m_add(float* d_input1, const float* d_input2, int rows, int cols);
    void m_add_v(float* d_input1_m, const float* d_input2_v, int rows1, int cols1, int rows2, int cols2);
    void m_sub(float* d_input1, const float* d_input2, int rows, int cols);
    void m_mul(const float* d_input1, const float* d_input2, float* d_output, int rows1, int cols1, int rows2, int cols2);
    void m_scalar_mul(float* d_input, float d_scalar, int rows, int cols);
    void m_transpose(float* d_input, float* d_output, int rows, int cols);
    void m_hadamard(float* d_input1, const float* d_input2, int rows, int cols);
    void m_Relu(float* input, int rows, int cols);
    void m_Relu_deriv(float* input, int rows, int cols);
    void m_index_to_one_hot(float* input, float* output, int rows, int cols);
    void m_softmax(float* input, int rows, int cols);
    bool m_check_nan_inf(const float* d_input, int rows, int cols);
}