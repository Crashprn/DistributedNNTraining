#pragma once
#include <cuda_runtime.h>
#include <string>
#include <iostream>

namespace cuda_matrix
{
    /*
    Helper function to allocate device memory for a matrix.
    If h_input is not null, it copies the data from host to device.
    IN:
        h_input: Pointer to the host input matrix (can be null).
        rows: Number of rows in the matrix.
        cols: Number of columns in the matrix.
    OUT:
        Returns a pointer to the allocated device memory.
    */
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

    /*
    Helper function to free device memory.
    IN:
        d_input: Pointer to the device input matrix (can be null).
    OUT:
        void
    */
    template <typename T>
    void device_free(T* d_input)
    {
        if (d_input != nullptr) {
            cudaFree(d_input);
            d_input = nullptr; // Set to null to avoid dangling pointer
        }
    }

    /*
    Helper function to copy data between host and device matrices.
    IN:
        h_mat: Pointer to the host matrix.
        d_mat: Pointer to the device matrix.
        rows: Number of rows in the matrix.
        cols: Number of columns in the matrix.
        device: String indicating the device type ("cpu" or "gpu").
    OUT:
        void 
    */
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

    /*
    Function to configure CUDA settings.
    IN:
        block_size_x: Number of threads in the x dimension of a block.
        block_size_y: Number of threads in the y dimension of a block.
        device_id: ID of the CUDA device to use.
    OUT:
        void
    */
    void cuda_config(int block_size_x, int block_size_y, int device_id);

    /*
    Function to get the number of available CUDA devices.
    IN:
        void
    OUT:
        Returns the number of CUDA devices available.
    */
    int cuda_device_count();

    /*
    Function to synchronize the CUDA device.
    IN:
        void
    OUT:
        void 
    */
    void cuda_synchronize();

    /*
    Function to copy a matrix from one device memory location to another.
    IN:
        d_src: Pointer to the source device matrix.
        d_dest: Pointer to the destination device matrix.
        rows: Number of rows in the matrix.
        cols: Number of columns in the matrix.
    OUT:
        d_dest: Pointer to the destination device matrix after copying.
    */
    void m_copy(const float* d_src, float* d_dest, int rows, int cols);

    /*
    Function to copy a specific row from one device matrix to another.
    IN:
        d_src: Pointer to the source device matrix.
        d_dest: Pointer to the destination device matrix.
        src_row: The row index in the source matrix to copy.
        dest_row: The row index in the destination matrix where the data will be copied.
        rows1: Total number of rows in the source matrix.
        rows2: Total number of rows in the destination matrix.
        cols: Number of columns in the matrices.
    OUT:
        d_dest: Pointer to the destination device matrix after copying the row.
    */
    void m_copy_row(const float* d_src, float* d_dest, int src_row, int dest_row, int rows1, int rows2, int cols);

    /*
    Function to perform a sum along a specified axis of a matrix.
    IN:
        d_input: Pointer to the input device matrix.
        d_output: Pointer to the output device matrix where the result will be stored.
        rows: Number of rows in the input matrix.
        cols: Number of columns in the input matrix.
        axis: Axis along which to perform the sum (0 for rows, 1 for columns).
    OUT:
        d_output: Pointer to the output device matrix after performing the sum.
        1 x cols if axis=0, rows x 1 if axis=1.
    */
    void m_sum(const float* d_input, float* d_output, int rows, int cols, int axis);

    /*
    Function to perform an argmax operation along a specified axis of a matrix.
    IN:
        d_input: Pointer to the input device matrix.
        d_output: Pointer to the output device matrix where the indices of the maximum values will be stored.
        rows: Number of rows in the input matrix.
        cols: Number of columns in the input matrix.
        axis: Axis along which to perform the argmax (0 for rows, 1 for columns).
    OUT:
        d_output: Pointer to the output device matrix after performing the argmax.
        1 x cols if axis=0, rows x 1 if axis=1. 
    */
    void m_argmax(const float* d_input, int* d_output, int rows, int cols, int axis);

    /*
    Function to perform element-wise addition of two matrices.
    IN:
        d_input1: Pointer to the first input device matrix.
        d_input2: Pointer to the second input device matrix.
        rows: Number of rows in the matrices.
        cols: Number of columns in the matrices.
    OUT:
        d_input1: Pointer to the first input device matrix after performing the addition.
    */
    void m_add(float* d_input1, const float* d_input2, int rows, int cols);

    /*
    Function to perform element-wise addition of a matrix and a vector with broadcasting.
    IN:
        d_input1_m: Pointer to the input device matrix.
        d_input2_v: Pointer to the input device vector (1D array).
        rows1: Number of rows in the matrix.
        cols1: Number of columns in the matrix.
        rows2: Number of rows in the vector (should be 1).
        cols2: Number of columns in the vector (should be 1).
    OUT:
        d_input1_m: Pointer to the input device matrix after performing the addition.
    */
    void m_add_v(float* d_input1_m, const float* d_input2_v, int rows1, int cols1, int rows2, int cols2);

    /*
    Function to perform element-wise subtraction of two matrices.
    IN:
        d_input1: Pointer to the first input device matrix.
        d_input2: Pointer to the second input device matrix.
        rows: Number of rows in the matrices.
        cols: Number of columns in the matrices.
    OUT:
        d_input1: Pointer to the first input device matrix after performing the subtraction. 
    */
    void m_sub(float* d_input1, const float* d_input2, int rows, int cols);

    /*
    Function to perform matrix multiplication of two matrices.
    IN:
        d_input1: Pointer to the first input device matrix (rows1 x cols1).
        d_input2: Pointer to the second input device matrix (rows2 x cols2).
        d_output: Pointer to the output device matrix (rows1 x cols2).
        rows1: Number of rows in the first matrix.
        cols1: Number of columns in the first matrix.
        rows2: Number of rows in the second matrix.
        cols2: Number of columns in the second matrix.
    OUT:
        d_output: Pointer to the output device matrix after performing the multiplication. 
    */
    void m_mul(const float* d_input1, const float* d_input2, float* d_output, int rows1, int cols1, int rows2, int cols2);

    /*
    Function to perform scalar multiplication of a matrix.
    IN:
        d_input: Pointer to the input device matrix.
        scalar: Scalar value to multiply with.
        rows: Number of rows in the matrix.
        cols: Number of columns in the matrix.
    OUT:
        d_input: Pointer to the input device matrix after performing the scalar multiplication.
    */
    void m_scalar_mul(float* d_input, float scalar, int rows, int cols);

    /*
    Function to perform matrix transposition.
    IN:
        d_input: Pointer to the input device matrix (rows x cols).
        d_output: Pointer to the output device matrix (cols x rows).
        rows: Number of rows in the input matrix.
        cols: Number of columns in the input matrix.
    OUT:
        d_output: Pointer to the output device matrix after performing the transposition.
    */
    void m_transpose(float* d_input, float* d_output, int rows, int cols);

    /*
    Function to perform element-wise Hadamard product of two matrices.
    IN:
        d_input1: Pointer to the first input device matrix.
        d_input2: Pointer to the second input device matrix.
        rows: Number of rows in the matrices.
        cols: Number of columns in the matrices.
    OUT:
        d_input1: Pointer to the first input device matrix after performing the Hadamard product.
    */
    void m_hadamard(float* d_input1, const float* d_input2, int rows, int cols);

    /*
    Function to apply the ReLU activation function element-wise.
    IN:
        d_input: Pointer to the input device matrix.
        rows: Number of rows in the matrix.
        cols: Number of columns in the matrix.
    OUT:
        d_input: Pointer to the input device matrix after applying the ReLU activation function.
    */
    void m_Relu(float* d_input, int rows, int cols);

    /*
    Function to compute the derivative of the ReLU activation function element-wise.
    IN:
        d_input: Pointer to the input device matrix.
        rows: Number of rows in the matrix.
        cols: Number of columns in the matrix.
    OUT:
        d_input: Pointer to the input device matrix after computing the derivative of the ReLU activation function.
    */
    void m_Relu_deriv(float* d_input, int rows, int cols);

    /*
    Function to convert an index matrix to a one-hot encoded matrix.
    IN:
        d_input: Pointer to the input device matrix containing indices (should be of size rows x 1).
        d_output: Pointer to the output device matrix for one-hot encoding (should be of size rows x cols).
        rows: Number of rows in the input matrix.
        cols: Number of columns in the output matrix (should match the number of unique indices).
    OUT:
        d_output: Pointer to the output device matrix after performing the one-hot encoding.
    */
    void m_index_to_one_hot(float* d_input, float* d_output, int rows, int cols);

    /*
    Function to apply the softmax function along the rows to a matrix.
    IN:
        d_input: Pointer to the input device matrix (should be of size rows x cols).
        rows: Number of rows in the matrix.
        cols: Number of columns in the matrix.
    OUT:
        d_input: Pointer to the input device matrix after applying the softmax function.
    */
    void m_softmax(float* d_input, int rows, int cols);

    /*
    Function to check for NaN or Inf values in a matrix.
    IN:
        d_input: Pointer to the input device matrix (should be of size rows x cols).
        rows: Number of rows in the matrix.
        cols: Number of columns in the matrix.
    OUT:
        Returns true if NaN or Inf values are found, false otherwise. 
    */
    bool m_check_nan_inf(const float* d_input, int rows, int cols);
}