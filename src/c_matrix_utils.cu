#include "c_matrix_utils.cuh"

/*
Copy the contents of the source matrix to the destination matrix.
The source and destination matrices must have the same dimensions.

IN:
    d_src: The source matrix. rows x cols
    d_dest: The destination matrix. rows x cols
    rows: The number of rows in the matrix.
    cols: The number of columns in the matrix.
OUT:
    d_dest: The destination matrix. rows x cols
*/
__global__
void m_copy_kernel(const float* d_src, float* d_dest, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols)
    {
        int idx = row * cols + col;
        d_dest[idx] = d_src[idx];
    }
}

/*
Copy a row from the source matrix to the destination matrix.
The source and destination matrices must have the same number of columns.
IN:
    d_src: The source matrix. rows1 x cols
    d_dest: The destination matrix. unknown_row2 x cols
    src_row: The row in the source matrix to copy.
    dest_row: The row in the destination matrix to copy to.
    rows1: The number of rows in the src matrix.
    rows2: The number of rows in the dest matrix.
    cols: The number of columns in the src and dest matrix.
OUT:
    d_dest: The destination matrix with the copied row. rows2 x cols
*/
__global__
void m_copy_row_kernel(const float* d_src, float* d_dest, int src_row, int dest_row, int rows1, int rows2, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cols) {
        d_dest[dest_row * cols + idx] = d_src[src_row * cols + idx];
    }
}

/*
Sum the elements of a matrix along a given axis.
IN:
    d_input: The input matrix. rows x cols
    d_output: The output matrix. 1 x cols or rows x 1
    rows: The number of rows in the input matrix.
    cols: The number of columns in the input matrix.
    axis: The axis to sum along. 0 for columns, 1 for rows.
OUT:
    d_output: The output matrix containing the sum along the specified axis.
    If axis is 0, d_output will be 1 x cols, containing the sum of each column.
    If axis is 1, d_output will be rows x 1, containing the sum of each row.
*/
__global__
void m_sum_kernel(const float* d_input, float* d_output, int rows, int cols, int axis)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (axis == 0) // Sum along columns
    { 
        if (idx < cols) {
            for (int row = 0; row < rows; row++) {
                sum += d_input[row * cols + idx];
            }
            d_output[idx] = sum;
        }
    }
    else if (axis == 1) // Sum along rows
    {         
        if (idx < rows) {
            for (int col = 0; col < cols; col++) {
                sum += d_input[idx * cols + col];
            }
            d_output[idx] = sum;
        }
    }
}

/*
Find the index of the maximum element in a matrix along a given axis.ADJ_SETOFFSET
IN:
    d_input: The input matrix. rows x cols
    d_output: The output matrix. 1 x cols or rows x 1
    rows: The number of rows in the input matrix.
    cols: The number of columns in the input matrix.
    axis: The axis to find the maximum along. 0 for columns, 1 for rows.
OUT:
    d_output: The output matrix containing the index of the maximum element along the specified axis.
    If axis is 0, d_output will be 1 x cols, containing the index of the maximum element in each column.
    If axis is 1, d_output will be rows x 1, containing the index of the maximum element in each row.
*/
__global__
void m_argmax_kernel(const float* d_input, int* d_output, int rows, int cols, int axis)
{

    int max_index = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float max_value;

    if (axis == 0)
    {
        if (idx < cols)
        {
            max_value = d_input[0 * cols + idx]; // Initialize with the first row's value
            for (int row = 1; row < rows; row++)
            {
                float value = d_input[row * cols + idx];
                if (value > max_value)
                {
                    max_value = value;
                    max_index = row;
                }
            }
            d_output[idx] = max_index; // Store the index of the maximum element
        }
    } else if (axis == 1)
    {
        if (idx < rows)
        {
            max_value = d_input[idx * cols + 0]; // Initialize with the first column's value
            for (int col = 1; col < cols; col++)
            {
                float value = d_input[idx * cols + col];
                if (value > max_value)
                {
                    max_value = value;
                    max_index = col;
                }
            }
            d_output[idx] = max_index; // Store the index of the maximum element
        }
    }
    else
    {
        // Invalid axis, do nothing
        return;
    }
}

/*
Add input2 matrix into input1 matrix element-wise.
IN:
    d_input1: The first input matrix. rows x cols
    d_input2: The second input matrix. rows x cols
    rows: The number of rows in the input matrices.
    cols: The number of columns in the input matrices.
OUT:
    d_input1: The output matrix with addition applied. rows x cols
*/
__global__
void m_add_kernel(float* d_input1, const float* d_input2, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        d_input1[idx] += d_input2[idx];
    }
}

/*
Add a vector to a matrix element-wise with broadcasting.
So the input matrix and the input vector must have the same number of rows or columns.
IN:
    d_input1_m: The input matrix. rows1 x cols1
    d_input2_v: The input vector. rows2 x cols2
    rows1: The number of rows in the input matrix.
    cols1: The number of columns in the input matrix.
    rows2: The number of rows in the input vector.
    cols2: The number of columns in the input vector.
OUT:
    d_input1_m: The output matrix with addition applied. rows1 x cols1
*/
__global__
void m_add_v_kernel(float* d_input1_m, const float* d_input2_v, int rows, int cols, int axis)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (axis == 1) // Broadcast along columns
    {
        if (row < rows && col < cols)
        {
            d_input1_m[row * cols + col] += d_input2_v[row];
        }
    }
    else // Broadcast along rows
    {
        if (row < rows && col < cols)
        {
            d_input1_m[row * cols + col] += d_input2_v[col];
        }
    }
}

/*
Subtract input2 matrix from input1 matrix element-wise.
IN:
    d_input1: The first input matrix. rows x cols
    d_input2: The second input matrix. rows x cols
    rows: The number of rows in the input matrices.
    cols: The number of columns in the input matrices.
OUT:
    d_input1: The output matrix with subtraction applied. rows x cols
*/
__global__
void m_sub_kernel(float* d_input1, const float* d_input2, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols)
    {
        int idx = row * cols + col;
        d_input1[idx] -= d_input2[idx];
    }
}

/*
Multiply two matrices together in the form A*B = C.
IN:
    d_input1: The first input matrix. rows1 x cols1
    d_input2: The second input matrix. cols1 x cols2
    d_output: The output matrix. rows1 x cols2
    rows1: The number of rows in the first input matrix.
    cols1: The number of columns in the first input matrix.
    rows2: The number of rows in the second input matrix.
    cols2: The number of columns in the second input matrix.
OUT:
    d_output: The output matrix. rows1 x cols2
*/
__global__
void m_mul_kernel(const float* d_input1, const float* d_input2, float* d_output, int rows1, int cols1, int rows2, int cols2)
{
    extern __shared__ float m_tile[];
    float* n_tile = (m_tile + blockDim.x * blockDim.y);

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x, ty = threadIdx.y;
    int tile_size = blockDim.x;

    
    float sum = 0.0f;
    for (int q = 0; q < ceil(1.0*cols1 / tile_size); q++)
    {
        // Load the M and N tiles into shared memory
        int offset = q * tile_size;

        int m_col = offset + tx; // Column index in M tile
        int m_row = row; // Row index in M tile

        int n_col = col; // Column index in N tile
        int n_row = offset + ty; // Row index in N tile

        m_tile[ty * tile_size + tx] = (m_col < cols1 && m_row < rows1) ? d_input1[m_row * cols1 + m_col]: 0.0f; // Load M tile
        n_tile[ty * tile_size + tx] = (n_row < rows2 && n_col < cols2) ? d_input2[n_row * cols2 + n_col]: 0.0f; // Load N tile

        __syncthreads(); // Ensure all threads have loaded their data
        // Perform the multiplication and accumulate sum

        for (int k = 0; k < tile_size; k++)
        {
            sum += m_tile[ty * tile_size + k] * n_tile[k * tile_size + tx];
        }
        __syncthreads(); // Ensure all threads have completed their calculations
    }

    if (row < rows1 && col < cols2)
    {
        // Write the result to the output matrix
        d_output[row * cols2 + col] = sum;
    }
}

/*
Multiply a matrix by a scalar.
IN:
    d_input: The input matrix. rows x cols
    scalar: The scalar to multiply by.
    rows: The number of rows in the matrix.
    cols: The number of columns in the matrix.
OUT:
    d_input: The output matrix with scalar multiplication applied. rows x cols
*/
__global__
void m_scalar_mul_kernel(float* d_input, float d_scalar, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        d_input[idx] *= d_scalar;
    }
}

/*
Transpose a matrix.
IN:
    d_input: The input matrix. rows x cols
    d_output: The output matrix. cols x rows
    rows: The number of rows in the input matrix.
    cols: The number of columns in the input matrix.
OUT:
    d_output: The output matrix. cols x rows
*/
__global__
void m_transpose_kernel(float* d_input, float* d_output, int rows, int cols)
{
    extern __shared__ float tile[];
    int tile_size = blockDim.x;
    
    // Calculate row and column indices
    int col = blockIdx.x * tile_size + threadIdx.x;
    int row = blockIdx.y * tile_size + threadIdx.y;
    int ty = threadIdx.y, tx = threadIdx.x;

    // Check if thread is in-bounds in the original matrix
    if (row < rows && col < cols)
    {
        // load data into shared memory
        int r_index = row * cols + col;
        tile[ty * tile_size + tx] = d_input[r_index];
    }

    __syncthreads(); // Ensure all threads have loaded their data

    // Define transposed indices
    int t_col = blockIdx.y * tile_size + threadIdx.x; 
    int t_row = blockIdx.x * tile_size + threadIdx.y; 

    // Transposed matrix is cols x rows, so we need to check bounds
    // when writing to the transposed matrix contiguously.
    if (t_row < cols && t_col < rows)
    {
        int t_index = t_row * rows + t_col;

        // Write from shared memory to global memory in a contiguous manner
        // Since tile is populated in [row][col], we need to swap the indices
        // and write in [col][row] order 
        d_output[t_index] = tile[tx * tile_size + ty];
    }
}

/*
Hadamard product of two matrices into input1.
IN:
    d_input1: The first input matrix. rows x cols
    d_input2: The second input matrix. rows x cols
    rows: The number of rows in the input matrices.
    cols: The number of columns in the input matrices.
OUT:
    d_input1: The output matrix with Hadamard product applied. rows x cols
*/
__global__
void m_hadamard_kernel(float* d_input1, const float* d_input2, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        d_input1[idx] = d_input1[idx] * d_input2[idx];
    }
}

/*
Apply the ReLU activation function to each element of the input matrix.
IN:
    d_input: The input matrix. rows x cols
    rows: The number of rows in the input matrix.
    cols: The number of columns in the input matrix.
OUT:
    d_input: The output matrix with ReLU applied. rows x cols
*/
__global__
void m_Relu_kernel(float* d_input, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        // Apply ReLU: max(0, input)
        if (d_input[idx] < 0) 
        {
            d_input[idx] = 0.1* d_input[idx]; // Leaky ReLU: 0.1 * input if input < 0
        }
    }
}
/*
Calculate the derivative of the ReLU activation function.
IN:
    d_input: The input matrix. rows x cols
    rows: The number of rows in the input matrix.
    cols: The number of columns in the input matrix.
*/
__global__
void m_Relu_deriv_kernel(float* d_input, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        // Apply derivative of ReLU: 1 if input > 0, else 0
        if (d_input[idx] < 0.0) d_input[idx] = 0.1f; // Derivative is 0 for negative inputs
        else d_input[idx] = 1.0f; // Derivative is 1 for positive inputs
    }
}

/*
Convert an index matrix to a one-hot encoded matrix.
IN:
    d_input: The input matrix containing indices. rows x 1
    d_output: The output one-hot encoded matrix. rows x cols
    rows: The number of rows in the input matrix.
    cols: The number of columns in the output matrix (number of classes).
OUT:
    d_output: The output one-hot encoded matrix. rows x cols
*/
__global__
void m_index_to_one_hot_kernel(float* d_input, float* d_output, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows)
    {
        int index = static_cast<int>(d_input[row]); // Get the index from the input
        for (int col = 0; col < cols; col++)
        {
            if (index == col) d_output[row * cols + col] = 1.0f; // Set the corresponding one-hot position to 1
            else d_output[row * cols + col] = 0.0f; // Initialize the output to zero
        }
    }

}

/*
Apply the softmax activation function to each row of the input matrix.
IN:
    input: The input matrix. rows x cols
    rows: The number of rows in the input matrix.
    cols: The number of columns in the input matrix.
OUT:
    input: The output matrix with softmax applied. rows x cols
*/
__global__
void m_softmax_kernel(float* d_input, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows)
    {
        float sum = 0.0f;
        // First pass: calculate the sum of exponentials
        for (int col = 0; col < cols; col++)
        {
            sum += exp(d_input[row * cols + col]);
            if (isnan(sum) || isinf(sum)) {
                // Handle overflow or underflow
                printf("Warning: Overflow/Underflow detected in softmax computation at row %d, col: %d, value: %f\n", row, col, d_input[row * cols + col]);
                sum = 1.0f; // Set to 1 to avoid division by zero
                break;
            }
        }

        // Second pass: calculate softmax
        for (int col = 0; col < cols; col++)
        {
            d_input[row * cols + col] = exp(d_input[row * cols + col]) / sum;
        }
    }
}

/*
Check if the input matrix contains NaN or Inf values.
IN:
    d_input: The input matrix. rows x cols
    d_nan_inf: The output matrix to store NaN or Inf values. rows x cols
    rows: The number of rows in the input matrix.
    cols: The number of columns in the input matrix.
OUT:
    d_nan_inf: The output matrix containing NaN or Inf values, if any.
*/
__global__
void check_nan_inf_kernel(const float* d_input, float* d_nan_inf, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
    {
        float value = d_input[idx];
        // Check for NaN or Inf
        if (isnan(value) || isinf(value))
        {
            d_nan_inf[idx] = value; // Set flag to indicate NaN or Inf found
        }
    }
}


namespace cuda_matrix
{
    // Define the grid size for CUDA kernels
    int BLOCK_SIZE_X = 16;
    int BLOCK_SIZE_Y = 16;  
    
    void cuda_error(cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }    

    void cuda_synchronize()
    {
        // Synchronize the device to ensure all operations are complete
        cuda_error(cudaDeviceSynchronize());
    }

    int cuda_device_count()
    {
        int device_count;
        cuda_error(cudaGetDeviceCount(&device_count));
        return device_count;
    }

    void cuda_config(int block_size_x, int block_size_y, int device_id)
    {
        // Set the device to the specified device ID
        cudaSetDevice(device_id);
        cuda_error(cudaGetLastError());

        // Set the block sizes
        BLOCK_SIZE_X = block_size_x;
        BLOCK_SIZE_Y = block_size_y;
        // Ensure the block sizes are positive
        if (BLOCK_SIZE_X <= 0 || BLOCK_SIZE_Y <= 0) {
            fprintf(stderr, "Error: Block sizes must be positive integers.\n");
            exit(EXIT_FAILURE);
        }
    }

    // Helper function to copy data on device
    void m_copy(const float* d_src, float* d_dest, int rows, int cols)
    {
        // Define block and grid sizes
        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        int grid_x = static_cast<unsigned int>(ceil(1.0 * cols / blockDim.x));
        int grid_y = static_cast<unsigned int>(ceil(1.0 * rows / blockDim.y));  
        dim3 gridDim(grid_x, grid_y);

        // Launch the kernel
        m_copy_kernel<<<gridDim, blockDim>>>(d_src, d_dest, rows, cols);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to copy a row from one matrix to another on device
    void m_copy_row(const float* d_src, float* d_dest, int src_row, int dest_row, int rows1, int rows2, int cols)
    {
        // Define block and grid sizes
        dim3 blockDim(BLOCK_SIZE_X*BLOCK_SIZE_Y);
        int grid_x = static_cast<unsigned int>(ceil(1.0 * cols / blockDim.x));
        int grid_y = 1; 
        dim3 gridDim(grid_x, grid_y);

        // Launch the kernel
        m_copy_row_kernel<<<gridDim, blockDim>>>(d_src, d_dest, src_row, dest_row, rows1, rows2, cols);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to sum the elements of a matrix along a given axis
    void m_sum(const float* d_input, float* d_output, int rows, int cols, int axis)
    {
        // Define block and grid sizes
        dim3 blockDim(BLOCK_SIZE_X*BLOCK_SIZE_Y);
        int grid_x, grid_y;

        if (axis == 0) // Sum along columns
            grid_x = static_cast<unsigned int>(ceil(1.0 * cols / blockDim.x)), grid_y = 1; // One row for each column
        else if (axis == 1) // Sum along rows
            grid_x = 1, grid_y = static_cast<unsigned int>(ceil(1.0 * rows / blockDim.x)); // One column for each row
        else
            return; // Invalid axis
    

        dim3 gridDim(grid_x, grid_y); // Calculate grid size based on rows and cols

        // Launch the kernel
        m_sum_kernel<<<gridDim, blockDim>>>(d_input, d_output, rows, cols, axis);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to argmax the elements of a matrix along a given axis
    void m_argmax(const float* d_input, int* d_output, int rows, int cols, int axis)
    {
        // Define block and grid sizes
        dim3 blockDim(BLOCK_SIZE_X*BLOCK_SIZE_Y);
        int grid_x, grid_y;

        if (axis == 0) // Max along columns
            grid_x = static_cast<unsigned int>(ceil(1.0 * cols / blockDim.x)), grid_y = 1; // One row for each column
        else if (axis == 1) // Max along rows
            grid_x = 1, grid_y = static_cast<unsigned int>(ceil(1.0 * rows / blockDim.x)); // One column for each row
        else
            return; // Invalid axis
    
        dim3 gridDim(grid_x, grid_y); // Calculate grid size based on rows and cols

        // Launch the kernel
        m_argmax_kernel<<<gridDim, blockDim>>>(d_input, d_output, rows, cols, axis);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to add two matrices element-wise
    void m_add(float* d_input1, const float* d_input2, int rows, int cols)
    {
        // Define block and grid sizes
        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        int grid_x = static_cast<unsigned int>(ceil(1.0 * cols / blockDim.x));
        int grid_y = static_cast<unsigned int>(ceil(1.0 * rows / blockDim.y));
        dim3 gridDim(grid_x, grid_y); 

        // Launch the kernel
        m_add_kernel<<<gridDim, blockDim>>>(d_input1, d_input2, rows, cols);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to add a vector to a matrix element-wise with broadcasting
    void m_add_v(float* d_input1_m, const float* d_input2_v, int rows1, int cols1, int rows2, int cols2)
    {
        // Defining block and grid sizes
        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        int grid_x = static_cast<unsigned int>(ceil(1.0 * cols1 / blockDim.x));
        int grid_y = static_cast<unsigned int>(ceil(1.0 * rows1 / blockDim.y));
        dim3 gridDim(grid_x, grid_y);

        // Identify axis to broadcast along
        if (rows1 == rows2)
        {
            // Broadcast along columns
            m_add_v_kernel<<<gridDim, blockDim>>>(d_input1_m, d_input2_v, rows1, cols1, 1);
        }
        else if (cols1 == cols2)
        {
            // Broadcast along rows
            m_add_v_kernel<<<gridDim, blockDim>>>(d_input1_m, d_input2_v, rows1, cols1, 0);
        }
        else
        {
            // Invalid dimensions for broadcasting
            fprintf(stderr, "Error: Incompatible dimensions for broadcasting.\n");
            return;
        }
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to subtract two matrices element-wise
    void m_sub(float* d_input1, const float* d_input2, int rows, int cols)
    {
        // Define block and grid sizes
        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        int grid_x = static_cast<unsigned int>(ceil(1.0 * cols / blockDim.x));
        int grid_y = static_cast<unsigned int>(ceil(1.0 * rows / blockDim.y));
        dim3 gridDim(grid_x, grid_y);

        // Launch the kernel
        m_sub_kernel<<<gridDim, blockDim>>>(d_input1, d_input2, rows, cols);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to multiply two matrices together
    void m_mul(const float* d_input1, const float* d_input2, float* d_output, int rows1, int cols1, int rows2, int cols2)
    {
        // Ensure the dimensions are compatible for matrix multiplication
        if (cols1 != rows2) {
            fprintf(stderr, "Error: Incompatible dimensions for matrix multiplication (%d != %d).\n", cols1, rows2);
            return;
        }

        // Define block and grid sizes
        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        int grid_x = static_cast<unsigned int>(ceil(1.0 * cols2 / blockDim.x));
        int grid_y = static_cast<unsigned int>(ceil(1.0 * rows1 / blockDim.y));
        dim3 gridDim(grid_x, grid_y);

        // Launch the kernel
        m_mul_kernel<<<gridDim, blockDim, 2 * blockDim.x * blockDim.y * sizeof(float)>>>(d_input1, d_input2, d_output, rows1, cols1, rows2, cols2);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to multiply a matrix by a scalar
    void m_scalar_mul(float* d_input, float d_scalar, int rows, int cols)
    {
        // Define block and grid sizes
        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        int grid_x = static_cast<unsigned int>(ceil(1.0 * cols / blockDim.x));
        int grid_y = static_cast<unsigned int>(ceil(1.0 * rows / blockDim.y));
        dim3 gridDim(grid_x, grid_y);

        // Launch the kernel
        m_scalar_mul_kernel<<<gridDim, blockDim>>>(d_input, d_scalar, rows, cols);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to transpose a matrix
    void m_transpose(float* d_input, float* d_output, int rows, int cols)
    {
        // Define block and grid sizes
        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        int grid_x = static_cast<unsigned int>(ceil(1.0 * cols / blockDim.x));
        int grid_y = static_cast<unsigned int>(ceil(1.0 * rows / blockDim.y));
        dim3 gridDim(grid_x, grid_y);

        // Launch the kernel
        m_transpose_kernel<<<gridDim, blockDim, blockDim.x * blockDim.y * sizeof(float)>>>(d_input, d_output, rows, cols);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to perform Hadamard product of two matrices
    void m_hadamard(float* d_input1, const float* d_input2, int rows, int cols)
    {
        // Define block and grid sizes
        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        int grid_x = static_cast<unsigned int>(ceil(1.0 * cols / blockDim.x));
        int grid_y = static_cast<unsigned int>(ceil(1.0 * rows / blockDim.y));
        dim3 gridDim(grid_x, grid_y); 

        // Launch the kernel
        m_hadamard_kernel<<<gridDim, blockDim>>>(d_input1, d_input2, rows, cols);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to apply the ReLU activation function
    void m_Relu(float* d_input, int rows, int cols)
    {
        // Define block and grid sizes
        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        int grid_x = static_cast<unsigned int>(ceil(1.0 * cols / blockDim.x));
        int grid_y = static_cast<unsigned int>(ceil(1.0 * rows / blockDim.y)); 
        dim3 gridDim(grid_x, grid_y);

        // Launch the kernel
        m_Relu_kernel<<<gridDim, blockDim>>>(d_input, rows, cols);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to apply the derivative ReLU activation function
    void m_Relu_deriv(float* input, int rows, int cols)
    {
        // Define block and grid sizes
        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        int grid_x = static_cast<unsigned int>(ceil(1.0 * cols / blockDim.x));
        int grid_y = static_cast<unsigned int>(ceil(1.0 * rows / blockDim.y)); 
        dim3 gridDim(grid_x, grid_y); 

        // Launch the kernel
        m_Relu_deriv_kernel<<<gridDim, blockDim>>>(input, rows, cols);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to convert index to one-hot encoding
    void m_index_to_one_hot(float* input, float* output, int rows, int cols)
    {
        // Define block and grid sizes
        dim3 blockDim(1, BLOCK_SIZE_X*BLOCK_SIZE_Y);
        int grid_x = 1; // One block for each row
        int grid_y = static_cast<unsigned int>(ceil(1.0 * rows / blockDim.y)); // Calculate grid size based on rows

        dim3 gridDim(grid_x, grid_y);

        m_index_to_one_hot_kernel<<<gridDim, blockDim>>>(input, output, rows, cols);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    // Helper function to apply the softmax activation function
    void m_softmax(float* input, int rows, int cols)
    {
        // Define block and grid sizes
        dim3 blockDim(1, BLOCK_SIZE_X*BLOCK_SIZE_Y);
        int grid_x = 1; // One block for each row
        int grid_y = static_cast<unsigned int>(ceil(1.0 * rows / blockDim.y)); // Calculate grid size based on rows

        dim3 gridDim(grid_x, grid_y); // Calculate grid size based on rows

        // Launch the kernel
        m_softmax_kernel<<<gridDim, blockDim>>>(input, rows, cols);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch
    }

    bool m_check_nan_inf(const float* d_input, int rows, int cols)
    {
        // Check for NaN or Inf in the matrix
        float* d_nan_inf;
        cudaMalloc((void**)&d_nan_inf, rows*cols*sizeof(float));
        cudaMemset(d_nan_inf, 0, rows*cols*sizeof(float));

        dim3 blockDim(BLOCK_SIZE_X*BLOCK_SIZE_Y);
        int grid_x = static_cast<unsigned int>(ceil(1.0 * (rows * cols) / blockDim.x));
        dim3 gridDim(grid_x, 1); // One block for each element

        check_nan_inf_kernel<<<gridDim, blockDim>>>(d_input, d_nan_inf, rows, cols);
        cuda_error(cudaGetLastError()); // Check for any errors during kernel launch

        float* h_nan_inf = new float[rows * cols];

        cuda_error(cudaMemcpy(h_nan_inf, d_nan_inf, rows*cols*sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < rows * cols; i++)
        {
            if (h_nan_inf[i] != 0.0f)
            {
                fprintf(stderr, "Warning: Found NaN or Inf in the matrix at index %d, value: %f\n", i, h_nan_inf[i]);
                return true; // If we found NaN or Inf, return true
            } 
        }

        cudaFree(d_nan_inf);
        delete[] h_nan_inf; // Free the host memory
        return false; // No NaN or Inf found
    }

}
