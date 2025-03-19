#include "matrix_utils.hpp"

void m_copy(const float* src, float* dest, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            dest[i * cols + j] = src[i * cols + j];
        }
    }
}

void m_copy_row(const float* src, float* dest, int src_row, int dest_row, int rows1, int rows2, int cols)
{
    if (src_row >= rows1)
    {
        std::cerr << "Matrix copy: Number row index " << src_row <<  " greater than src matrix rows " << rows1 << std::endl;
        return;
    }
    if (dest_row >= rows2)
    {
        std::cerr << "Matrix copy: Number row index " << dest_row <<  " greater than dest matrix rows " << rows2 << std::endl;
        return;
    }


    for (int j = 0; j < cols; ++j)
    {
        dest[dest_row * cols + j] = src[src_row * cols + j];
    }
}


void m_sum(const float* input, float* output, int rows, int cols, int axis)
{
    if (axis == 0)
    {
        // Sum along columns
        for (int j = 0; j < cols; ++j)
        {
            for (int i = 0; i < rows; ++i)
            {
                output[j] += input[i*cols + j];
            }
        }
    }
    else if (axis == 1)
    {
        // Sum along rows
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                output[i] += input[i*cols + j];
            }
        }
    }
}



void m_add(float* input1, const float* input2, int rows, int cols)
{   
    // Addition has no ordering so one just simply needs to loop through the elements and add them.
    for (int i = 0; i < rows * cols; ++i)
    {
        input1[i] = input1[i] + input2[i]; 
    }
}


void m_add_v(float* input1_m, const float* input2_v, int rows1, int cols1, int rows2, int cols2)
{
    if (rows1 == rows2)
    {
        // Loop through rows and add the vector to each column of the matrix
        for (int i = 0; i < rows1; ++i)
        {
            for (int j = 0; j < cols1; ++j)
            {
                input1_m[i*cols1 + j] = input1_m[i*cols1 + j] + input2_v[i];
            }
        }
    }
    else if (cols1 == cols2)
    {
        // Loop through the rows of the matrix and add the vector to each row of the matrix
        for (int i = 0; i < rows1; ++i)
        {
            for (int j = 0; j < cols1; ++j)
            {
                input1_m[i*cols1 + j] = input1_m[i*cols1 + j] + input2_v[j];
            }
        }
    }
    else
    {
        std::cerr << "Matrix add: Number of rows or columns in input1 and input2 do not match." << std::endl;
    }
}


void m_sub(float* input1, const float* input2, int rows, int cols)
{
    // Subtraction has no ordering so one just simply needs to loop through the elements and subtract them.
    for (int i = 0; i < rows * cols; ++i)
    {
        input1[i] = input1[i] - input2[i]; 
    }
}


void m_mul(const float* input1, const float* input2, float* output, int rows1, int cols1, int rows2, int cols2)
{
    if (cols1 != rows2)
    {
        std::cerr << "Matrix multiply: Number of columns in input1 must match number of rows in input2." << std::endl;
        return;
    }

    // Loop through rows in input1
    for (int i = 0; i < rows1; ++i)
    {
        // Loop through columns in input2
        for (int j = 0; j < cols2; ++j)
        {
            // Perform dot product of row input1[i,:] and column input2[:,j]
            float sum = 0.0;
            for (int k = 0; k < cols1; ++k)
            {
                sum += input1[i * cols1 + k] * input2[k * cols2 + j];
            }
            output[i * cols2 + j] = sum;
        }
    }
}


void m_scalar_mul(float* input, float scalar, int rows, int cols)
{
    // Scalar multiplication has no ordering so one just simply needs to loop through the elements and multiply them.
    for (int i = 0; i < rows * cols; ++i)
    {
        input[i] = input[i] * scalar;
    }
}


void m_transpose(const float* input, float* output, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}


void m_hadamard(float* input1, const float* input2, int rows, int cols)
{
    for (int i = 0; i < rows*cols; ++i)
    {
        input1[i] = input1[i] * input2[i];
    }
}


