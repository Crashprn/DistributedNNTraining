#pragma once

#include <math.h>
#include <random>
#include <iostream>
#include <limits>

/*
Copy the contents of the source matrix to the destination matrix.
The source and destination matrices must have the same dimensions.

IN:
    src: The source matrix. rows x cols
    dest: The destination matrix. rows x cols
    rows: The number of rows in the matrix.
    cols: The number of columns in the matrix.
OUT:
    dest: The destination matrix. rows x cols
*/
void m_copy(const float* src, float* dest, int rows, int cols);

/*
Copy a row from the source matrix to the destination matrix.
The source and destination matrices must have the same number of columns.
IN:
    src: The source matrix. rows1 x cols
    dest: The destination matrix. unknown_row2 x cols
    src_row: The row in the source matrix to copy.
    dest_row: The row in the destination matrix to copy to.
    rows1: The number of rows in the src matrix.
    rows2: The number of rows in the dest matrix.
    cols: The number of columns in the src and dest matrix.
*/
void m_copy_row(const float* src, float* dest, int src_row, int dest_row, int rows1, int rows2, int cols);

/*
Sum the elements of a matrix along a given axis.
IN:
    input: The input matrix. rows x cols
    output: The output matrix. 1 x cols or rows x 1
    rows: The number of rows in the input matrix.
    cols: The number of columns in the input matrix.
    axis: The axis to sum along. 0 for columns, 1 for rows.
*/
void m_sum(const float* input, float* output, int rows, int cols, int axis);

/*
Find the index of the maximum element in a matrix along a given axis.ADJ_SETOFFSET
IN:
    input: The input matrix. rows x cols
    output: The output matrix. 1 x cols or rows x 1
    rows: The number of rows in the input matrix.
    cols: The number of columns in the input matrix.
    axis: The axis to find the maximum along. 0 for columns, 1 for rows.
*/
void m_argmax(const float* input, int* output, int rows, int cols, int axis);

/*
Add input2 matrix into input1 matrix element-wise.
IN:
    input1: The first input matrix. rows x cols
    input2: The second input matrix. rows x cols
    rows: The number of rows in the input matrices.
    cols: The number of columns in the input matrices.
*/
void m_add(float* input1, const float* input2, int rows, int cols);

/*
Add a vector to a matrix element-wise with broadcasting.
So the input matrix and the input vector must have the same number of rows or columns.
IN:
    input1_m: The input matrix. rows1 x cols1
    input2_v: The input vector. rows2 x cols2
    rows1: The number of rows in the input matrix.
    cols1: The number of columns in the input matrix.
    rows2: The number of rows in the input vector.
    cols2: The number of columns in the input vector.
*/
void m_add_v(float* input1_m, const float* input2_v, int rows1, int cols1, int rows2, int cols2);

/*
Subtract input2 matrix from input1 matrix element-wise.
IN:
    input1: The first input matrix. rows x cols
    input2: The second input matrix. rows x cols
    rows: The number of rows in the input matrices.
    cols: The number of columns in the input matrices.
*/
void m_sub(float* input1, const float* input2, int rows, int cols);

/*
Multiply two matrices together in the form A*B = C.
IN:
    input1: The first input matrix. rows1 x cols1
    input2: The second input matrix. cols1 x cols2
    output: The output matrix. rows1 x cols2
    rows1: The number of rows in the first input matrix.
    cols1: The number of columns in the first input matrix.
    rows2: The number of rows in the second input matrix.
    cols2: The number of columns in the second input matrix.
*/
void m_mul(const float* input1, const float* input2, float* output, int rows1, int cols1, int rows2, int cols2);

/*
Multiply a matrix by a scalar.
IN:
    input: The input matrix. rows x cols
    scalar: The scalar to multiply by.
    rows: The number of rows in the matrix.
    cols: The number of columns in the matrix.
*/
void m_scalar_mul(float* input, float scalar, int rows, int cols);

/*
Transpose a matrix.
IN:
    input: The input matrix. rows x cols
    output: The output matrix. cols x rows
    rows: The number of rows in the input matrix.
    cols: The number of columns in the input matrix.
*/
void m_transpose(const float* input, float* output, int rows, int cols);

/*
Hadamard product of two matrices into input1.
IN:
    input1: The first input matrix. rows x cols
    input2: The second input matrix. rows x cols
    rows: The number of rows in the input matrices.
    cols: The number of columns in the input matrices.
*/
void m_hadamard(float* input1, const float* input2, int rows, int cols);


