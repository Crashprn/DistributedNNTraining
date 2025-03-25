#pragma once

#include "matrix_utils.hpp"
#include "data_utils.hpp"
#include <tuple>
#include <math.h>
#include <omp.h>

void m_index_to_one_hot(float* input, float* output, int rows, int cols);
void m_softmax(float* input, int rows, int cols);
void v_softmax(float* input, int n);
float cross_entropy_loss(float* y_hat, float* y, int rows, int cols);
float accuracy(const int* y_hat, const float* y, int rows);
void m_he_weight_init(float* weight_mat, int rows, int cols, std::mt19937 &gen);
void m_xavier_weight_init(float* weight_mat, int rows, int cols, std::mt19937 &gen);
void m_constant_weight_init(float* weight_mat, int rows, int cols, float value);
void m_Relu(float* input, int rows, int cols);
void m_Relu_deriv(float* input, int rows, int cols);

void forward_pass(
    std::tuple<float*, float*, float*, float*> &weights, 
    std::tuple<float*, float*, float*, float*> &biases,
    std::tuple<float*, int, int> input,
    std::tuple<float*, float*, float*, float*> &z,
    std::tuple<float*, float*, float*> &a,
    std::tuple<int, int, int, int> &dims
);

void backward_pass(
    std::tuple<float*, float*, float*> &weights_T, 
    std::tuple<float*, float*, float*, float*> &weight_grads,
    std::tuple<float*, float*, float*, float*> &bias_grads,
    std::tuple<float*, int, int> input_T,
    std::tuple<float*, int, int> target,
    std::tuple<float*, float*, float*, float*> &z,
    std::tuple<float*, float*, float*> &a_T,
    std::tuple<float*, float*, float*> &deltas,
    std::tuple<int, int, int, int> &dims
);