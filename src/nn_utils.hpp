#pragma once

#include "matrix_utils.hpp"
#include "c_matrix_utils.cuh"
#include "data_utils.hpp"

#include <tuple>
#include <math.h>
#include <omp.h>

/*
Function to calculate the cross-entropy loss between predicted and target values.
IN:
    y_hat: Pointer to the predicted values (output of the model).
    y: Pointer to the target values (ground truth).
    rows: Number of samples (rows) in the dataset.
    cols: Number of classes (columns) in the dataset.
OUT:
    Returns the average cross-entropy loss across all samples.
*/
float cross_entropy_loss(float* y_hat, float* y, int rows, int cols);

/*
Function to calculate the accuracy of predictions.
IN:
    y_hat: Pointer to the predicted class labels (output of the model).
    y: Pointer to the true class labels (ground truth).
    rows: Number of samples (rows) in the dataset.
OUT:
    Returns the accuracy as a float value, which is the proportion of correct predictions.
*/
float accuracy(const int* y_hat, const float* y, int rows);

/*
Function to initialize weights using He initialization.
IN:
    weight_mat: Pointer to the weight matrix to be initialized.
    rows: Number of rows in the weight matrix.
    cols: Number of columns in the weight matrix.
    gen: Random number generator for generating weights.
OUT:
    weight_mat is filled with weights initialized using He initialization.
*/
void m_he_weight_init(float* weight_mat, int rows, int cols, std::mt19937 &gen);

/*
Function to initialize weights using Xavier initialization.
IN:
    weight_mat: Pointer to the weight matrix to be initialized.
    rows: Number of rows in the weight matrix.
    cols: Number of columns in the weight matrix.
    gen: Random number generator for generating weights.
OUT:
    weight_mat is filled with weights initialized using Xavier initialization.
*/
void m_xavier_weight_init(float* weight_mat, int rows, int cols, std::mt19937 &gen);

/*
Function to initialize weights with a constant value.
IN:
    weight_mat: Pointer to the weight matrix to be initialized.
    rows: Number of rows in the weight matrix.
    cols: Number of columns in the weight matrix.
    value: The constant value to fill the weight matrix with.
OUT:
    weight_mat is filled with the specified constant value.
*/
void m_constant_weight_init(float* weight_mat, int rows, int cols, float value);

/*
Function to perform the forward pass of a neural network.
IN:
    weights: Tuple containing pointers to weight matrices for each layer.
    biases: Tuple containing pointers to bias vectors for each layer.
    input: Tuple containing the input data and its dimensions.
    z: Tuple to store the intermediate results (z values) for each layer.
    a: Tuple to store the activation values for each layer.
    dims: Tuple containing the dimensions of the layers (number of neurons).
OUT:
    The function computes the forward pass through the network, calculating the z values and activation values for each layer.
    The results are stored in the z and a tuples.
*/
void forward_pass_cpu(
    std::tuple<float*, float*, float*, float*> &weights, 
    std::tuple<float*, float*, float*, float*> &biases,
    std::tuple<float*, int, int> input,
    std::tuple<float*, float*, float*, float*> &z,
    std::tuple<float*, float*, float*> &a,
    std::tuple<int, int, int, int> &dims
);

/*
Function to perform the forward pass of a neural network on GPU.
IN:
    weights: Tuple containing pointers to weight matrices for each layer.
    biases: Tuple containing pointers to bias vectors for each layer.
    input: Tuple containing the input data and its dimensions.
    z: Tuple to store the intermediate results (z values) for each layer.
    a: Tuple to store the activation values for each layer.
    dims: Tuple containing the dimensions of the layers (number of neurons).
OUT:
    The function computes the forward pass through the network, calculating the z values and activation values for each layer.
    The results are stored in the z and a tuples.
*/
void forward_pass_gpu(
    std::tuple<float*, float*, float*, float*> &weights, 
    std::tuple<float*, float*, float*, float*> &biases,
    std::tuple<float*, int, int> input,
    std::tuple<float*, float*, float*, float*> &z,
    std::tuple<float*, float*, float*> &a,
    std::tuple<int, int, int, int> &dims
);

/*
Function to perform the backward pass of a neural network on CPU.
IN:
    weights_T: Tuple containing pointers to transposed weight matrices for each layer.
    weight_grads: Tuple containing pointers to weight gradients for each layer.
    bias_grads: Tuple containing pointers to bias gradients for each layer.
    input_T: Tuple containing the transposed input data and its dimensions.
    target: Tuple containing the target values and their dimensions.
    z: Tuple containing the z values for each layer.
    a_T: Tuple containing the transposed activation values for each layer.
    deltas: Tuple containing the deltas for each layer.
    dims: Tuple containing the dimensions of the layers (number of neurons).
OUT:
    The function computes the gradients for weights and biases based on the backward pass through the network.
    The results are stored in the weight_grads and bias_grads tuples.
*/
void backward_pass_cpu(
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

/*
Function to perform the backward pass of a neural network on GPU.
IN:
    weights_T: Tuple containing pointers to transposed weight matrices for each layer.
    weight_grads: Tuple containing pointers to weight gradients for each layer.
    bias_grads: Tuple containing pointers to bias gradients for each layer.
    input_T: Tuple containing the transposed input data and its dimensions.
    target: Tuple containing the target values and their dimensions.
    z: Tuple containing the z values for each layer.
    a_T: Tuple containing the transposed activation values for each layer.
    deltas: Tuple containing the deltas for each layer.
    dims: Tuple containing the dimensions of the layers (number of neurons).
OUT:
    The function computes the gradients for weights and biases based on the backward pass through the network.
    The results are stored in the weight_grads and bias_grads tuples.
*/
void backward_pass_gpu(
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