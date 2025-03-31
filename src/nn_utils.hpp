#pragma once

#include "matrix_utils.hpp"
#include <tuple>
#include <math.h>
#include "data_utils.hpp"

/*
Convert a 1D array of class indices to a 2D one-hot encoded array.
INPUT:
    input: 1D array of class indices
    output: 2D one-hot encoded array
    rows: number of samples
    cols: number of classes
OUTPUT:
    void
*/
void m_index_to_one_hot(int* input, float* output, int rows, int cols);

/*
Take the softmax of a 2D array along the rows.
INPUT:
    input: 2D array of shape (rows, cols)
    rows: number of samples
    cols: number of classes
OUTPUT:
    void
*/
void m_softmax(float* input, int rows, int cols);

/*
Take the softmax of a 1D array.
INPUT:
    input: 1D array of shape (n,)
    n: number of classes
OUTPUT:
    void
*/
void v_softmax(float* input, int n);

/*
Calculate the cross-entropy loss between the predicted and true labels.
INPUT:
    y_hat: 2D array of predicted labels (softmax output)
    y: 1D array of true labels (floats containing class indices)
    rows: number of samples
    cols: number of classes
OUTPUT:
    loss: cross-entropy loss
*/
float cross_entropy_loss(float* y_hat, float* y, int rows, int cols);

/*
Calculate the accuracy of the model.
INPUT:
    y_hat: 1D array of predicted labels (class indices)
    y: 1D array of true labels (class indices)
    rows: number of samples
OUTPUT:
    accuracy: accuracy of the model
*/
float accuracy(const int* y_hat, const float* y, int rows);

/*
Initialize weights using He initialization.
INPUT:
    weight_mat: 2D array of weights
    rows: number of rows in the weight matrix
    cols: number of columns in the weight matrix
    gen: random number generator
OUTPUT:
    void
*/
void m_he_weight_init(float* weight_mat, int rows, int cols, std::mt19937 &gen);

/*
Initialize weights using Xavier initialization.
INPUT:
    weight_mat: 2D array of weights
    rows: number of rows in the weight matrix
    cols: number of columns in the weight matrix
    gen: random number generator
OUTPUT:
    void
*/
void m_xavier_weight_init(float* weight_mat, int rows, int cols, std::mt19937 &gen);

/*
Initialize weights with a constant value.
INPUT:
    weight_mat: 2D array of weights
    rows: number of rows in the weight matrix
    cols: number of columns in the weight matrix
    value: constant value to initialize the weights with
OUTPUT:
    void
*/
void m_constant_weight_init(float* weight_mat, int rows, int cols, float value);

/*
Apply Leaky Relu activation function.
INPUT:
    input: 2D array of shape (rows, cols)
    output: 2D array of shape (rows, cols)
    rows: number of samples
    cols: number of features
OUTPUT:
    void
*/
void m_Relu(float* input, float* output, int rows, int cols);

/*
Apply Leaky Relu activation function derivative.
INPUT:
    input: 2D array of shape (rows, cols)
    output: 2D array of shape (rows, cols)
    rows: number of samples
    cols: number of features
OUTPUT:
    void
*/
void m_Relu_deriv(float* input, float* output, int rows, int cols);

/*
Forward pass through the neural network. Saving intermediate values in z.
INPUT:
    weights: tuple of weight matrices for each layer
    biases: tuple of bias vectors for each layer
    input: input data (batch_size, input_feats)
    z: tuple of intermediate values for each layer
    dims: tuple of dimensions for each layer
OUTPUT:
    void
*/
void forward_pass(
    std::tuple<float*, float*, float*, float*> &weights, 
    std::tuple<float*, float*, float*, float*> &biases,
    std::tuple<float*, int, int> input,
    std::tuple<float*, float*, float*, float*> &z,
    std::tuple<int, int, int, int> &dims
);

/*
Backward pass through the neural network. Calculating gradients.
INPUT:
    weights_T: tuple of transposed weight matrices for each layer
    weight_grads: tuple of gradients for each weight matrix
    bias_grads: tuple of gradients for each bias vector
    input_T: input data (batch_size, input_feats)
    target: target labels (batch_size, output_layer_size)
    z: tuple of intermediate values for each layer
    dims: tuple of dimensions for each layer
OUTPUT:
    void
*/
void backward_pass(
    std::tuple<float*, float*, float*> &weights_T, 
    std::tuple<float*, float*, float*, float*> &weight_grads,
    std::tuple<float*, float*, float*, float*> &bias_grads,
    std::tuple<float*, int, int> input_T,
    std::tuple<float*, int, int> target,
    std::tuple<float*, float*, float*, float*> &z,
    std::tuple<int, int, int, int> &dims
);