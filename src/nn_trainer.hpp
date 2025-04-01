#pragma once

#include "matrix_utils.hpp"
#include "nn_utils.hpp"

#include <tuple>
#include <iostream>
#include <tuple>

/*
Function to train a neural network using the training data
INPUTS:
    train_x: Pointer to the training data
    train_y: Pointer to the training labels
    train_rows: Number of rows in the training data
    train_x_cols: Number of columns in the training data
    train_y_cols: Number of columns in the training labels
    output_layer_size: Size of the output layer
    hidden_layer_size: Size of the hidden layer
    epochs: Number of epochs to train for
    batch_size: Size of each batch
    learning_rate: Learning rate for the optimizer
*/
void training_loop(
    float* train_x,
    float* train_y,
    int train_rows,
    int train_x_cols,
    int train_y_cols,
    int output_layer_size,
    int hidden_layer_size,
    int epochs,
    int batch_size,
    float learning_rate
);


/*
Function to randomly select an index from the training data
INPUTS:
    max_value: The maximum value for the random index
    gen: The random number generator
OUTPUTS:
    Returns a random index in the range of [0, max_value)
*/
int random_index(int max_value, std::mt19937& gen);

