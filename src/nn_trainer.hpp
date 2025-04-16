#pragma once

#include "matrix_utils.hpp"
#include "c_matrix_utils.cuh"
#include "nn_utils.hpp"
#include "data_utils.hpp"

#include <tuple>
#include <iostream>
#include <tuple>
#include <omp.h>
#include <mpi.h>

/*
Function to perform the training loop on CPU.
IN:
    train_x: Pointer to the training input data.
    train_y: Pointer to the training target data.
    train_rows: Number of rows in the training dataset.
    train_x_cols: Number of columns in the training input data.
    train_y_cols: Number of columns in the training target data.
    output_layer_size: Size of the output layer.
    hidden_layer_size: Size of the hidden layer.
    epochs: Number of epochs for training.
    batch_size: Size of each batch for training.
    learning_rate: Learning rate for the optimizer.
    my_rank: Rank of the current process in MPI.
    comm_size: Total number of processes in MPI.
    MASTER_RANK: Rank of the master process (usually 0).
    save_model: Flag to indicate whether to save the model after training.
    save_dir: Directory where the model should be saved if save_model is true.
OUT:
    void
*/
void training_loop_cpu(
    float* train_x,
    float* train_y,
    int train_rows,
    int train_x_cols,
    int train_y_cols,
    int output_layer_size,
    int hidden_layer_size,
    int epochs,
    int batch_size,
    float learning_rate,
    int my_rank,
    int comm_size,
    int MASTER_RANK,
    bool save_model,
    const std::string& save_dir
);

/*
Function to perform the training loop on GPU.
IN:
    train_x: Pointer to the training input data.
    train_y: Pointer to the training target data.
    train_rows: Number of rows in the training dataset.
    train_x_cols: Number of columns in the training input data.
    train_y_cols: Number of columns in the training target data.
    output_layer_size: Size of the output layer.
    hidden_layer_size: Size of the hidden layer.
    epochs: Number of epochs for training.
    batch_size: Size of each batch for training.
    learning_rate: Learning rate for the optimizer.
    my_rank: Rank of the current process in MPI.
    comm_size: Total number of processes in MPI.
    MASTER_RANK: Rank of the master process (usually 0).
    save_model: Flag to indicate whether to save the model after training.
    save_dir: Directory where the model should be saved if save_model is true.
OUT:
    void
*/
void training_loop_gpu(
    float* train_x,
    float* train_y,
    int train_rows,
    int train_x_cols,
    int train_y_cols,
    int output_layer_size,
    int hidden_layer_size,
    int epochs,
    int batch_size,
    float learning_rate,
    int my_rank,
    int comm_size,
    int MASTER_RANK,
    bool save_model,
    const std::string& save_dir
);

/*
Function to generate random indices for batch sampling.
IN:
    batch_indices: Pointer to an array where the random indices will be stored.
    size: The number of indices to generate.
    max_value: The maximum value for the random indices (exclusive).
    gen: A reference to a random number generator.
OUT:
    batch_indices: Filled with random indices in the range [0, max_value).
*/
void random_indices(int* batch_indices, int size, int max_value, std::mt19937& gen);

