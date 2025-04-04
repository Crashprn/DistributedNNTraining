#pragma once

#include "matrix_utils.hpp"
#include "nn_utils.hpp"

#include <tuple>
#include <iostream>
#include <tuple>
#include <omp.h>
#include <mpi.h>

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
    float learning_rate,
    int threads,
    int my_rank,
    int comm_size,
    int MASTER_RANK
);


void random_index(int* batch_indices, int size, int max_value, std::mt19937& gen);

