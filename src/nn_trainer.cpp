#include "nn_trainer.hpp"

void training_loop(
    float* train_x,
    float* train_y,
    int train_rows,
    int train_x_cols,
    int train_y_cols,
    int output_layer_size,
    int hidden_layer_size,
    int epochs,
    int glob_batch_size,
    float learning_rate,
    int threads,
    int my_rank,
    int comm_size,
    int MASTER_RANK
)
{
    int input_layer_size = train_x_cols;

    // Creating weights and biases
    float* w1 = new float[input_layer_size * hidden_layer_size]; // (input_layer_size, hidden_layer_size)
    float* w2 = new float[hidden_layer_size * hidden_layer_size]; // (hidden_layer_size, hidden_layer_size)
    float* w3 = new float[hidden_layer_size * hidden_layer_size]; // (hidden_layer_size, hidden_layer_size)
    float* w4 = new float[hidden_layer_size * output_layer_size]; // (hidden_layer_size, output_layer_size

    float* w1_T = new float[hidden_layer_size * input_layer_size]; // (hidden_layer_size, input_layer_size)
    float* w2_T = new float[hidden_layer_size * hidden_layer_size]; // (hidden_layer_size, hidden_layer_size)
    float* w3_T = new float[hidden_layer_size * hidden_layer_size]; // (hidden_layer_size, hidden_layer_size)
    float* w4_T = new float[output_layer_size * hidden_layer_size]; // (output_layer_size, hidden_layer_size)

    float* b1 = new float[hidden_layer_size];
    float* b2 = new float[hidden_layer_size];
    float* b3 = new float[hidden_layer_size];
    float* b4 = new float[output_layer_size];

    // Initializing gradients
    float* dw1 = new float[input_layer_size * hidden_layer_size];
    float* dw2 = new float[hidden_layer_size * hidden_layer_size];
    float* dw3 = new float[hidden_layer_size * hidden_layer_size];
    float* dw4 = new float[hidden_layer_size * output_layer_size];

    float* db1 = new float[hidden_layer_size];
    float* db2 = new float[hidden_layer_size];
    float* db3 = new float[hidden_layer_size];
    float* db4 = new float[output_layer_size];
    
    // Calculating batch size for each process
    float batch_size_f = static_cast<float>(glob_batch_size);
    int my_batch_size = static_cast<int>(floor(batch_size_f / static_cast<float>(comm_size)));
    if (my_rank == comm_size - 1)
    {
        my_batch_size += glob_batch_size % comm_size;
    }

    // Creating global gradients for MASTER_RANK
    float* glob_dw1, *glob_dw2, *glob_dw3, *glob_dw4;
    float* glob_db1, *glob_db2, *glob_db3, *glob_db4;

    if (my_rank == MASTER_RANK)
    {
        glob_dw1 = new float[input_layer_size * hidden_layer_size];
        glob_dw2 = new float[hidden_layer_size * hidden_layer_size];
        glob_dw3 = new float[hidden_layer_size * hidden_layer_size];
        glob_dw4 = new float[hidden_layer_size * output_layer_size];

        glob_db1 = new float[hidden_layer_size];
        glob_db2 = new float[hidden_layer_size];
        glob_db3 = new float[hidden_layer_size];
        glob_db4 = new float[output_layer_size];
    }

    // Initializing z values for each layer
    float* z1 = new float[my_batch_size * hidden_layer_size];
    float* z2 = new float[my_batch_size * hidden_layer_size];
    float* z3 = new float[my_batch_size * hidden_layer_size];
    float* z4 = new float[my_batch_size * output_layer_size];

    // Initializing an output matrix and class vector
    float* y_hat = new float[my_batch_size * output_layer_size];
    int* y_class = new int[my_batch_size * 1];

    // Creating matrices for batch of data
    float* my_batch_x = new float[my_batch_size * input_layer_size]; // batch_size x input_layer_size
    float* my_batch_x_T = new float[input_layer_size * my_batch_size]; // input_layer_size x batch_size
    float* my_batch_y = new float[my_batch_size * train_y_cols]; // batch_size x 1
    int* my_batch_indices = new int[my_batch_size];

    std::random_device rd;
    std::mt19937 normal(43);
    std::mt19937 unif(43);

    if (my_rank == MASTER_RANK)
    {
        // Initializing weights and biases
        

        std::cout << "Initializing weights..." << std::endl;
        m_he_weight_init(w1, input_layer_size, hidden_layer_size,  normal);
        m_he_weight_init(w2, hidden_layer_size, hidden_layer_size, normal);
        m_he_weight_init(w3, hidden_layer_size, hidden_layer_size, normal);
        m_he_weight_init(w4, hidden_layer_size, output_layer_size, normal);

        std::cout << "Initializing biases..." << std::endl;

        //float bias_init = 0.1f;
        m_xavier_weight_init(b1, hidden_layer_size, 1, unif);
        m_xavier_weight_init(b2, hidden_layer_size, 1, unif);
        m_xavier_weight_init(b3, hidden_layer_size, 1, unif);
        m_xavier_weight_init(b4, output_layer_size, 1, unif);

        std::cout << "Weights and biases initialized." << std::endl;
    }

    // Creating batch indices for scattering
    int* batch_indices, *count_per_process, *displs;

    if (my_rank == MASTER_RANK)
    {
        batch_indices = new int[glob_batch_size];
        count_per_process = new int[comm_size];
        displs = new int[comm_size];

        for (int i = 0; i < comm_size; ++i)
        {
            count_per_process[i] = static_cast<int>(floor(batch_size_f / static_cast<float>(comm_size)));
            if (i == comm_size - 1)
            {
                count_per_process[i] += glob_batch_size % comm_size;
            }
            displs[i] = i * count_per_process[0];
        }
    }

    // Broadcasting weights and biases
    MPI_Bcast(w1, input_layer_size * hidden_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(w2, hidden_layer_size * hidden_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(w3, hidden_layer_size * hidden_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(w4, hidden_layer_size * output_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

    MPI_Bcast(b1, hidden_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(b2, hidden_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(b3, hidden_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(b4, output_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

    // Defining forward inputs
    std::tuple<float*, float*, float*, float*> weights = std::make_tuple(w1, w2, w3, w4);
    std::tuple<float*, float*, float*, float*> biases = std::make_tuple(b1, b2, b3, b4);
    std::tuple<float*, int, int> b_input = std::make_tuple(my_batch_x, my_batch_size, input_layer_size);
    std::tuple<float*, float*, float*, float*> z_values = std::make_tuple(z1, z2, z3, z4); 
    std::tuple<int, int, int, int> dims = std::make_tuple(hidden_layer_size, hidden_layer_size, hidden_layer_size, output_layer_size);

    // Defining backward inputs
    std::tuple<float*, float*, float*> weights_T = std::make_tuple(w2_T, w3_T, w4_T);
    std::tuple<float*, float*, float*, float*> weight_grads = std::make_tuple(dw1, dw2, dw3, dw4);
    std::tuple<float*, float*, float*, float*> bias_grads = std::make_tuple(db1, db2, db3, db4);
    std::tuple<float*, int, int> b_input_T = std::make_tuple(my_batch_x_T, input_layer_size, my_batch_size);
    std::tuple<float*, int, int> target = std::make_tuple(my_batch_y, my_batch_size, output_layer_size);

    if (my_rank == MASTER_RANK)
    {
        std::cout << "Starting training loop..." << std::endl;
    }

    // Timing the training loop
    auto start = std::chrono::high_resolution_clock::now();

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        if (my_rank == MASTER_RANK)
        {
            random_index(batch_indices, glob_batch_size, train_rows, unif);
        }

        // Scattering batch indices
        MPI_Scatterv(batch_indices, count_per_process, displs, MPI_INT, my_batch_indices, my_batch_size, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
        
        for (int i = 0; i < my_batch_size; ++i)
        {
            m_copy_row(train_x, my_batch_x, my_batch_indices[i], i, train_rows, my_batch_size, input_layer_size);
            m_copy_row(train_y, my_batch_y, my_batch_indices[i], i, train_rows, my_batch_size, train_y_cols);
        }

        // Forward pass
        
        forward_pass(weights, biases, b_input, z_values, dims, threads);
        m_copy(z4, y_hat, my_batch_size, output_layer_size);


        // Backward pass
        m_transpose(w1, w1_T, input_layer_size, hidden_layer_size);
        m_transpose(w2, w2_T, hidden_layer_size, hidden_layer_size);
        m_transpose(w3, w3_T, hidden_layer_size, hidden_layer_size);
        m_transpose(w4, w4_T, hidden_layer_size, output_layer_size);
        m_transpose(my_batch_x, my_batch_x_T, input_layer_size, my_batch_size);

        backward_pass(weights_T, weight_grads, bias_grads, b_input_T, target, z_values, dims, threads);

        // Reduce gradients
        MPI_Reduce(dw1, glob_dw1, input_layer_size * hidden_layer_size, MPI_FLOAT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Reduce(dw2, glob_dw2, hidden_layer_size * hidden_layer_size, MPI_FLOAT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Reduce(dw3, glob_dw3, hidden_layer_size * hidden_layer_size, MPI_FLOAT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Reduce(dw4, glob_dw4, hidden_layer_size * output_layer_size, MPI_FLOAT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);

        MPI_Reduce(db1, glob_db1, hidden_layer_size, MPI_FLOAT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Reduce(db2, glob_db2, hidden_layer_size, MPI_FLOAT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Reduce(db3, glob_db3, hidden_layer_size, MPI_FLOAT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Reduce(db4, glob_db4, output_layer_size, MPI_FLOAT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);

        // Update weights and biases
        if (my_rank == MASTER_RANK)
        {
            #pragma omp parallel num_threads(threads)
            {
            m_scalar_mul(glob_dw1, learning_rate/batch_size_f, input_layer_size, hidden_layer_size);
            m_scalar_mul(glob_dw2, learning_rate/batch_size_f, hidden_layer_size, hidden_layer_size);
            m_scalar_mul(glob_dw3, learning_rate/batch_size_f, hidden_layer_size, hidden_layer_size);
            m_scalar_mul(glob_dw4, learning_rate/batch_size_f, hidden_layer_size, output_layer_size);

            m_scalar_mul(glob_db1, learning_rate/batch_size_f, 1, hidden_layer_size);
            m_scalar_mul(glob_db2, learning_rate/batch_size_f, 1, hidden_layer_size);
            m_scalar_mul(glob_db3, learning_rate/batch_size_f, 1, hidden_layer_size);
            m_scalar_mul(glob_db4, learning_rate/batch_size_f, 1, output_layer_size);
            
            m_sub(w1, glob_dw1, input_layer_size , hidden_layer_size);
            m_sub(w2, glob_dw2, hidden_layer_size, hidden_layer_size);
            m_sub(w3, glob_dw3, hidden_layer_size, hidden_layer_size);
            m_sub(w4, glob_dw4, hidden_layer_size, output_layer_size);

            m_sub(b1, glob_db1, 1, hidden_layer_size);
            m_sub(b2, glob_db2, 1, hidden_layer_size);
            m_sub(b3, glob_db3, 1, hidden_layer_size);
            m_sub(b4, glob_db4, 1, output_layer_size);
            }
        }

        MPI_Bcast(w1, input_layer_size * hidden_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Bcast(w2, hidden_layer_size * hidden_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Bcast(w3, hidden_layer_size * hidden_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Bcast(w4, hidden_layer_size * output_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

        MPI_Bcast(b1, hidden_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Bcast(b2, hidden_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Bcast(b3, hidden_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Bcast(b4, output_layer_size, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
        

        if ((epoch + 1) % 10 == 0 && my_rank == MASTER_RANK)
        {
            m_argmax(y_hat, y_class, my_batch_size, output_layer_size, 1);
            float acc = accuracy(y_class, my_batch_y, my_batch_size);
            float loss = cross_entropy_loss(y_hat, my_batch_y, my_batch_size, output_layer_size);
            std::cout << "Epoch: " << epoch+1 << "---" << "Loss: " << loss << " Accuracy: " << acc << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    if (my_rank == MASTER_RANK)
    {
        double total_duration;
        MPI_Reduce(&duration, &total_duration, 1, MPI_DOUBLE, MPI_MAX, MASTER_RANK, MPI_COMM_WORLD);
        std::cout << "Training loop took " << duration.count() << " seconds." << std::endl;
    }
    else
    {
        MPI_Reduce(&duration, NULL, 1, MPI_DOUBLE, MPI_MAX, MASTER_RANK, MPI_COMM_WORLD);
    }

    if (my_rank == MASTER_RANK)
    {
        delete[] glob_dw1;
        delete[] glob_dw2;
        delete[] glob_dw3;
        delete[] glob_dw4;
        delete[] glob_db1;
        delete[] glob_db2;
        delete[] glob_db3;
        delete[] glob_db4;
    }

    // Freeing memory
    delete[] w1;
    delete[] w2;
    delete[] w3;
    delete[] w4;
    delete[] w1_T;
    delete[] w2_T;
    delete[] w3_T;
    delete[] w4_T;
    delete[] b1;
    delete[] b2;
    delete[] b3;
    delete[] b4;
    delete[] dw1;
    delete[] dw2;
    delete[] dw3;
    delete[] dw4;
    delete[] db1;
    delete[] db2;
    delete[] db3;
    delete[] db4;
    delete[] z1;
    delete[] z2;
    delete[] z3;
    delete[] z4;
    delete[] my_batch_x;
    delete[] my_batch_x_T;
    delete[] my_batch_y;
}

// DO NOT MULTITHREAD THIS FUNCTION: Distribution is not thread safe
void random_index(int* batch_indices, int size, int max_value, std::mt19937& gen)
{
    std::uniform_int_distribution<> dis(0, max_value - 1);
    for (int i = 0; i < size; ++i)
    {
        batch_indices[i] = dis(gen);
    }
}