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
    int batch_size,
    float learning_rate,
    int threads
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

    // Initializing z values for each layer
    float* z1 = new float[batch_size * hidden_layer_size];
    float* z2 = new float[batch_size * hidden_layer_size];
    float* z3 = new float[batch_size * hidden_layer_size];
    float* z4 = new float[batch_size * output_layer_size];

    // Initializing an output matrix and class vector
    float* y_hat = new float[batch_size * output_layer_size];
    int* y_class = new int[batch_size * 1];

    // Creating matrice for batch of data
    float* batch_x = new float[batch_size * input_layer_size]; // batch_size x input_layer_size
    float* batch_x_T = new float[input_layer_size * batch_size]; // input_layer_size x batch_size
    float* batch_y = new float[batch_size * train_y_cols]; // batch_size x 1
    float batch_size_f = static_cast<float>(batch_size);
    int* batch_indices = new int[batch_size];

    // Initializing weights and biases
    std::random_device rd;
    std::mt19937 normal(rd());
    std::mt19937 unif(rd());

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

    // Defining forward inputs
    std::tuple<float*, float*, float*, float*> weights = std::make_tuple(w1, w2, w3, w4);
    std::tuple<float*, float*, float*, float*> biases = std::make_tuple(b1, b2, b3, b4);
    std::tuple<float*, int, int> b_input = std::make_tuple(batch_x, batch_size, input_layer_size);
    std::tuple<float*, float*, float*, float*> z_values = std::make_tuple(z1, z2, z3, z4); 
    std::tuple<int, int, int, int> dims = std::make_tuple(hidden_layer_size, hidden_layer_size, hidden_layer_size, output_layer_size);

    // Defining backward inputs
    std::tuple<float*, float*, float*> weights_T = std::make_tuple(w2_T, w3_T, w4_T);
    std::tuple<float*, float*, float*, float*> weight_grads = std::make_tuple(dw1, dw2, dw3, dw4);
    std::tuple<float*, float*, float*, float*> bias_grads = std::make_tuple(db1, db2, db3, db4);
    std::tuple<float*, int, int> b_input_T = std::make_tuple(batch_x_T, input_layer_size, batch_size);
    std::tuple<float*, int, int> target = std::make_tuple(batch_y, batch_size, output_layer_size);

    std::cout << "Starting training loop..." << std::endl;
    // Timing the training loop
    auto start = std::chrono::high_resolution_clock::now();

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        random_index(batch_indices, batch_size, train_rows, unif);

        
        for (int i = 0; i < batch_size; ++i)
        {
            int index = batch_indices[i];
            m_copy_row(train_x, batch_x, index, i, train_rows, batch_size, input_layer_size);
            m_copy_row(train_y, batch_y, index, i, train_rows, batch_size, train_y_cols);
        }

        // Forward pass
        
        forward_pass(weights, biases, b_input, z_values, dims, threads);
        m_copy(z4, y_hat, batch_size, output_layer_size);

        // Backward pass
        m_transpose(w1, w1_T, input_layer_size, hidden_layer_size);
        m_transpose(w2, w2_T, hidden_layer_size, hidden_layer_size);
        m_transpose(w3, w3_T, hidden_layer_size, hidden_layer_size);
        m_transpose(w4, w4_T, hidden_layer_size, output_layer_size);

        m_transpose(batch_x, batch_x_T, input_layer_size, batch_size);

        
        backward_pass(weights_T, weight_grads, bias_grads, b_input_T, target, z_values, dims, threads);

        // Update weights and biases then assign to copy
        #pragma omp parallel num_threads(threads)
        {

        m_scalar_mul(dw1, learning_rate/batch_size_f, input_layer_size, hidden_layer_size);
        m_scalar_mul(dw2, learning_rate/batch_size_f, hidden_layer_size, hidden_layer_size);
        m_scalar_mul(dw3, learning_rate/batch_size_f, hidden_layer_size, hidden_layer_size);
        m_scalar_mul(dw4, learning_rate/batch_size_f, hidden_layer_size, output_layer_size);

        m_scalar_mul(db1, learning_rate/batch_size_f, 1, hidden_layer_size);
        m_scalar_mul(db2, learning_rate/batch_size_f, 1, hidden_layer_size);
        m_scalar_mul(db3, learning_rate/batch_size_f, 1, hidden_layer_size);
        m_scalar_mul(db4, learning_rate/batch_size_f, 1, output_layer_size);
        
        m_sub(w1, dw1, input_layer_size , hidden_layer_size);
        m_sub(w2, dw2, hidden_layer_size, hidden_layer_size);
        m_sub(w3, dw3, hidden_layer_size, hidden_layer_size);
        m_sub(w4, dw4, hidden_layer_size, output_layer_size);

        m_sub(b1, db1, 1, hidden_layer_size);
        m_sub(b2, db2, 1, hidden_layer_size);
        m_sub(b3, db3, 1, hidden_layer_size);
        m_sub(b4, db4, 1, output_layer_size);
        }

        if (epoch % 10 == 0)
        {
            m_argmax(y_hat, y_class, batch_size, output_layer_size, 1);
            float acc = accuracy(y_class, batch_y, batch_size);
            float loss = cross_entropy_loss(y_hat, batch_y, batch_size, output_layer_size);
            std::cout << "Epoch: " << epoch << "---" << "Loss: " << loss << " Accuracy: " << acc << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Training loop took " << duration.count() << " seconds." << std::endl;



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
    delete[] batch_x;
    delete[] batch_x_T;
    delete[] batch_y;
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