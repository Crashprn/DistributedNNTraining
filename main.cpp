#include <iostream>
#include <filesystem>
#include <chrono>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "src/matrix_utils.hpp"
#include "src/data_utils.hpp"
#include "src/nn_utils.hpp"
#include "src/nn_trainer.hpp"


int count_non_zero(const float* matrix, int rows, int cols)
{   
    int count = 0;
    for (int i = 0; i < rows*cols; ++i)
    {
        if (matrix[i] > 0.0f)
        {
            count++;
        }
    }
    return count;
}


int main(int argc, char* argv[])
{

    int threads, num_epochs, batch_size;

    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <num_epochs> <batch_size>" << std::endl;
        threads = 1;
        num_epochs = 10;
        batch_size = 32;
    }
    else
    {
        threads = std::stoi(argv[1]);
        num_epochs = std::stoi(argv[2]);
        batch_size = std::stoi(argv[3]);

    }

    
    int mnist_cols = 28*28;
    int mnist_rows = 3000;

    float* mnist_train_x = new float[mnist_rows * mnist_cols];
    float* mnist_train_y = new float[mnist_rows * 1];


    std::string mnist_train_x_file = "../data/mnist_x.csv";
    std::string mnist_train_y_file = "../data/mnist_y.csv";

    try
    {
    read_matrix_from_file(mnist_train_x_file, mnist_train_x, mnist_rows, mnist_cols);
    read_matrix_from_file(mnist_train_y_file, mnist_train_y, mnist_rows, 1);
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Successfully read data from files." << std::endl;
    m_scalar_mul(mnist_train_x, 1.0f/255.0f, mnist_rows, mnist_cols);
    std::cout << "Successfully normalized data." << std::endl;

    int input_layer_size = mnist_cols;
    int hidden_layer_size = 300;
    int output_layer_size = 10;
    float learning_rate = 0.005f;

    std::cout << "Starting training loop..." << std::endl;
    // Timing the training loop
    auto start = std::chrono::high_resolution_clock::now();

    training_loop(
        mnist_train_x,
        mnist_train_y,
        mnist_rows,
        input_layer_size,
        1,
        output_layer_size,
        hidden_layer_size,
        num_epochs,
        batch_size,
        learning_rate,
        threads
    );
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Training loop took " << duration.count() << " seconds." << std::endl;

    return 0;
}