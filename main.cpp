#include <iostream>
#include <filesystem>

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


int main()
{
    int mnist_cols = 28*28;
    int mnist_rows = 1000;

    float* mnist_train_x = new float[mnist_rows * mnist_cols];
    float* mnist_train_y = new float[mnist_rows * 1];

    //std::string mnist_train_x_file = "../data/mnist_x.csv";
    //std::string mnist_train_y_file = "../data/mnist_y.csv";

    std::string mnist_train_x_file = "data/mnist_x.csv";
    std::string mnist_train_y_file = "data/mnist_y.csv";

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
    int hidden_layer_size = 100;
    int output_layer_size = 10;
    int num_epochs = 10;
    float learning_rate = 0.00001f;
    int batch_size = 64;

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
        learning_rate
    );

    return 0;
}