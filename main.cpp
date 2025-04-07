#include <iostream>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include <mpi.h>

#include "src/matrix_utils.hpp"
#include "src/data_utils.hpp"
#include "src/nn_utils.hpp"
#include "src/nn_trainer.hpp"

int MASTER_RANK = 0;

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
    int my_rank, comm_size;
    std::string device; // default device

    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <num_epochs> <batch_size> <device>" << std::endl;
        threads = 2;
        num_epochs = 2;
        batch_size = 100;
        device = "cpu"; // default device
    }
    else
    {
    if (my_rank == MASTER_RANK)
    {
        threads = std::stoi(argv[1]);
        num_epochs = std::stoi(argv[2]);
        batch_size = std::stoi(argv[3]);
        device = argv[4];
    }
    }

    MPI_Bcast(&threads, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&num_epochs, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&batch_size, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&device[0], static_cast<int>(device.size()) + 1, MPI_CHAR, MASTER_RANK, MPI_COMM_WORLD); // +1 for null terminator

    omp_set_num_threads(threads);
    
    int mnist_cols = 28*28;
    int mnist_rows = 20000;

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
    if (my_rank == MASTER_RANK)
    {
        std::cout << "Successfully read data from files." << std::endl;
    }

    cpu_matrix::m_scalar_mul(mnist_train_x, 1.0f/255.0f, mnist_rows, mnist_cols);

    int input_layer_size = mnist_cols;
    int hidden_layer_size = 300;
    int output_layer_size = 10;
    float learning_rate = 0.005f;

    if (device == "gpu" || device == "cuda")
    {
        std::cout << "Using GPU for training." << std::endl;
        training_loop_gpu(
            mnist_train_x,
            mnist_train_y,
            mnist_rows,
            input_layer_size,
            1, // train_y_cols
            output_layer_size,
            hidden_layer_size,
            num_epochs,
            batch_size,
            learning_rate,
            my_rank,
            comm_size,
            MASTER_RANK
        );
    }
    else
    {
        std::cout << "Using CPU for training." << std::endl;
        training_loop_cpu(
            mnist_train_x,
            mnist_train_y,
            mnist_rows,
            input_layer_size,
            1, // train_y_cols
            output_layer_size,
            hidden_layer_size,
            num_epochs,
            batch_size,
            learning_rate,
            my_rank,
            comm_size,
            MASTER_RANK
        );
    }
    
    MPI_Finalize();    

    delete[] mnist_train_x;
    delete[] mnist_train_y;

    return 0;
}