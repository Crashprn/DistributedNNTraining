#include "nn_utils.hpp"

float cross_entropy_loss(float* y_hat, float* y, int rows, int cols)
{
    float loss = 0.0;
    //#pragma omp for reduction(+:loss)
    for (int i = 0; i < rows; ++i)
    {
        int col_idx = static_cast<int>(y[i]);
        loss += -log(y_hat[i * cols + col_idx]);
    }
    return loss /= static_cast<float>(rows);
}

float accuracy(const int* y_hat, const float* y, int rows)
{
    int correct = 0;
    //#pragma omp for reduction(+:correct)
    for (int i = 0; i < rows; ++i)
    {
        int label = static_cast<int>(y[i]);
        if (label == y_hat[i])
        {
            correct += 1;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(rows);
}

// DO NOT MULTI-THREAD: Random number generation is not thread-safe
void m_he_weight_init(float* weight_mat, int rows, int cols, std::mt19937 &gen)
{
    float normalize = 2.0f / sqrt(static_cast<float>(rows));
    std::normal_distribution<float> dist(0.0, normalize);
    for (int i = 0 ; i < rows * cols; ++i)
    {
        weight_mat[i] = dist(gen);
    }

}

// DO NOT MULTI-THREAD: Random number generation is not thread-safe
void m_xavier_weight_init(float* weight_mat, int rows, int cols, std::mt19937 &gen)
{
    float normalize = 1.0f / sqrt(static_cast<float>(rows));
    std::uniform_real_distribution<float> dist(-normalize, normalize);

    for (int i = 0 ; i < rows * cols; ++i)
    {
        weight_mat[i] = dist(gen);
    }
}

void m_constant_weight_init(float* weight_mat, int rows, int cols, float value)
{
    #pragma omp for
    for (int i = 0 ; i < rows * cols; ++i)
    {
        weight_mat[i] = value;
    }
}

void forward_pass(
    std::tuple<float*, float*, float*, float*> &weights, 
    std::tuple<float*, float*, float*, float*> &biases,
    std::tuple<float*, int, int> input,
    std::tuple<float*, float*, float*, float*> &z,
    std::tuple<float*, float*, float*> &a,
    std::tuple<int, int, int, int> &dims
)
{

    float* w1 = std::get<0>(weights);
    float* w2 = std::get<1>(weights);
    float* w3 = std::get<2>(weights);
    float* w4 = std::get<3>(weights);

    float* b1 = std::get<0>(biases);
    float* b2 = std::get<1>(biases);
    float* b3 = std::get<2>(biases);
    float* b4 = std::get<3>(biases);

    float* x = std::get<0>(input);
    int batch_size = std::get<1>(input);
    int input_feats = std::get<2>(input);

    float* z1 = std::get<0>(z);
    float* z2 = std::get<1>(z);
    float* z3 = std::get<2>(z);
    float* z4 = std::get<3>(z);

    float* a1 = std::get<0>(a);
    float* a2 = std::get<1>(a);
    float* a3 = std::get<2>(a);

    int h1 = std::get<0>(dims);
    int h2 = std::get<1>(dims);
    int h3 = std::get<2>(dims);
    int out_dim = std::get<3>(dims);

    // Layer 1
    cpu_matrix::m_mul(x, w1, z1, batch_size, input_feats, input_feats, h1); // (batch_size, input_cols) * (input_cols, h1) -> (batch_size, h1)
    cpu_matrix::m_add_v(z1, b1, batch_size, h1, 1, h1); // (batch_size, h1) + (,h1) -> (batch_size, h1)
    cpu_matrix::m_copy(z1, a1, batch_size, h1); // z1 -> inter1
    cpu_matrix::m_Relu(a1, batch_size, h1); // (batch_size, h1) -> (batch_size, h1)
    // Layer 2
    cpu_matrix::m_mul(a1, w2, z2, batch_size, h1, h1, h2); // (batch_size, h1) * (h1, h2) -> (batch_size, h2)
    cpu_matrix::m_add_v(z2, b2, batch_size, h2, 1, h2); // (batch_size, h2) + (,h2) -> (batch_size, h2)
    cpu_matrix::m_copy(z2, a2, batch_size, h2); // z2 -> inter2 
    cpu_matrix::m_Relu(a2, batch_size, h2); // (batch_size, h2) -> (batch_size, h2)

    // Layer 3
    cpu_matrix::m_mul(a2, w3, z3, batch_size, h2, h2, h3); // (batch_size, h2) * (h2, h3) -> (batch_size, h3)
    cpu_matrix::m_add_v(z3, b3, batch_size, h3, 1, h3); // (batch_size, h3) + (,h3) -> (batch_size, h3)
    cpu_matrix::m_copy(z3, a3, batch_size, h3); // z3 -> inter3 
    cpu_matrix::m_Relu(a3, batch_size, h3); // (batch_size, h3) -> (batch_size, h3)

    // Layer 4
    cpu_matrix::m_mul(a3, w4, z4, batch_size, h3, h3, out_dim); // (batch_size, h3) * (h3, out_dim) -> (batch_size, out_dim)
    cpu_matrix::m_add_v(z4, b4, batch_size, out_dim, 1, out_dim); // (batch_size, out_dim) + (,out_dim) -> (batch_size, out_dim)
    cpu_matrix::m_softmax(z4, batch_size, out_dim); // (batch_size, out_dim) -> (batch_size, out_dim)
}

void forward_pass_gpu(
    std::tuple<float*, float*, float*, float*> &weights, 
    std::tuple<float*, float*, float*, float*> &biases,
    std::tuple<float*, int, int> input,
    std::tuple<float*, float*, float*, float*> &z,
    std::tuple<float*, float*, float*> &a,
    std::tuple<int, int, int, int> &dims
)
{

    float* d_w1 = std::get<0>(weights);
    float* d_w2 = std::get<1>(weights);
    float* d_w3 = std::get<2>(weights);
    float* d_w4 = std::get<3>(weights);

    float* d_b1 = std::get<0>(biases);
    float* d_b2 = std::get<1>(biases);
    float* d_b3 = std::get<2>(biases);
    float* d_b4 = std::get<3>(biases);

    float* d_x = std::get<0>(input);
    int batch_size = std::get<1>(input);
    int input_feats = std::get<2>(input);

    float* d_z1 = std::get<0>(z);
    float* d_z2 = std::get<1>(z);
    float* d_z3 = std::get<2>(z);
    float* d_z4 = std::get<3>(z);

    float* d_a1 = std::get<0>(a);
    float* d_a2 = std::get<1>(a);
    float* d_a3 = std::get<2>(a);

    int h1 = std::get<0>(dims);
    int h2 = std::get<1>(dims);
    int h3 = std::get<2>(dims);
    int out_dim = std::get<3>(dims);

    // Layer 1
    cuda_matrix::m_mul(d_x, d_w1, d_z1, batch_size, input_feats, input_feats, h1); // (batch_size, input_cols) * (input_cols, h1) -> (batch_size, h1) 
    cuda_matrix::m_add_v(d_z1, d_b1, batch_size, h1, 1, h1); // (batch_size, h1) + (,h1) -> (batch_size, h1)
    cuda_matrix::m_copy(d_z1, d_a1, batch_size, h1); // d_z1 -> inter1
    cuda_matrix::m_Relu(d_a1, batch_size, h1); // (batch_size, h1) -> (batch_size, h1)



    // Layer 2
    cuda_matrix::m_mul(d_a1, d_w2, d_z2, batch_size, h1, h1, h2); // (batch_size, h1) * (h1, h2) -> (batch_size, h2)
    cuda_matrix::m_add_v(d_z2, d_b2, batch_size, h2, 1, h2); // (batch_size, h2) + (,h2) -> (batch_size, h2)
    cuda_matrix::m_copy(d_z2, d_a2, batch_size, h2); // d_z2 -> inter2
    cuda_matrix::m_Relu(d_a2, batch_size, h2); // (batch_size, h2) -> (batch_size, h2)

    // Layer 3
    cuda_matrix::m_mul(d_a2, d_w3, d_z3, batch_size, h2, h2, h3); // (batch_size, h2) * (h2, h3) -> (batch_size, h3) 
    cuda_matrix::m_add_v(d_z3, d_b3, batch_size, h3, 1, h3); // (batch_size, h3) + (,h3) -> (batch_size, h3)
    cuda_matrix::m_copy(d_z3, d_a3, batch_size, h3); // d_z3 -> inter3
    cuda_matrix::m_Relu(d_a3, batch_size, h3); // (batch_size, h3) -> (batch_size, h3)
    

    // Layer 4
    cuda_matrix::m_mul(d_a3, d_w4, d_z4, batch_size, h3, h3, out_dim); // (batch_size, h3) * (h3, out_dim) -> (batch_size, out_dim)
    cuda_matrix::m_add_v(d_z4, d_b4, batch_size, out_dim, 1, out_dim); // (batch_size, out_dim) + (,out_dim) -> (batch_size, out_dim)
    cuda_matrix::m_softmax(d_z4, batch_size, out_dim); // (batch_size, out_dim) -> (batch_size, out_dim)
}

void backward_pass(
    std::tuple<float*, float*, float*> &weights_T, 
    std::tuple<float*, float*, float*, float*> &weight_grads,
    std::tuple<float*, float*, float*, float*> &bias_grads,
    std::tuple<float*, int, int> input_T,
    std::tuple<float*, int, int> target,
    std::tuple<float*, float*, float*, float*> &z,
    std::tuple<float*, float*, float*> &a_T,
    std::tuple<float*, float*, float*> &deltas,
    std::tuple<int, int, int, int> &dims
)
{
    int h1 = std::get<0>(dims);
    int h2 = std::get<1>(dims);
    int h3 = std::get<2>(dims);
    int out_dim = std::get<3>(dims);

    float* w2_T = std::get<0>(weights_T);
    float* w3_T = std::get<1>(weights_T);
    float* w4_T = std::get<2>(weights_T);

    float* dw1 = std::get<0>(weight_grads);
    float* dw2 = std::get<1>(weight_grads);
    float* dw3 = std::get<2>(weight_grads);
    float* dw4 = std::get<3>(weight_grads);

    float* db1 = std::get<0>(bias_grads);
    float* db2 = std::get<1>(bias_grads);
    float* db3 = std::get<2>(bias_grads);
    float* db4 = std::get<3>(bias_grads);

    float* x_T = std::get<0>(input_T);
    int input_feats = std::get<1>(input_T);
    int batch_size = std::get<2>(input_T);

    float* y_targ = std::get<0>(target);

    float* z1 = std::get<0>(z);
    float* z2 = std::get<1>(z);
    float* z3 = std::get<2>(z);
    float* z4 = std::get<3>(z);

    float* a1_T = std::get<0>(a_T);
    float* a2_T = std::get<1>(a_T);
    float* a3_T = std::get<2>(a_T);

    float* delta1 = std::get<0>(deltas);
    float* delta2 = std::get<1>(deltas);
    float* delta3 = std::get<2>(deltas);


    // Layer 4 Gradients
    // d4 = S(z4) - y
    cpu_matrix::m_sub(z4, y_targ, batch_size, out_dim); // (batch_size, out_dim) - (batch_size, out_dim) -> (batch_size, out_dim)

    // grad b = 1/batch_size * sum(d4)
    cpu_matrix::m_sum(z4, db4, batch_size, out_dim, 0); // (batch_size, out_dim) -> (1, out_dim)

    // grad w = 1/batch_size * sig(z3)^T * d4
    cpu_matrix::m_mul(a3_T, z4, dw4, h3, batch_size, batch_size, out_dim); // (h3, batch_size) * (batch_size, out_dim) -> (h3, out_dim)

    // Layer 3 Gradients
    
    // Calculating delta3
    cpu_matrix::m_mul(z4, w4_T, delta3, batch_size, out_dim, out_dim, h3); // (batch_size, out_dim) * (out_dim, h3) -> (batch_size, h3)
    cpu_matrix::m_Relu_deriv(z3, batch_size, h3); // (batch_size, h3) -> (batch_size, h3)
    cpu_matrix::m_hadamard(delta3, z3, batch_size, h3); // (batch_size, h3) * (batch_size, h3) -> (batch_size, h3)

    // grad b = 1/batch_size * sum(d3)
    cpu_matrix::m_sum(delta3, db3, batch_size, h3, 0); // (batch_size, h3) -> (1, h3)

    // grad w = 1/batch_size * sig(z_2)^T * d3
    cpu_matrix::m_mul(a2_T, delta3, dw3, h2, batch_size, batch_size, h3); // (h2, batch_size) * (batch_size, h3) -> (h2, h3)

    // Layer 2 Gradients
    
    // Calculating delta2
    cpu_matrix::m_mul(delta3, w3_T, delta2, batch_size, h3, h3, h2); // (batch_size, h3) * (h3, h2) -> (batch_size, h2)
    cpu_matrix::m_Relu_deriv(z2, batch_size, h2); // (batch_size, h2) -> (batch_size, h2)
    cpu_matrix::m_hadamard(delta2, z2, batch_size, h2); // (batch_size, h2) * (batch_size, h2) -> (batch_size, h2)

    // grad b = 1/batch_size * sum(d2)
    cpu_matrix::m_sum(delta2, db2, batch_size, h2, 0); // (batch_size, h2) -> (1, h2)

    // grad w = 1/batch_size * sig(z_1)^T * d2
    cpu_matrix::m_mul(a1_T, delta2, dw2, h1, batch_size, batch_size, h2); // (h1, batch_size) * (batch_size, h2) -> (h1, h2)

    // Layer 1 Gradients

    // Calculating delta1
    cpu_matrix::m_mul(delta2, w2_T, delta1, batch_size, h2, h2, h1); // (batch_size, h2) * (h2, h1) -> (batch_size, h1)
    cpu_matrix::m_Relu_deriv(z1, batch_size, h1); // (batch_size, h1) -> (batch_size, h1)
    cpu_matrix::m_hadamard(delta1, z1, batch_size, h1); // (batch_size, h1) * (batch_size, h1) -> (batch_size, h1)

    // grad b = 1/batch_size * sum(d1)
    cpu_matrix::m_sum(delta1, db1, batch_size, h1, 0); // (batch_size, h1) -> (1, h1)

    // grad w = 1/batch_size * x^T * d1
    cpu_matrix::m_mul(x_T, delta1, dw1, input_feats, batch_size, batch_size, h1); // (input_feats, batch_size) * (batch_size, h1) -> (input_feats, h1)
}

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
)
{
    int h1 = std::get<0>(dims);
    int h2 = std::get<1>(dims);
    int h3 = std::get<2>(dims);
    int out_dim = std::get<3>(dims);

    float* d_w2_T = std::get<0>(weights_T);
    float* d_w3_T = std::get<1>(weights_T);
    float* d_w4_T = std::get<2>(weights_T);

    float* d_dw1 = std::get<0>(weight_grads);
    float* d_dw2 = std::get<1>(weight_grads);
    float* d_dw3 = std::get<2>(weight_grads);
    float* d_dw4 = std::get<3>(weight_grads);

    float* d_db1 = std::get<0>(bias_grads);
    float* d_db2 = std::get<1>(bias_grads);
    float* d_db3 = std::get<2>(bias_grads);
    float* d_db4 = std::get<3>(bias_grads);

    float* d_x_T = std::get<0>(input_T);
    int input_feats = std::get<1>(input_T);
    int batch_size = std::get<2>(input_T);

    float* d_y_targ = std::get<0>(target);

    float* d_z1 = std::get<0>(z);
    float* d_z2 = std::get<1>(z);
    float* d_z3 = std::get<2>(z);
    float* d_z4 = std::get<3>(z);

    float* d_a1_T = std::get<0>(a_T);
    float* d_a2_T = std::get<1>(a_T);
    float* d_a3_T = std::get<2>(a_T);

    float* d_delta1 = std::get<0>(deltas);
    float* d_delta2 = std::get<1>(deltas);
    float* d_delta3 = std::get<2>(deltas);


    // Layer 4 Gradients
    // d4 = S(z4) - y
    cuda_matrix::m_sub(d_z4, d_y_targ, batch_size, out_dim); // (batch_size, out_dim) - (batch_size, out_dim) -> (batch_size, out_dim)

    // grad b = sum(d4)
    cuda_matrix::m_sum(d_z4, d_db4, batch_size, out_dim, 0); // (batch_size, out_dim) -> (1, out_dim)

    // grad w = sig(z3)^T * d4
    cuda_matrix::m_mul(d_a3_T, d_z4, d_dw4, h3, batch_size, batch_size, out_dim); // (h3, batch_size) * (batch_size, out_dim) -> (h3, out_dim)

    // Layer 3 Gradients
    // Calculating delta3
    cuda_matrix::m_mul(d_z4, d_w4_T, d_delta3, batch_size, out_dim, out_dim, h3); // (batch_size, out_dim) * (out_dim, h3) -> (batch_size, h3)
    cuda_matrix::m_Relu_deriv(d_z3, batch_size, h3); // (batch_size, h3) -> (batch_size, h3)
    cuda_matrix::m_hadamard(d_delta3, d_z3, batch_size, h3); // (batch_size, h3) * (batch_size, h3) -> (batch_size, h3)

    // grad b = 1/batch_size * sum(d3)
    cuda_matrix::m_sum(d_delta3, d_db3, batch_size, h3, 0); // (batch_size, h3) -> (1, h3)

    // grad w = 1/batch_size * sig(z_2)^T * d3
    cuda_matrix::m_mul(d_a2_T, d_delta3, d_dw3, h2, batch_size, batch_size, h3); // (h2, batch_size) * (batch_size, h3) -> (h2, h3)
    
    // Layer 2 Gradients
    // Calculating delta2
    cuda_matrix::m_mul(d_delta3, d_w3_T, d_delta2, batch_size, h3, h3, h2); // (batch_size, h3) * (h3, h2) -> (batch_size, h2)
    cuda_matrix::m_Relu_deriv(d_z2, batch_size, h2); // (batch_size, h2) -> (batch_size, h2)
    cuda_matrix::m_hadamard(d_delta2, d_z2, batch_size, h2); // (batch_size, h2) * (batch_size, h2) -> (batch_size, h2)

    // grad b = 1/batch_size * sum(d2)
    cuda_matrix::m_sum(d_delta2, d_db2, batch_size, h2, 0); // (batch_size, h2) -> (1, h2)

    // grad w = 1/batch_size * sig(z_1)^T * d2
    cuda_matrix::m_mul(d_a1_T, d_delta2, d_dw2, h1, batch_size, batch_size, h2); // (h1, batch_size) * (batch_size, h2) -> (h1, h2)

    // Layer 1 Gradients
    // Calculating delta1
    cuda_matrix::m_mul(d_delta2, d_w2_T, d_delta1, batch_size, h2, h2, h1); // (batch_size, h2) * (h2, h1) -> (batch_size, h1)
    cuda_matrix::m_Relu_deriv(d_z1, batch_size, h1); // (batch_size, h1) -> (batch_size, h1)
    cuda_matrix::m_hadamard(d_delta1, d_z1, batch_size, h1); // (batch_size, h1) * (batch_size, h1) -> (batch_size, h1)

    // grad b = 1/batch_size * sum(d1)
    cuda_matrix::m_sum(d_delta1, d_db1, batch_size, h1, 0); // (batch_size, h1) -> (1, h1)

    // grad w = 1/batch_size * x^T * d1
    cuda_matrix::m_mul(d_x_T, d_delta1, d_dw1, input_feats, batch_size, batch_size, h1); // (input_feats, batch_size) * (batch_size, h1) -> (input_feats, h1)
}