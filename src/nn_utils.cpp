#include "nn_utils.hpp"

void m_index_to_one_hot(float* input, float* output, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            int entry = static_cast<int>(input[i]);
            if (entry == j)
            {
                output[i * cols + j] = 1.0;
            }
            else
            {
                output[i * cols + j] = 0.0;
            }
        }
    }
}

void m_Relu(float* input, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i)
    {
        if (input[i] < 0.0)
        {
            input[i] = 0.0;
        }
    }
}

void m_Relu_deriv(float* input, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i)
    {
        if (input[i] < 0.0)
        {
            input[i] = 0.0;
        }
        else
        {
            input[i] = 1.0;
        }
    }
}

void m_softmax(float* input, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        float* start_in = &input[i*cols];
        v_softmax(start_in, cols);
    }
}

void v_softmax(float* input, int n)
{
    // Loop through the input array and calculate the exponential of each element
    float sum = 0.0;
    for (int i = 0; i < n; ++i)
    {
        input[i] = exp(input[i]);
        sum += input[i];
    }
    // Loop through the input array again and divide each element by the sum
    for (int i = 0; i < n; ++i)
    {
        input[i] /= sum;
    }
}

float cross_entropy_loss(float* y_hat, float* y, int rows, int cols)
{
    float loss = 0.0;
    for (int i = 0; i < rows; ++i)
    {
        int col_idx = static_cast<int>(y[i]);
        loss += -log(y_hat[i * cols + col_idx]);
    }
    return loss / static_cast<float>(rows);
}

void m_he_weight_init(float* weight_mat, int rows, int cols, std::mt19937 &gen)
{
    float normalize = 2.0f / sqrt(static_cast<float>(rows));
    std::normal_distribution<float> dist(0.0, normalize);

    for (int i = 0 ; i < rows * cols; ++i)
    {
        weight_mat[i] = dist(gen);
    }

}

void forward_pass(
    std::tuple<float*, float*, float*, float*> &weights, 
    std::tuple<float*, float*, float*, float*> &biases,
    std::tuple<float*, int, int> input,
    std::tuple<float*, float*, float*, float*> &z,
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

    int h1 = std::get<0>(dims);
    int h2 = std::get<1>(dims);
    int h3 = std::get<2>(dims);
    int out_dim = std::get<3>(dims);

    float* inter1 = new float[batch_size * h1];
    float* inter2 = new float[batch_size * h2];
    float* inter3 = new float[batch_size * h3];
    
    // Layer 1
    m_mul(x, w1, z1, batch_size, input_feats, input_feats, h1); // (batch_size, input_cols) * (input_cols, h1) -> (batch_size, h1)
    m_add_v(z1, b1, batch_size, h1, 1, h1); // (batch_size, h1) + (,h1) -> (batch_size, h1)
    m_copy(z1, inter1, batch_size, h1); // z1 -> inter1
    m_Relu(inter1, batch_size, h1); // (batch_size, h1) -> (batch_size, h1)
    
    // Layer 2
    m_mul(inter1, w2, z2, batch_size, h1, h1, h2); // (batch_size, h1) * (h1, h2) -> (batch_size, h2)
    m_add_v(z2, b2, batch_size, h2, 1, h2); // (batch_size, h2) + (,h2) -> (batch_size, h2)
    m_copy(z2, inter2, batch_size, h2); // z2 -> inter2 
    m_Relu(inter2, batch_size, h2); // (batch_size, h2) -> (batch_size, h2)
    
    // Layer 3
    m_mul(inter2, w3, z3, batch_size, h2, h2, h3); // (batch_size, h2) * (h2, h3) -> (batch_size, h3)
    m_add_v(z3, b3, batch_size, h3, 1, h3); // (batch_size, h3) + (,h3) -> (batch_size, h3)
    m_copy(z3, inter3, batch_size, h3); // z3 -> inter3 
    m_Relu(inter3, batch_size, h3); // (batch_size, h3) -> (batch_size, h3)

    // Layer 4
    m_mul(inter3, w4, z4, batch_size, h3, h3, out_dim); // (batch_size, h3) * (h3, out_dim) -> (batch_size, out_dim)
    m_add_v(z4, b4, batch_size, out_dim, 1, out_dim); // (batch_size, out_dim) + (,out_dim) -> (batch_size, out_dim)
    m_softmax(z4, batch_size, out_dim); // (batch_size, out_dim) -> (batch_size, out_dim)

    delete[] inter1;
    delete[] inter2;
    delete[] inter3;
}

void backward_pass(
    std::tuple<float*, float*, float*> &weights_T, 
    std::tuple<float*, float*, float*, float*> &weight_grads,
    std::tuple<float*, float*, float*, float*> &bias_grads,
    std::tuple<float*, int, int> input_T,
    std::tuple<float*, int, int> target,
    std::tuple<float*, float*, float*, float*> &z,
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
    float batch_size_f = static_cast<float>(batch_size);

    float* y = std::get<0>(target);

    float* y_hat = new float[batch_size*out_dim];

    m_index_to_one_hot(y, y_hat, batch_size, out_dim);

    float* z1 = std::get<0>(z);
    float* z2 = std::get<1>(z);
    float* z3 = std::get<2>(z);
    float* z4 = std::get<3>(z);

    
    // Layer 4 Gradients
    float* sig_z3 = new float[batch_size*h3];
    float* sig_z3_T = new float[h3*batch_size];

    // d4 = S(z4) - y
    m_sub(z4, y_hat, batch_size, out_dim); // (batch_size, out_dim) - (batch_size, out_dim) -> (batch_size, out_dim)
    float* delta4 = z4;

    // grad b = 1/batch_size * sum(d4)
    m_sum(delta4, db4, batch_size, out_dim, 0); // (batch_size, out_dim) -> (1, out_dim)
    m_scalar_mul(db4, 1.0f /batch_size_f, 1, out_dim); // (1, out_dim) * (1.0/batch_size) -> (1, out_dim)

    // grad w = 1/batch_size * sig(z3)^T * d4
    m_copy(z3, sig_z3, batch_size, h3); // z3 -> sig_z3
    m_Relu(sig_z3, batch_size, h3); // (batch_size, h3) -> (batch_size, h3)
    m_transpose(sig_z3, sig_z3_T, batch_size, h3); // (batch_size, h3) -> (h3, batch_size)
    m_mul(sig_z3_T, delta4, dw4, h3, batch_size, batch_size, out_dim); // (h3, batch_size) * (batch_size, out_dim) -> (h3, out_dim)
    m_scalar_mul(dw4, 1.0f /batch_size_f, h3, out_dim); // (h3, out_dim) * (1.0/batch_size) -> (h3, out_dim)

    // Layer 3 Gradients
    float* delta3 = new float[batch_size*h3];
    float* sig_z2 = new float[batch_size*h2];
    float* sig_z2_T = new float[h2*batch_size];

    // Calculating delta3
    m_mul(delta4, w4_T, delta3, batch_size, out_dim, out_dim, h3); // (batch_size, out_dim) * (out_dim, h3) -> (batch_size, h3)
    m_Relu_deriv(z3, batch_size, h3); // (batch_size, h3) -> (batch_size, h3)
    m_hadamard(delta3, z3, batch_size, h3); // (batch_size, h3) * (batch_size, h3) -> (batch_size, h3)

    // grad b = 1/batch_size * sum(d3)
    m_sum(delta3, db2, batch_size, h3, 0); // (batch_size, h3) -> (1, h3)
    m_scalar_mul(db3, 1.0f /batch_size_f, 1, h3); // (1, h3) * (1.0/batch_size) -> (1, h3)

    // grad w = 1/batch_size * sig(z_2)^T * d3
    m_copy(z2, sig_z2, batch_size, h2); // z2 -> sig_z2
    m_Relu(sig_z2, batch_size, h2); // (batch_size, h2) -> (batch_size, h2)
    m_transpose(sig_z2, sig_z2_T, batch_size, h2); // (batch_size, h2) -> (h2, batch_size)
    m_mul(sig_z2_T, delta3, dw3, h2, batch_size, batch_size, h3); // (h2, batch_size) * (batch_size, h3) -> (h2, h3)
    m_scalar_mul(dw3, 1.0f /batch_size_f, h2, h3); // (h2, h3) * (1.0/batch_size) -> (h2, h3)

    // Layer 2 Gradients
    float* delta2 = new float[batch_size*h2];
    float* sig_z1 = new float[batch_size*h1];
    float* sig_z1_T = new float[h1*batch_size];

    // Calculating delta2
    m_mul(delta3, w3_T, delta2, batch_size, h3, h3, h2); // (batch_size, h3) * (h3, h2) -> (batch_size, h2)
    m_Relu_deriv(z2, batch_size, h2); // (batch_size, h2) -> (batch_size, h2)
    m_hadamard(delta2, z2, batch_size, h2); // (batch_size, h2) * (batch_size, h2) -> (batch_size, h2)

    // grad b = 1/batch_size * sum(d2)
    m_sum(delta2, db2, batch_size, h2, 0); // (batch_size, h2) -> (1, h2)
    m_scalar_mul(db2, 1.0f /batch_size_f, 1, h2); // (1, h2) * (1.0/batch_size) -> (1, h2)

    // grad w = 1/batch_size * sig(z_1)^T * d2
    m_copy(z1, sig_z1, batch_size, h1); // z1 -> sig_z1
    m_Relu(sig_z1, batch_size, h1); // (batch_size, h1) -> (batch_size, h1)
    m_transpose(sig_z1, sig_z1_T, batch_size, h1); // (batch_size, h1) -> (h1, batch_size)
    m_mul(sig_z1_T, delta2, dw2, h1, batch_size, batch_size, h2); // (h1, batch_size) * (batch_size, h2) -> (h1, h2)
    m_scalar_mul(dw2, 1.0f /batch_size_f, h1, h2); // (h1, h2) * (1.0/batch_size) -> (h1, h2)

    // Layer 1 Gradients
    float* delta1 = new float[batch_size*h1];

    // Calculating delta1
    m_mul(delta2, w2_T, delta1, batch_size, h2, h2, h1); // (batch_size, h2) * (h2, h1) -> (batch_size, h1)
    m_Relu_deriv(z1, batch_size, h1); // (batch_size, h1) -> (batch_size, h1)
    m_hadamard(delta1, z1, batch_size, h1); // (batch_size, h1) * (batch_size, h1) -> (batch_size, h1)

    // grad b = 1/batch_size * sum(d1)
    m_sum(delta1, db1, batch_size, h1, 0); // (batch_size, h1) -> (1, h1)
    m_scalar_mul(db1, 1.0f /batch_size_f, 1, h1); // (1, h1) * (1.0/batch_size) ->

    // grad w = 1/batch_size * x^T * d1
    m_mul(x_T, delta1, dw1, input_feats, batch_size, batch_size, h1); // (input_feats, batch_size) * (batch_size, h1) -> (input_feats, h1)
    m_scalar_mul(dw1, 1.0f /batch_size_f, input_feats, h1); // (input_feats, h1) * (1.0/batch_size) -> (input_feats, h1)

    delete[] y_hat;
    // Freeing layer 4
    delete[] sig_z3;
    delete[] sig_z3_T;
    // Freeing layer 3
    delete[] delta3;
    delete[] sig_z2;
    delete[] sig_z2_T;
    // Freeing layer 2
    delete[] delta2;
    delete[] sig_z1;
    delete[] sig_z1_T;
    // Freeing layer 1
    delete[] delta1;
}