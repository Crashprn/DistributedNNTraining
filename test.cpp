#include "src/matrix_utils.hpp"
#include "src/nn_utils.hpp"
#include "src/c_matrix_utils.cuh"

#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(Cpu_Matrix_Utils, Matrix_Copy_Single_Thread)
{
    float src[6] = {1, 2, 3, 4, 5, 6};
    float dest_v[6] = {0, 0, 0, 0, 0, 0};
    float dest_m[6] = {0, 0, 0, 0, 0, 0};

    cpu_matrix::m_copy(src, dest_v, 1, 6);
    cpu_matrix::m_copy(src, dest_m, 2, 3);

    for (int i = 0; i < 6; ++i)
    {
        EXPECT_EQ(src[i], dest_v[i]);
        EXPECT_EQ(src[i], dest_m[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Copy_Multi_Thread)
{
    int num_threads = 2;
    float src[6] = {1, 2, 3, 4, 5, 6};
    float dest_v[6] = {0, 0, 0, 0, 0, 0};
    float dest_m[6] = {0, 0, 0, 0, 0, 0};
    #pragma omp parallel num_threads(num_threads)
    {
            // Ensure only one thread executes the copy operations
            cpu_matrix::m_copy(src, dest_v, 1, 6);
            cpu_matrix::m_copy(src, dest_m, 2, 3);
    }

    for (int i = 0; i < 6; ++i)
    {
        EXPECT_EQ(src[i], dest_v[i]);
        EXPECT_EQ(src[i], dest_m[i]);
    }
}

TEST(Cuda_Matrix_Utils, Matrix_Copy)
{
    float src[6] = {1, 2, 3, 4, 5, 6};
    float dest_v[6] = {0, 0, 0, 0, 0, 0};
    float dest_m[6] = {0, 0, 0, 0, 0, 0};

    float* d_src = cuda_matrix::device_alloc<float>(src, 1, 6);
    float* d_dest_v = cuda_matrix::device_alloc<float>(dest_v, 1, 6);
    float* d_dest_m = cuda_matrix::device_alloc<float>(dest_m, 2, 3);

    cuda_matrix::m_copy(d_src, d_dest_v, 1, 6);
    cuda_matrix::m_copy(d_src, d_dest_m, 2, 3);

    // Copy back the results from device to host
    cuda_matrix::to<float>(dest_v, d_dest_v, 1, 6, "cpu");
    cuda_matrix::to<float>(dest_m, d_dest_m, 2, 3, "cpu");

    cuda_matrix::device_free<float>(d_src);
    cuda_matrix::device_free<float>(d_dest_v);
    cuda_matrix::device_free<float>(d_dest_m);

    for (int i = 0; i < 6; ++i)
    {
        EXPECT_EQ(src[i], dest_v[i]);
        EXPECT_EQ(src[i], dest_m[i]);
    }
}


TEST(Cpu_Matrix_Utils, Matrix_Row_Copy_Single_Thread)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float dest[9] = {0};

    for (int i = 0; i < 3; ++i)
    {
        cpu_matrix::m_copy_row(src, dest, 2-i, i, 3, 3, 3);
    }

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_EQ(src[(2-i)*3 + j], dest[i*3 + j]);
        }
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Row_Copy_Multi_Thread)
{
    int num_threads = 2;
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float dest[9] = {0};

    for (int i = 0; i < 3; ++i)
    {
        #pragma omp parallel num_threads(num_threads)
        cpu_matrix::m_copy_row(src, dest, 2-i, i, 3, 3, 3);
    }

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_EQ(src[(2-i)*3 + j], dest[i*3 + j]);
        }
    }
}

TEST(Cuda_Matrix_Utils, Matrix_Row_Copy)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float dest[9] = {0};

    // Allocate device memory
    float* d_src = cuda_matrix::device_alloc<float>(src, 3, 3);
    float* d_dest = cuda_matrix::device_alloc<float>(dest, 3, 3);

    // Copy rows from device source to device destination
    for (int i = 0; i < 3; ++i)
    {
        cuda_matrix::m_copy_row(d_src, d_dest, 2-i, i, 3, 3, 3);
    }
    
    // Copy back the results from device to host
    cuda_matrix::to<float>(dest, d_dest, 3, 3, "cpu");

    // Free device memory
    cuda_matrix::device_free<float>(d_dest);
    cuda_matrix::device_free<float>(d_src);


    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_EQ(src[(2-i)*3 + j], dest[i*3 + j]);
        }
    }
}


TEST(Cpu_Matrix_Utils, Matrix_Sum_Single_Thread)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float dest1[3] = {0};
    float dest2[3] = {0};
    float ans_axis_0[3] = {12, 15, 18};
    float ans_axis_1[3] = {6, 15, 24};

    cpu_matrix::m_sum(src, dest1, 3, 3, 0);
    cpu_matrix::m_sum(src, dest2, 3, 3, 1);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(dest1[i], ans_axis_0[i]);
        EXPECT_EQ(dest2[i], ans_axis_1[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Sum_Multi_Thread)
{
    int threads = 2;
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float dest1[3] = {0};
    float dest2[3] = {0};
    float ans_axis_0[3] = {12, 15, 18};
    float ans_axis_1[3] = {6, 15, 24};

    #pragma omp parallel num_threads(threads)
    {
        cpu_matrix::m_sum(src, dest1, 3, 3, 0);
        cpu_matrix::m_sum(src, dest2, 3, 3, 1);
    }

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(dest1[i], ans_axis_0[i]);
        EXPECT_EQ(dest2[i], ans_axis_1[i]);
    }
}

TEST(Cuda_Matrix_Utils, Matrix_Sum)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float dest1[3] = {0};
    float dest2[3] = {0};
    float ans_axis_0[3] = {12, 15, 18};
    float ans_axis_1[3] = {6, 15, 24};

    // Allocate device memory
    float* d_src = cuda_matrix::device_alloc<float>(src, 3, 3);
    float* d_dest1 = cuda_matrix::device_alloc<float>(dest1, 3, 1);
    float* d_dest2 = cuda_matrix::device_alloc<float>(dest2, 3, 1);

    // Perform sum on device
    cuda_matrix::m_sum(d_src, d_dest1, 3, 3, 0);
    cuda_matrix::m_sum(d_src, d_dest2, 3, 3, 1);

    // Copy back the results from device to host
    cuda_matrix::to<float>(dest1, d_dest1, 3, 1, "cpu");
    cuda_matrix::to<float>(dest2, d_dest2, 3, 1, "cpu");

    // Free device memory
    cuda_matrix::device_free<float>(d_src);
    cuda_matrix::device_free<float>(d_dest1);
    cuda_matrix::device_free<float>(d_dest2);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(dest1[i], ans_axis_0[i]);
        EXPECT_EQ(dest2[i], ans_axis_1[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Argmax_Single_Thread)
{
    float src[9] = {
        1, 2, 9,
        4, 8, 6,
        7, 5, 9
    };
    int dest1[3] = {0};
    int dest2[3] = {0};
    int ans_axis_0[3] = {2, 1, 0};
    int ans_axis_1[3] = {2, 1, 2};

    cpu_matrix::m_argmax(src, dest1, 3, 3, 0);
    cpu_matrix::m_argmax(src, dest2, 3, 3, 1);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(dest1[i], ans_axis_0[i]);
        EXPECT_EQ(dest2[i], ans_axis_1[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Argmax_Multi_Thread)
{
    float src[9] = {
        1, 2, 9,
        4, 8, 6,
        7, 5, 9
    };
    int dest1[3] = {0};
    int dest2[3] = {0};
    int ans_axis_0[3] = {2, 1, 0};
    int ans_axis_1[3] = {2, 1, 2};

    #pragma omp parallel num_threads(2)
    {
        cpu_matrix::m_argmax(src, dest1, 3, 3, 0);
        cpu_matrix::m_argmax(src, dest2, 3, 3, 1);
    }

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(dest1[i], ans_axis_0[i]);
        EXPECT_EQ(dest2[i], ans_axis_1[i]);
    }
}

TEST(Cuda_Matrix_Utils, Matrix_Argmax)
{
    float src[9] = {
        1, 2, 9,
        4, 8, 6,
        7, 5, 9
    };
    int dest1[3] = {0};
    int dest2[3] = {0};
    int ans_axis_0[3] = {2, 1, 0};
    int ans_axis_1[3] = {2, 1, 2};

    // Allocate device memory
    float* d_src = cuda_matrix::device_alloc<float>(src, 3, 3);
    int* d_dest1 = cuda_matrix::device_alloc<int>(dest1, 3, 1);
    int* d_dest2 = cuda_matrix::device_alloc<int>(dest2, 3, 1);

    // Perform argmax on device
    cuda_matrix::m_argmax(d_src, d_dest1, 3, 3, 0);
    cuda_matrix::m_argmax(d_src, d_dest2, 3, 3, 1);

    cuda_matrix::to<int>(dest1, d_dest1, 3, 1, "cpu");
    cuda_matrix::to<int>(dest2, d_dest2, 3, 1, "cpu");

    // Free device memory
    cuda_matrix::device_free<float>(d_src);
    cuda_matrix::device_free<int>(d_dest1);
    cuda_matrix::device_free<int>(d_dest2);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ((int)dest1[i], ans_axis_0[i]);
        EXPECT_EQ((int)dest2[i], ans_axis_1[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Add_Single_Thread)
{
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[9] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    };
    float dest[9] = {0};
    float ans[9] = {10, 10, 10, 10, 10, 10, 10, 10, 10};

    cpu_matrix::m_add(src1, src2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Add_Multi_Thread)
{   
    int threads = 3;
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[9] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    };
    float ans[9] = {10, 10, 10, 10, 10, 10, 10, 10, 10};

    #pragma omp parallel num_threads(threads)
    cpu_matrix::m_add(src1, src2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }
}

TEST(Cuda_Matrix_Utils, Matrix_Add)
{
    int rows = 32;
    int mat_size = rows*rows; // 16x16 matrix
    float src1[mat_size] = {1};
    std::fill(src1, src1 + mat_size, 1.0f); // Fill src1 with 1s
    float src2[mat_size] = {3};
    std::fill(src2, src2 + mat_size, 3.0f); // Fill src2 with 3s
    float ans[mat_size] = {4};
    std::fill(ans, ans + mat_size, 4.0f); // Fill ans with 4s

    // Allocate device memory
    float* d_src1 = cuda_matrix::device_alloc<float>(src1, 1, mat_size);
    float* d_src2 = cuda_matrix::device_alloc<float>(src2, 1, mat_size);

    cuda_matrix::m_add(d_src1, d_src2, rows, rows);

    // Copy back the results from device to host
    cuda_matrix::to<float>(src1, d_src1, 1, mat_size, "cpu");

    // Free device memory
    cuda_matrix::device_free<float>(d_src1);
    cuda_matrix::device_free<float>(d_src2);

    for (int i = 0; i < mat_size; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }


}

TEST(Cpu_Matrix_Utils, Matrix_Add_Vector_Single_Thread)
{
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    float v_1[3] = {1, 2, 3};
    float v_2[3] = {1, 2, 3};

    float ans1[9] = {
        2, 3, 4,
        6, 7, 8,
        10, 11, 12
    };
    float ans2[9] = {
        2, 4, 6,
        5, 7, 9,
        8, 10, 12
    };

    cpu_matrix::m_add_v(src1, v_1, 3, 3, 3, 1);
    cpu_matrix::m_add_v(src2, v_2, 3, 3, 1, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans1[i]);
        EXPECT_EQ(src2[i], ans2[i]);
    }
    
}

TEST(Cpu_Matrix_Utils, Matrix_Add_Vector_Multi_Thread)
{
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    float v_1[3] = {1, 2, 3};
    float v_2[3] = {1, 2, 3};

    float ans1[9] = {
        2, 3, 4,
        6, 7, 8,
        10, 11, 12
    };
    float ans2[9] = {
        2, 4, 6,
        5, 7, 9,
        8, 10, 12
    };
    int threads = 2;

    #pragma omp parallel num_threads(threads)
    {
    cpu_matrix::m_add_v(src1, v_1, 3, 3, 3, 1);
    cpu_matrix::m_add_v(src2, v_2, 3, 3, 1, 3);
    }

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans1[i]);
        EXPECT_EQ(src2[i], ans2[i]);
    }
    
}

TEST(Cuda_Matrix_Utils, Matrix_Add_Vector)
{
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    float v_1[3] = {1, 2, 3};
    float v_2[3] = {1, 2, 3};

    float ans1[9] = {
        2, 3, 4,
        6, 7, 8,
        10, 11, 12
    };
    float ans2[9] = {
        2, 4, 6,
        5, 7, 9,
        8, 10, 12
    };

    // Allocate device memory
    float* d_src1 = cuda_matrix::device_alloc<float>(src1, 3, 3);
    float* d_src2 = cuda_matrix::device_alloc<float>(src2, 3, 3);
    float* d_v1 = cuda_matrix::device_alloc<float>(v_1, 1, 3);
    float* d_v2 = cuda_matrix::device_alloc<float>(v_2, 1, 3);

    // Perform vector addition on device
    cuda_matrix::m_add_v(d_src1, d_v1, 3, 3, 3, 1);
    cuda_matrix::m_add_v(d_src2, d_v2, 3, 3, 1, 3);

    // Copy back the results from device to host
    cuda_matrix::to<float>(src1, d_src1, 3, 3, "cpu");
    cuda_matrix::to<float>(src2, d_src2, 3, 3, "cpu");

    // Free device memory
    cuda_matrix::device_free<float>(d_src1);
    cuda_matrix::device_free<float>(d_src2);
    cuda_matrix::device_free<float>(d_v1);
    cuda_matrix::device_free<float>(d_v2);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans1[i]);
        EXPECT_EQ(src2[i], ans2[i]);
    }
    
}

TEST(Cpu_Matrix_Utils, Matrix_Subtract_Single_Thread)
{
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[9] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    };
    float ans[9] = {-8, -6, -4, -2, 0, 2, 4, 6, 8};

    cpu_matrix::m_sub(src1, src2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Subtract_Multi_Thread)
{
    int threads = 2;
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[9] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    };
    float ans[9] = {-8, -6, -4, -2, 0, 2, 4, 6, 8};

    #pragma omp parallel num_threads(threads)
    cpu_matrix::m_sub(src1, src2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }
}

TEST(Cuda_Matrix_Utils, Matrix_Subtract)
{
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[9] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    };
    float ans[9] = {-8, -6, -4, -2, 0, 2, 4, 6, 8};

    // Allocate device memory
    float* d_src1 = cuda_matrix::device_alloc<float>(src1, 3, 3);
    float* d_src2 = cuda_matrix::device_alloc<float>(src2, 3, 3);

    // Perform subtraction on device
    cuda_matrix::m_sub(d_src1, d_src2, 3, 3);

    // Copy back the results from device to host
    cuda_matrix::to<float>(src1, d_src1, 3, 3, "cpu");

    // Free device memory
    cuda_matrix::device_free<float>(d_src1);
    cuda_matrix::device_free<float>(d_src2);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Multiply_Single_Thread)
{
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[9] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    };
    float dest[9] = {0};
    float ans[9] = {30, 24, 18, 84, 69, 54, 138, 114, 90};

    cpu_matrix::m_mul(src1, src2, dest, 3, 3, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(dest[i], ans[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Multiply_Multi_Thread)
{
    int threads = 2;
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[9] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    };
    float dest[9] = {0};
    float ans[9] = {30, 24, 18, 84, 69, 54, 138, 114, 90};

    #pragma omp parallel num_threads(threads)
    cpu_matrix::m_mul(src1, src2, dest, 3, 3, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(dest[i], ans[i]);
    }
}

TEST(Cuda_Matrix_Utils, Matrix_Multiply)
{
    int rows = 32;
    int mat_size = rows * rows; // 16x16 matrix
    float src1[mat_size] = {1};
    std::fill(src1, src1 + mat_size, 1.0f); // Fill src1 with 1s
    float src2[mat_size] = {1};
    std::fill(src2, src2 + mat_size, 1.0f); // Fill src2 with 1s
    float dest[mat_size] = {0};
    float ans[mat_size] = {static_cast<float>(rows)};
    std::fill(ans, ans + mat_size, static_cast<float>(rows)); // Fill ans with rows

    // Allocate device memory
    float* d_src1 = cuda_matrix::device_alloc<float>(src1, rows, rows);
    float* d_src2 = cuda_matrix::device_alloc<float>(src2, rows, rows);
    float* d_dest = cuda_matrix::device_alloc<float>(dest, rows, rows);

    // Perform multiplication on device
    cuda_matrix::m_mul(d_src1, d_src2, d_dest, rows, rows, rows, rows);

    // Copy back the results from device to host
    cuda_matrix::to<float>(dest, d_dest, rows, rows, "cpu");

    // Free device memory
    cuda_matrix::device_free<float>(d_src1);
    cuda_matrix::device_free<float>(d_src2);
    cuda_matrix::device_free<float>(d_dest);

    for (int i = 0; i < mat_size; ++i)
    {
        if (dest[i] != ans[i])
        {
            std::cout << "Mismatch at index " << i << ": dest[" << i << "] = " 
                      << dest[i] << ", expected = " << ans[i] << std::endl;
        }
        EXPECT_EQ(dest[i], ans[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Scalar_Multiply_Single_Thread)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float ans[9] = {2, 4, 6, 8, 10, 12, 14, 16, 18};

    cpu_matrix::m_scalar_mul(src, 2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src[i], ans[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Scalar_Multiply_Multi_Thread)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float ans[9] = {2, 4, 6, 8, 10, 12, 14, 16, 18};

    #pragma omp parallel num_threads(2)
    cpu_matrix::m_scalar_mul(src, 2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src[i], ans[i]);
    }
}

TEST(Cuda_Matrix_Utils, Matrix_Scalar_Multiply)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float ans[9] = {2, 4, 6, 8, 10, 12, 14, 16, 18};

    // Allocate device memory
    float* d_src = cuda_matrix::device_alloc<float>(src, 3, 3);

    // Perform scalar multiplication on device
    cuda_matrix::m_scalar_mul(d_src, 2, 3, 3);

    // Copy back the results from device to host
    cuda_matrix::to<float>(src, d_src, 3, 3, "cpu");

    // Free device memory
    cuda_matrix::device_free<float>(d_src);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src[i], ans[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Transpose_Single_Thread)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float dest[9] = {0};
    float ans[9] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    cpu_matrix::m_transpose(src, dest, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(dest[i], ans[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Transpose_Multi_Thread)
{
    int threads = 2;
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float dest[9] = {0};
    float ans[9] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    #pragma omp parallel num_threads(threads)
    cpu_matrix::m_transpose(src, dest, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(dest[i], ans[i]);
    }
}

TEST(Cuda_Matrix_Utils, Matrix_Transpose)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float dest[9] = {0};
    float ans[9] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    // Allocate device memory
    float* d_src = cuda_matrix::device_alloc<float>(src, 3, 3);
    float* d_dest = cuda_matrix::device_alloc<float>(dest, 3, 3);

    // Perform transpose on device
    cuda_matrix::m_transpose(d_src, d_dest, 3, 3);

    // Copy back the results from device to host
    cuda_matrix::to<float>(dest, d_dest, 3, 3, "cpu");

    // Free device memory
    cuda_matrix::device_free<float>(d_src);
    cuda_matrix::device_free<float>(d_dest);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(dest[i], ans[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Hadamard_Single_Thread)
{
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[9] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    };
    float ans[9] = {9, 16, 21, 24, 25, 24, 21, 16, 9};

    cpu_matrix::m_hadamard(src1, src2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }
}

TEST(Cpu_Matrix_Utils, Matrix_Hadamard_Multi_Thread)
{
    int threads = 2;
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[9] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    };
    float ans[9] = {9, 16, 21, 24, 25, 24, 21, 16, 9};

    #pragma omp parallel num_threads(threads)
    cpu_matrix::m_hadamard(src1, src2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }
}

TEST(Cuda_Matrix_Utils, Matrix_Hadamard)
{
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[9] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    };
    float ans[9] = {9, 16, 21, 24, 25, 24, 21, 16, 9};

    // Allocate device memory
    float* d_src1 = cuda_matrix::device_alloc<float>(src1, 3, 3);
    float* d_src2 = cuda_matrix::device_alloc<float>(src2, 3, 3);

    // Perform Hadamard product on device
    cuda_matrix::m_hadamard(d_src1, d_src2, 3, 3);

    // Copy back the results from device to host
    cuda_matrix::to<float>(src1, d_src1, 3, 3, "cpu");

    // Free device memory
    cuda_matrix::device_free<float>(d_src1);
    cuda_matrix::device_free<float>(d_src2);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }
}

TEST(Cpu_Matrix_Utils, Index_To_One_Hot_Single_Thread)
{
    float src[3] = {0, 1, 2};
    float dest[9] = {0};
    float ans[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    cpu_matrix::m_index_to_one_hot(src, dest, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(dest[i], ans[i]);
    }
}

TEST(Cpu_Matrix_Utils, Index_To_One_Hot_Multi_Thread)
{
    int threads = 2;
    float src[3] = {0, 1, 2};
    float dest[9] = {0};
    float ans[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    #pragma omp parallel num_threads(threads)
    cpu_matrix::m_index_to_one_hot(src, dest, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(dest[i], ans[i]);
    }
}

TEST(Cuda_Matrix_Utils, Index_To_One_Hot)
{
    float src[3] = {0, 1, 2};
    float dest[9] = {0};
    float ans[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    // Allocate device memory
    float* d_src = cuda_matrix::device_alloc<float>(src, 1, 3);
    float* d_dest = cuda_matrix::device_alloc<float>(dest, 3, 3);

    // Perform index to one-hot on device
    cuda_matrix::m_index_to_one_hot(d_src, d_dest, 3, 3);

    // Copy back the results from device to host
    cuda_matrix::to<float>(dest, d_dest, 3, 3, "cpu");

    // Free device memory
    cuda_matrix::device_free<float>(d_src);
    cuda_matrix::device_free<float>(d_dest);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(dest[i], ans[i]);
    }
}


TEST(Cpu_Matrix_Utils, Softmax_Single_Thread)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float ans[9] = {
        0.09003057317038046, 0.24472847105479764, 0.6652409557748218,
        0.09003057317038046, 0.24472847105479764, 0.6652409557748218,
        0.09003057317038046, 0.24472847105479764, 0.6652409557748218
    };

    cpu_matrix::m_softmax(src, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(src[i], ans[i], 1e-6);
    }
}

TEST(Cpu_Matrix_Utils, Softmax_Multi_Thread)
{
    int threads = 2;
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float ans[9] = {
        0.09003057317038046, 0.24472847105479764, 0.6652409557748218,
        0.09003057317038046, 0.24472847105479764, 0.6652409557748218,
        0.09003057317038046, 0.24472847105479764, 0.6652409557748218
    };

    #pragma omp parallel num_threads(threads)
    cpu_matrix::m_softmax(src, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(src[i], ans[i], 1e-6);
    }
}

TEST(Cuda_Matrix_Utils, Softmax)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float ans[9] = {
        0.09003057317038046, 0.24472847105479764, 0.6652409557748218,
        0.09003057317038046, 0.24472847105479764, 0.6652409557748218,
        0.09003057317038046, 0.24472847105479764, 0.6652409557748218
    };

    // Allocate device memory
    float* d_src = cuda_matrix::device_alloc<float>(src, 3, 3);

    // Perform softmax on device
    cuda_matrix::m_softmax(d_src, 3, 3);

    // Copy back the results from device to host
    cuda_matrix::to<float>(src, d_src, 3, 3, "cpu");

    // Free device memory
    cuda_matrix::device_free<float>(d_src);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(src[i], ans[i], 1e-6);
    }
}

TEST(Cpu_Matrix_Utils, ReLU_Single_Thread)
{
    float src[9] = {
        1, -2, 3,
        -4, 5, -6,
        7, -8, 9
    };
    float ans[9] = {
        1, -0.2, 3,
        -0.4, 5, -0.6,
        7, -0.8, 9
    };

    cpu_matrix::m_Relu(src, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src[i], ans[i]);
    }
}

TEST(Cpu_Matrix_Utils, ReLU_Multi_Thread)
{
    int threads = 2;
    float src[9] = {
        1, -2, 3,
        -4, 5, -6,
        7, -8, 9
    };
    float ans[9] = {
        1, -0.2, 3,
        -0.4, 5, -0.6,
        7, -0.8, 9
    };

    #pragma omp parallel num_threads(threads)
    cpu_matrix::m_Relu(src, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src[i], ans[i]);
    }
}

TEST(Cuda_Matrix_Utils, ReLU)
{
    float src[9] = {
        1, -2, 3,
        -4, 5, -6,
        7, -8, 9
    };
    float ans[9] = {
        1, -0.2, 3,
        -0.4, 5, -0.6,
        7, -0.8, 9
    };

    // Allocate device memory
    float* d_src = cuda_matrix::device_alloc<float>(src, 3, 3);

    // Perform ReLU on device
    cuda_matrix::m_Relu(d_src, 3, 3);

    // Copy back the results from device to host
    cuda_matrix::to<float>(src, d_src, 3, 3, "cpu");

    // Free device memory
    cuda_matrix::device_free<float>(d_src);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src[i], ans[i]);
    }
}

TEST(NN_Utils, Cross_Entropy_Single_Thread)
{
    float src1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float src2[3] = {0, 1, 2};

    float ans = -3.8066624 / 3.0f;

    float loss = cross_entropy_loss(src1, src2, 3, 3);

    EXPECT_NEAR(loss, ans, 1e-6);
}

TEST(NN_Utils, Accuracy_Single_Thread)
{
    int src1[3] = {0, 2, 2};
    float src2[3] = {0, 1, 2};

    float ans = 2.0f / 3.0f;

    float acc = accuracy(src1, src2, 3);

    EXPECT_EQ(acc, ans);
}