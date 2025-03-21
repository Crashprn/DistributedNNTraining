#include "src/matrix_utils.hpp"
#include "src/nn_utils.hpp"

#include <gtest/gtest.h>
#include <omp.h>

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(Matrix_Utils, Matrix_Copy_Single_Thread)
{
    float src[6] = {1, 2, 3, 4, 5, 6};
    float dest_v[6] = {0, 0, 0, 0, 0, 0};
    float dest_m[6] = {0, 0, 0, 0, 0, 0};
    m_copy(src, dest_v, 1, 6);
    m_copy(src, dest_m, 2, 3);
    for (int i = 0; i < 6; ++i)
    {
        EXPECT_EQ(src[i], dest_v[i]);
        EXPECT_EQ(src[i], dest_m[i]);
    }
}

TEST(Matrix_Utils, Matrix_Copy_Multi_Thread)
{
    int num_threads = 2;
    float src[6] = {1, 2, 3, 4, 5, 6};
    float dest_v[6] = {0, 0, 0, 0, 0, 0};
    float dest_m[6] = {0, 0, 0, 0, 0, 0};
    #pragma omp parallel num_threads(num_threads)
    {
        m_copy(src, dest_v, 1, 6);
        m_copy(src, dest_m, 2, 3);
    }

    for (int i = 0; i < 6; ++i)
    {
        EXPECT_EQ(src[i], dest_v[i]);
        EXPECT_EQ(src[i], dest_m[i]);
    }
}

TEST(Matrix_Utils, Matrix_Row_Copy_Single_Thread)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float dest[9] = {0};

    for (int i = 0; i < 3; ++i)
    {
        m_copy_row(src, dest, 2-i, i, 3, 3, 3);
    }

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_EQ(src[(2-i)*3 + j], dest[i*3 + j]);
        }
    }
}

TEST(Matrix_Utils, Matrix_Row_Copy_Multi_Thread)
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
        m_copy_row(src, dest, 2-i, i, 3, 3, 3);
    }

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_EQ(src[(2-i)*3 + j], dest[i*3 + j]);
        }
    }
}

TEST(Matrix_Utils, Matrix_Sum_Single_Thread)
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

    m_sum(src, dest1, 3, 3, 0);
    m_sum(src, dest2, 3, 3, 1);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(dest1[i], ans_axis_0[i]);
        EXPECT_EQ(dest2[i], ans_axis_1[i]);
    }
}

TEST(Matrix_Utils, Matrix_Sum_Multi_Thread)
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
    m_sum(src, dest1, 3, 3, 0);
    m_sum(src, dest2, 3, 3, 1);
    }

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(dest1[i], ans_axis_0[i]);
        EXPECT_EQ(dest2[i], ans_axis_1[i]);
    }
}

TEST(Matrix_Utils, Matrix_Argmax_Single_Thread)
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

    m_argmax(src, dest1, 3, 3, 0);
    m_argmax(src, dest2, 3, 3, 1);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(dest1[i], ans_axis_0[i]);
        EXPECT_EQ(dest2[i], ans_axis_1[i]);
    }
}

TEST(Matrix_Utils, Matrix_Argmax_Multi_Thread)
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
    m_argmax(src, dest1, 3, 3, 0);
    m_argmax(src, dest2, 3, 3, 1);
    }

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(dest1[i], ans_axis_0[i]);
        EXPECT_EQ(dest2[i], ans_axis_1[i]);
    }
}

TEST(Matrix_Utils, Matrix_Add_Single_Thread)
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

    m_add(src1, src2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }
}

TEST(Matrix_Utils, Matrix_Add_Multi_Thread)
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
    float ans[9] = {10, 10, 10, 10, 10, 10, 10, 10, 10};

    #pragma omp parallel num_threads(threads)
    m_add(src1, src2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }
}

TEST(Matrix_Utils, Matrix_Add_Vector_Single_Thread)
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

    m_add_v(src1, v_1, 3, 3, 3, 1);
    m_add_v(src2, v_2, 3, 3, 1, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans1[i]);
        EXPECT_EQ(src2[i], ans2[i]);
    }
    
}

TEST(Matrix_Utils, Matrix_Add_Vector_Multi_Thread)
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
    m_add_v(src1, v_1, 3, 3, 3, 1);
    m_add_v(src2, v_2, 3, 3, 1, 3);
    }

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans1[i]);
        EXPECT_EQ(src2[i], ans2[i]);
    }
    
}

TEST(Matrix_Utils, Matrix_Subtract_Single_Thread)
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

    m_sub(src1, src2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }
}

TEST(Matrix_Utils, Matrix_Subtract_Multi_Thread)
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
    m_sub(src1, src2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }
}

TEST(Matrix_Utils, Matrix_Multiply_Single_Thread)
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

    m_mul(src1, src2, dest, 3, 3, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(dest[i], ans[i]);
    }
}

TEST(Matrix_Utils, Matrix_Multiply_Multi_Thread)
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
    m_mul(src1, src2, dest, 3, 3, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(dest[i], ans[i]);
    }
}

TEST(Matrix_Utils, Matrix_Scalar_Multiply_Single_Thread)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float ans[9] = {2, 4, 6, 8, 10, 12, 14, 16, 18};

    m_scalar_mul(src, 2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src[i], ans[i]);
    }
}

TEST(Matrix_Utils, Matrix_Scalar_Multiply_Multi_Thread)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float ans[9] = {2, 4, 6, 8, 10, 12, 14, 16, 18};

    #pragma omp parallel num_threads(2)
    m_scalar_mul(src, 2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src[i], ans[i]);
    }
}

TEST(Matrix_Utils, Matrix_Transpose_Single_Thread)
{
    float src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float dest[9] = {0};
    float ans[9] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    m_transpose(src, dest, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(dest[i], ans[i]);
    }
}

TEST(Matrix_Utils, Matrix_Transpose_Multi_Thread)
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
    m_transpose(src, dest, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(dest[i], ans[i]);
    }
}

TEST(Matrix_Utils, Matrix_Hadamard_Single_Thread)
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

    m_hadamard(src1, src2, 3, 3);

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_EQ(src1[i], ans[i]);
    }
}