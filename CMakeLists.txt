cmake_minimum_required(VERSION 3.19)
set(PROJECT_NAME NNTraining)
set(UNIT_TEST_RUNNER UnitTestRunner)
set(CMAKE_CUDA_ARCHITECTURES "native")

project(${PROJECT_NAME} LANGUAGES C CXX CUDA)

# Setting Source and Header Files
set (SOURCE_FILES 
    "${PROJECT_SOURCE_DIR}/src/nn_utils.cpp" 
    "${PROJECT_SOURCE_DIR}/src/data_utils.cpp"
    "${PROJECT_SOURCE_DIR}/src/nn_trainer.cpp"
    "${PROJECT_SOURCE_DIR}/src/matrix_utils.cpp"
    )
set (HEADER_FILES
    "${PROJECT_SOURCE_DIR}/src/nn_utils.hpp"
    "${PROJECT_SOURCE_DIR}/src/data_utils.hpp"
    "${PROJECT_SOURCE_DIR}/src/nn_trainer.hpp"
    "${PROJECT_SOURCE_DIR}/src/matrix_utils.hpp"
        )
set (TEST_FILES "${PROJECT_SOURCE_DIR}/test.cpp")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(kernels
    "${PROJECT_SOURCE_DIR}/src/c_matrix_utils.cu"
    "${PROJECT_SOURCE_DIR}/src/c_matrix_utils.cuh"
)

target_compile_features(kernels PUBLIC cxx_std_20)
set_target_properties(kernels
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

# Creating main and unit test runner executables
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} main.cpp)
add_executable(${UNIT_TEST_RUNNER} ${HEADER_FILES} ${SOURCE_FILES} ${TEST_FILES})

# Set the C++ standard for the main project and unit test runner
set_property(TARGET ${PROJECT_NAME} PROPERTY
    CXX_STANDARD 20
    )
set_property(TARGET ${UNIT_TEST_RUNNER} PROPERTY 
    CXX_STANDARD 20
    )

target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart kernels)
target_link_libraries(${UNIT_TEST_RUNNER} PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart kernels)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

target_link_libraries(${UNIT_TEST_RUNNER} PRIVATE gtest gtest_main)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wconversion -O1)
