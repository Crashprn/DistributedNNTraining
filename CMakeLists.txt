cmake_minimum_required(VERSION 3.12)
set(PROJECT_NAME NNTraining)

project(${PROJECT_NAME})


set (SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/matrix_utils.cpp" "${PROJECT_SOURCE_DIR}/src/nn_utils.cpp" "${PROJECT_SOURCE_DIR}/src/data_utils.cpp" "${PROJECT_SOURCE_DIR}/src/nn_trainer.cpp")
set (HEADER_FILES "${PROJECT_SOURCE_DIR}/src/matrix_utils.hpp" "${PROJECT_SOURCE_DIR}/src/nn_utils.hpp" "${PROJECT_SOURCE_DIR}/src/data_utils.hpp" "${PROJECT_SOURCE_DIR}/src/nn_trainer.hpp")

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} main.cpp)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /permissive-)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Wconversion -Wuninitialized -O1)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)
endif()

if(OpenMP_CXX_FOUND)
    target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
