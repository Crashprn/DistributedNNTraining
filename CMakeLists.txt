cmake_minimum_required(VERSION 3.19)
set(PROJECT_NAME NNTraining)
set(UNIT_TEST_RUNNER UnitTestRunner)

project(${PROJECT_NAME})


set (SOURCE_FILES 
    "${PROJECT_SOURCE_DIR}/src/matrix_utils.cpp"
    "${PROJECT_SOURCE_DIR}/src/nn_utils.cpp" 
    "${PROJECT_SOURCE_DIR}/src/data_utils.cpp"
    "${PROJECT_SOURCE_DIR}/src/nn_trainer.cpp"
    )
set (HEADER_FILES
    "${PROJECT_SOURCE_DIR}/src/matrix_utils.hpp"
    "${PROJECT_SOURCE_DIR}/src/nn_utils.hpp"
    "${PROJECT_SOURCE_DIR}/src/data_utils.hpp"
    "${PROJECT_SOURCE_DIR}/src/nn_trainer.hpp"
    )
set (TEST_FILES "${PROJECT_SOURCE_DIR}/test.cpp")

find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} main.cpp)
add_executable(${UNIT_TEST_RUNNER} ${HEADER_FILES} ${SOURCE_FILES} ${TEST_FILES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${UNIT_TEST_RUNNER} PROPERTY CXX_STANDARD 20)

if(OpenMP_CXX_FOUND)
    message("OpenMP found")
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(${UNIT_TEST_RUNNER} PUBLIC OpenMP::OpenMP_CXX)
else()
    message("OpenMP not found")
endif()

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


if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /permissive-)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Wconversion -Wuninitialized -O3)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)
endif()
