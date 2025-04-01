#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <filesystem>
#include <iostream>

/*
Function to read a matrix from a CSV file
INPUTS:
    filename: The name of the CSV file to read
    matrix: Pointer to the matrix to fill with data
    rows: Number of rows in the matrix
    cols: Number of columns in the matrix
OUTPUTS:
    Fills the matrix with data from the CSV file
*/
void read_matrix_from_file(const std::string& filename, float* matrix, int rows, int cols);

/*
Function to print a matrix to the console
INPUTS:
    matrix: Pointer to the matrix to print
    rows: Number of rows in the matrix
    cols: Number of columns in the matrix
OUTPUTS:    
    Prints the matrix to the console
*/
void print_matrix(const float* matrix, int rows, int cols);
