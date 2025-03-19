#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <filesystem>
#include <iostream>

void read_matrix_from_file(const std::string& filename, float* matrix, int rows, int cols);
void print_matrix(const float* matrix, int rows, int cols);
