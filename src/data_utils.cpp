#include "data_utils.hpp"

void read_matrix_from_file(const std::string& filename, float* matrix, int rows, int cols)
{
    std::ifstream input_stream;

    std::filesystem::path path = std::filesystem::current_path();
    path /= filename;
    std::string full_path = path.string();

    input_stream.open(full_path);
    if (input_stream.fail())
    {
        throw std::runtime_error("Could not open file: " + full_path);
    }

    std::string line;

    for (int i = 0; i < rows; ++i)
    {
        std::getline(input_stream, line, '\n');

        // Get all values before a comma
        decltype(line.size()) found = 0;
        int found_nums = 0;
        for (long unsigned int idx = 0; idx < line.size(); ++idx)
        {   
            if (line[idx] == ',')
            {
                std::string value = line.substr(found, idx - found);
                matrix[i * cols + found_nums] = std::stof(value);
                found = idx + 1;
                found_nums++;
            }
        }
        // Get the last value after the last comma
        std::string value = line.substr(found, line.size() - found);
        matrix[i * cols + found_nums] = std::stof(value);
    }
    input_stream.close();
}

void print_matrix(const float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}