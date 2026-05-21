#include "utils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <random>

void Utils::checkCudaError(cudaError_t error, const std::string &message)
{
    // if any cuda function returned an error, print it out and crash the program so we can fix it
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA operation failed");
    }
}

void Utils::printDeviceInfo()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "CUDA Device Count: " << deviceCount << std::endl;

    // loop through each gpu we have and print out all its specs so we know what were working with
    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "\nDevice " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Thread Dimensions: [" << deviceProp.maxThreadsDim[0]
                  << ", " << deviceProp.maxThreadsDim[1]
                  << ", " << deviceProp.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max Grid Dimensions: [" << deviceProp.maxGridSize[0]
                  << ", " << deviceProp.maxGridSize[1]
                  << ", " << deviceProp.maxGridSize[2] << "]" << std::endl;
    }
}

void Utils::setRandomSeed(unsigned int seed)
{
    // set the seed for cpus random number generator so we get reproducible results
    srand(seed);
    // note: for gpu random numbers we would need to use curand library instead
}

std::vector<float> Utils::generateRandomData(int size, float min_val, float max_val)
{
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);

    // fill the vector with random numbers between min_val and max_val for testing
    for (auto &value : data)
    {
        value = dist(gen);
    }

    return data;
}

std::vector<std::vector<float>> Utils::generateRandomMatrix(int rows, int cols, float min_val, float max_val)
{
    std::vector<std::vector<float>> matrix(rows);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);

    // create a 2d matrix filled with random values for testing neural networks
    for (int i = 0; i < rows; ++i)
    {
        matrix[i].resize(cols);
        for (int j = 0; j < cols; ++j)
        {
            matrix[i][j] = dist(gen);
        }
    }

    return matrix;
}

float Utils::calculateAccuracy(const std::vector<float> &predictions, const std::vector<float> &labels)
{
    if (predictions.size() != labels.size())
    {
        throw std::invalid_argument("Predictions and labels must have same size");
    }

    int correct = 0;
    // count how many predictions are close enough to the true labels to be considered correct
    for (size_t i = 0; i < predictions.size(); ++i)
    {
        if (std::abs(predictions[i] - labels[i]) < 0.5f)
        {
            correct++;
        }
    }

    // return the percentage of correct predictions
    return static_cast<float>(correct) / predictions.size();
}

void Utils::printMatrix(const std::vector<std::vector<float>> &matrix, const std::string &name)
{
    std::cout << name << ":" << std::endl;
    // print out each row of the matrix with nice formatting for debugging
    for (const auto &row : matrix)
    {
        for (const auto &value : row)
        {
            std::cout << std::fixed << std::setprecision(4) << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Utils::printVector(const std::vector<float> &vec, const std::string &name)
{
    std::cout << name << ": [";
    // print out the vector elements in a nice comma-separated list for debugging
    for (size_t i = 0; i < vec.size(); ++i)
    {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i < vec.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}
