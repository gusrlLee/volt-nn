#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>

class Utils
{
public:
    // handy functions for debugging cuda programs and making sure stuff is working
    static void checkCudaError(cudaError_t error, const std::string &message);

    // print out info about the gpu so we know what hardware were working with
    static void printDeviceInfo();

    // generate random data for testing neural networks when we dont have real data
    static void setRandomSeed(unsigned int seed);
    static std::vector<float> generateRandomData(int size, float min_val = -1.0f, float max_val = 1.0f);
    static std::vector<std::vector<float>> generateRandomMatrix(int rows, int cols,
                                                                float min_val = -1.0f, float max_val = 1.0f);

    // figure out how well our model is performing
    static float calculateAccuracy(const std::vector<float> &predictions, const std::vector<float> &labels);

    // tools for printing out arrays and matrices so we can see what's going on
    static void printMatrix(const std::vector<std::vector<float>> &matrix, const std::string &name = "Matrix");
    static void printVector(const std::vector<float> &vec, const std::string &name = "Vector");
};
