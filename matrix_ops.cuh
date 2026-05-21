#pragma once

#include <cuda_runtime.h>

class MatrixOps
{
public:
    // multiplies matrix A by matrix B then adds bias vector to each output, this is what neural network layers do
    static void matmulAddBias(const float *A, const float *B, const float *bias,
                              float *C, int batch_size, int input_size, int output_size);

    // multiplies A with transpose of B, needed for backpropagation to calculate gradient updates
    static void matmulTranspose(const float *A, const float *B, float *C,
                                int batch_size, int input_size, int output_size);

    // basic arithmetic operations that happen element by element
    static void subtract(const float *A, const float *B, float *C, int size);
    static void addBias(float *input, const float *bias, int size);

    // functions for calculating loss and aggregating results across batches
    static void squaredError(const float *y_true, const float *y_pred, float *error, int size);
    static void sumReduce(const float *input, float *output, int size);
    static void sumRows(const float *input, float *output, int rows, int cols);

    // gradient descent updates for adjusting neural network parameters
    static void updateWeights(float *weights, const float *gradients,
                              float learning_rate, int size);
    static void updateBiases(float *biases, const float *gradients,
                             float learning_rate, int size);
};
