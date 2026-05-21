#include "matrix_ops.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// kernel that multiplies two matrices together and adds a bias vector to each output element
__global__ void matmulAddBiasKernel(const float* A, const float* B, const float* bias,
                                   float* C, int m, int n, int k) {
    // figure out which row and column this thread is supposed to compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // make sure we dont try to access memory outside our matrix bounds
    if (row < m && col < n) {
        // start with zero and add up all the multiplications from this row times this column
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        // add the bias term for this column before storing the result
        C[row * n + col] = sum + bias[col];
    }
}

// kernel that multiplies matrix A with transpose of matrix B, used heavily in neural network backpropagation
__global__ void matmulTransposeKernel(const float* A, const float* B, float* C,
                                     int m, int n, int k) {
    // figure out which row and column this thread should compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        // matrix B is already stored in transposed format, so we access it differently
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[col * k + i];
        }
        C[row * n + col] = sum;
    }
}

// simple kernel that subtracts corresponding elements from two arrays one by one
__global__ void subtractKernel(const float* A, const float* B, float* C, int size) {
    // each thread handles one subtraction operation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        C[idx] = A[idx] - B[idx];
    }
}

// kernel that adds bias terms to input values, used after matrix multiplication in neural networks
__global__ void addBiasKernel(const float* input, const float* bias, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // this calculation maps the input index to the correct bias index for the current batch
        output[idx] = input[idx] + bias[idx % (size / (blockIdx.x + 1))];
    }
}

// kernel that calculates the squared difference between predicted and actual values for loss computation
__global__ void squaredErrorKernel(const float* y_true, const float* y_pred, float* error, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // compute the difference and then square it to get the error magnitude
        float diff = y_true[idx] - y_pred[idx];
        error[idx] = diff * diff;
    }
}

// kernel that sums up all elements in an array using parallel reduction, gets called when we need to add everything up
__global__ void sumReduceKernel(const float* input, float* output, int size) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // each thread loads one value from global memory into shared memory
    sdata[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();
    
    // do a tree-like reduction where we pair up threads and add their values together
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // only the first thread in each block writes its result back to global memory
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// kernel that adds up all values in each column of a matrix, useful for gradient calculations
__global__ void sumRowsKernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols) {
        // sum up all the values in this column across all rows
        float sum = 0.0f;
        for (int row = 0; row < rows; ++row) {
            sum += input[row * cols + col];
        }
        output[col] = sum;
    }
}

// kernel that updates weight values using gradient descent, subtracts learning_rate * gradient from each weight
__global__ void updateWeightsKernel(float* weights, const float* gradients, 
                                   float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // move weights in the opposite direction of the gradient to minimize loss
        weights[idx] -= learning_rate * gradients[idx];
    }
}

// kernel that updates bias values using gradient descent, similar to weights but these are scalars
__global__ void updateBiasesKernel(float* biases, const float* gradients,
                                  float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // apply gradient descent to bias terms to reduce error
        biases[idx] -= learning_rate * gradients[idx];
    }
}

void MatrixOps::matmulAddBias(const float* A, const float* B, const float* bias,
                             float* C, int batch_size, int input_size, int output_size) {
    // set up 2d thread blocks to handle the matrix dimensions efficiently
    dim3 blockSize(16, 16);
    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x,
                  (batch_size + blockSize.y - 1) / blockSize.y);
    
    // launch the kernel to do matrix multiplication with bias addition
    matmulAddBiasKernel<<<gridSize, blockSize>>>(
        A, B, bias, C, batch_size, output_size, input_size
    );
    
    // wait for all threads to finish before returning
    cudaDeviceSynchronize();
}

void MatrixOps::matmulTranspose(const float* A, const float* B, float* C,
                              int batch_size, int input_size, int output_size) {
    // 2d grid setup for efficient matrix processing
    dim3 blockSize(16, 16);
    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x,
                  (batch_size + blockSize.y - 1) / blockSize.y);
    
    // launch transpose multiplication kernel for backpropagation gradient calculations
    matmulTransposeKernel<<<gridSize, blockSize>>>(
        A, B, C, batch_size, output_size, input_size
    );
    
    cudaDeviceSynchronize();
}

void MatrixOps::subtract(const float* A, const float* B, float* C, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // simple element-wise subtraction across the arrays
    subtractKernel<<<numBlocks, blockSize>>>(A, B, C, size);
    cudaDeviceSynchronize();
}

void MatrixOps::addBias(float* input, const float* bias, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // add bias terms to the input, modifying it in place
    addBiasKernel<<<numBlocks, blockSize>>>(input, bias, input, size);
    cudaDeviceSynchronize();
}

void MatrixOps::squaredError(const float* y_true, const float* y_pred, float* error, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // calculate squared differences for each prediction vs actual pair
    squaredErrorKernel<<<numBlocks, blockSize>>>(y_true, y_pred, error, size);
    cudaDeviceSynchronize();
}

void MatrixOps::sumReduce(const float* input, float* output, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // sum up all values in the array using parallel reduction with shared memory
    sumReduceKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(input, output, size);
    cudaDeviceSynchronize();
}

void MatrixOps::sumRows(const float* input, float* output, int rows, int cols) {
    int blockSize = 256;
    int numBlocks = (cols + blockSize - 1) / blockSize;
    
    // sum across rows for each column, used for gradient calculations
    sumRowsKernel<<<numBlocks, blockSize>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

void MatrixOps::updateWeights(float* weights, const float* gradients, 
                             float learning_rate, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // apply gradient descent update to weight values
    updateWeightsKernel<<<numBlocks, blockSize>>>(weights, gradients, learning_rate, size);
    cudaDeviceSynchronize();
}

void MatrixOps::updateBiases(float* biases, const float* gradients,
                            float learning_rate, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // apply gradient descent update to bias values
    updateBiasesKernel<<<numBlocks, blockSize>>>(biases, gradients, learning_rate, size);
    cudaDeviceSynchronize();
}



