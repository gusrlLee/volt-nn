#include "gnn.cuh"
#include "matrix_ops.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

// cuda kernel for graph convolution layer - implements message passing between connected nodes
__global__ void graphConvolutionKernel(const float *node_features, const float *adjacency_matrix,
                                       const float *weights, const float *bias, float *output_features,
                                       int num_nodes, int feature_dim, int output_dim)
{
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (node_idx < num_nodes && feature_idx < output_dim)
    {
        float sum = 0.0f;

        // aggregate neighbor features by summing them up
        // adjacency matrix tells us which nodes are connected
        for (int neighbor = 0; neighbor < num_nodes; ++neighbor)
        {
            if (adjacency_matrix[node_idx * num_nodes + neighbor] > 0.0f)
            {
                for (int f = 0; f < feature_dim; ++f)
                {
                    sum += node_features[neighbor * feature_dim + f] *
                           weights[f * output_dim + feature_idx];
                }
            }
        }

        // add bias term for this output dimension
        output_features[node_idx * output_dim + feature_idx] = sum + bias[feature_idx];
    }
}

// cuda kernel for softmax activation - converts outputs into probability distribution
__global__ void softmaxKernel(const float *input, float *output, int num_nodes, int num_classes)
{
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_idx < num_nodes)
    {
        // find maximum value for numerical stability
        float max_val = input[node_idx * num_classes];
        for (int i = 1; i < num_classes; ++i)
        {
            max_val = fmaxf(max_val, input[node_idx * num_classes + i]);
        }

        // compute exponentials and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; ++i)
        {
            float exp_val = expf(input[node_idx * num_classes + i] - max_val);
            output[node_idx * num_classes + i] = exp_val;
            sum_exp += exp_val;
        }

        // normalize to get probabilities
        for (int i = 0; i < num_classes; ++i)
        {
            output[node_idx * num_classes + i] /= sum_exp;
        }
    }
}

// cuda kernel for sigmoid activation - bounded output for property prediction
__global__ void sigmoidKernel(const float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        // clip to prevent overflow
        float x = fmaxf(-500.0f, fminf(500.0f, input[idx]));
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

// cuda kernel for relu activation - used in hidden layers
__global__ void reluKernel(const float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// cuda kernel for computing cross entropy loss - used in classification tasks
__global__ void crossEntropyLossKernel(const float *predictions, const float *labels,
                                       float *loss, int num_nodes, int num_classes)
{
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_idx < num_nodes)
    {
        float node_loss = 0.0f;
        for (int i = 0; i < num_classes; ++i)
        {
            float pred = fmaxf(1e-15f, fminf(1.0f - 1e-15f, predictions[node_idx * num_classes + i]));
            node_loss += labels[node_idx * num_classes + i] * logf(pred);
        }
        loss[node_idx] = -node_loss;
    }
}

// cuda kernel for computing mean squared error loss - used in regression tasks
__global__ void mseLossKernel(const float *predictions, const float *labels,
                              float *loss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        float diff = predictions[idx] - labels[idx];
        loss[idx] = diff * diff;
    }
}

// cuda kernel for computing binary cross entropy loss - used in property prediction
__global__ void binaryCrossEntropyLossKernel(const float *predictions, const float *labels,
                                             float *loss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        float pred = fmaxf(1e-15f, fminf(1.0f - 1e-15f, predictions[idx]));
        loss[idx] = -(labels[idx] * logf(pred) + (1.0f - labels[idx]) * logf(1.0f - pred));
    }
}

// cuda kernel for computing gradients through graph convolution - implements graph-aware backpropagation
// similar to python's _backprop_graph_conv method
__global__ void graphConvolutionGradientKernel(const float *grad_output, const float *adjacency_matrix,
                                               const float *weights, float *grad_input, float *grad_weights,
                                               int num_nodes, int feature_dim, int output_dim)
{
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (node_idx < num_nodes && feature_idx < feature_dim)
    {
        float grad_sum = 0.0f;

        // gradient flows back through the graph structure
        for (int neighbor = 0; neighbor < num_nodes; ++neighbor)
        {
            if (adjacency_matrix[node_idx * num_nodes + neighbor] > 0.0f)
            {
                for (int out_dim = 0; out_dim < output_dim; ++out_dim)
                {
                    grad_sum += grad_output[neighbor * output_dim + out_dim] *
                                weights[feature_idx * output_dim + out_dim];
                }
            }
        }

        grad_input[node_idx * feature_dim + feature_idx] = grad_sum;
    }

    // compute weight gradients
    if (node_idx < feature_dim && feature_idx < output_dim)
    {
        float weight_grad = 0.0f;

        for (int n = 0; n < num_nodes; ++n)
        {
            for (int neighbor = 0; neighbor < num_nodes; ++neighbor)
            {
                if (adjacency_matrix[n * num_nodes + neighbor] > 0.0f)
                {
                    weight_grad += grad_output[n * output_dim + feature_idx];
                }
            }
        }

        grad_weights[node_idx * output_dim + feature_idx] = weight_grad;
    }
}

// cuda kernel for computing bias gradients
__global__ void biasGradientKernel(const float *grad_output, float *grad_bias,
                                   int num_nodes, int output_dim)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < output_dim)
    {
        float bias_grad = 0.0f;

        for (int n = 0; n < num_nodes; ++n)
        {
            bias_grad += grad_output[n * output_dim + out_idx];
        }

        grad_bias[out_idx] = bias_grad / num_nodes; // average over nodes
    }
}

// cuda kernel for relu derivative - used in backpropagation
__global__ void reluDerivativeKernel(const float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// cuda kernel for softmax derivative - used in classification backpropagation
__global__ void softmaxDerivativeKernel(const float *predictions, const float *labels,
                                        float *grad_output, int num_nodes, int num_classes)
{
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_idx < num_nodes)
    {
        for (int i = 0; i < num_classes; ++i)
        {
            grad_output[node_idx * num_classes + i] = predictions[node_idx * num_classes + i] -
                                                      labels[node_idx * num_classes + i];
        }
    }
}

// cuda kernel for sigmoid derivative - used in property prediction backpropagation
__global__ void sigmoidDerivativeKernel(const float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        float sigmoid_val = 1.0f / (1.0f + expf(-input[idx]));
        output[idx] = sigmoid_val * (1.0f - sigmoid_val);
    }
}

// cuda kernel for multiplying by 2 - used in MSE gradient computation
__global__ void multiplyByTwoKernel(float *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        data[idx] *= 2.0f;
    }
}

// constructor that sets up the graph neural network with the specified architecture and hyperparameters
GNN::GNN(int InputDim, const std::vector<int> &HiddenDims, int OutputDim, const std::string &TaskType, float LearningRate)
    : InputDim_(InputDim), HiddenDims_(HiddenDims), OutputDim_(OutputDim), TaskType_(TaskType), LearningRate_(LearningRate), IsTrained_(false)
{

    // build the gnn layers
    LayerSizes_.push_back(InputDim);
    for (int HiddenSize : HiddenDims)
    {
        LayerSizes_.push_back(HiddenSize);
    }
    LayerSizes_.push_back(OutputDim);

    // initialize weights and biases for each layer
    initializeWeights();
}

// cleanup gpu memory when the graph neural network is destroyed
GNN::~GNN()
{
    // free all gpu memory
    for (auto &W : DWeights_)
    {
        if (W)
            cudaFree(W);
    }
    for (auto &B : DBiases_)
    {
        if (B)
            cudaFree(B);
    }
}

// initialize weights and biases for all layers using xavier initialization
void GNN::initializeWeights()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    for (size_t i = 0; i < LayerSizes_.size() - 1; ++i)
    {
        int InputSize = LayerSizes_[i];
        int OutputSize = LayerSizes_[i + 1];

        // allocate gpu memory for weights
        float *DW;
        cudaMalloc(&DW, InputSize * OutputSize * sizeof(float));

        // initialize weights using xavier initialization
        std::vector<float> HostW(InputSize * OutputSize);
        float StdDev = std::sqrt(2.0f / InputSize);
        std::normal_distribution<float> Dist(0.0f, StdDev);

        for (auto &W : HostW)
        {
            W = Dist(gen);
        }

        cudaMemcpy(DW, HostW.data(),
                   InputSize * OutputSize * sizeof(float), cudaMemcpyHostToDevice);

        DWeights_.push_back(DW);

        // allocate and initialize biases to zero
        float *DB;
        cudaMalloc(&DB, OutputSize * sizeof(float));
        cudaMemset(DB, 0, OutputSize * sizeof(float));

        DBiases_.push_back(DB);
    }
}

// apply activation function based on task type and layer
void GNN::applyActivation(const float *Input, float *Output, const std::string &Activation, int Size)
{
    int BlockSize = 256;
    int NumBlocks = (Size + BlockSize - 1) / BlockSize;

    if (Activation == "relu")
    {
        reluKernel<<<NumBlocks, BlockSize>>>(Input, Output, Size);
    }
    else if (Activation == "sigmoid")
    {
        sigmoidKernel<<<NumBlocks, BlockSize>>>(Input, Output, Size);
    }
    else if (Activation == "softmax")
    {
        // softmax needs special handling for 2d arrays
        int NumNodes = Size / OutputDim_;
        int NumClasses = OutputDim_;
        int NodeBlocks = (NumNodes + BlockSize - 1) / BlockSize;
        softmaxKernel<<<NodeBlocks, BlockSize>>>(Input, Output, NumNodes, NumClasses);
    }
    else
    {
        // linear activation - just copy input to output
        cudaMemcpy(Output, Input, Size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();
}

// forward pass through entire gnn - processes node features through all graph convolution layers
void GNN::forward(const std::vector<std::vector<float>> &NodeFeatures,
                  const std::vector<std::vector<float>> &AdjacencyMatrix,
                  std::vector<std::vector<float>> &OutputFeatures)
{

    int NumNodes = NodeFeatures.size();
    int FeatureDim = NodeFeatures[0].size();

    // flatten input data for gpu processing
    std::vector<float> FlatFeatures(NumNodes * FeatureDim);
    std::vector<float> FlatAdjacency(NumNodes * NumNodes);

    for (int i = 0; i < NumNodes; ++i)
    {
        for (int j = 0; j < FeatureDim; ++j)
        {
            FlatFeatures[i * FeatureDim + j] = NodeFeatures[i][j];
        }
        for (int j = 0; j < NumNodes; ++j)
        {
            FlatAdjacency[i * NumNodes + j] = AdjacencyMatrix[i][j];
        }
    }

    // allocate gpu memory
    float *DFeatures, *DAdjacency, *DOutput;
    cudaMalloc(&DFeatures, NumNodes * FeatureDim * sizeof(float));
    cudaMalloc(&DAdjacency, NumNodes * NumNodes * sizeof(float));
    cudaMalloc(&DOutput, NumNodes * OutputDim_ * sizeof(float));

    // copy data to gpu
    cudaMemcpy(DFeatures, FlatFeatures.data(),
               NumNodes * FeatureDim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DAdjacency, FlatAdjacency.data(),
               NumNodes * NumNodes * sizeof(float), cudaMemcpyHostToDevice);

    // forward pass through gnn layers
    float *H = DFeatures;
    int CurrentDim = FeatureDim;

    // pass through hidden layers with relu activation
    for (size_t Layer = 0; Layer < LayerSizes_.size() - 1; ++Layer)
    {
        int OutputDim = LayerSizes_[Layer + 1];

        // graph convolution operation
        dim3 BlockSize(16, 16);
        dim3 GridSize((NumNodes + BlockSize.x - 1) / BlockSize.x,
                      (OutputDim + BlockSize.y - 1) / BlockSize.y);

        graphConvolutionKernel<<<GridSize, BlockSize>>>(
            H, DAdjacency, DWeights_[Layer], DBiases_[Layer], DOutput,
            NumNodes, CurrentDim, OutputDim);

        cudaDeviceSynchronize();

        // apply activation function
        if (Layer == LayerSizes_.size() - 2)
        {
            // final layer - apply task-specific activation
            if (TaskType_ == "classification")
            {
                applyActivation(DOutput, DOutput, "softmax", NumNodes * OutputDim);
            }
            else if (TaskType_ == "property_prediction")
            {
                applyActivation(DOutput, DOutput, "sigmoid", NumNodes * OutputDim);
            }
            // regression uses linear activation (no activation)
        }
        else
        {
            // hidden layers use relu activation
            applyActivation(DOutput, DOutput, "relu", NumNodes * OutputDim);
        }

        // update for next layer
        if (Layer < LayerSizes_.size() - 2)
        {
            if (H != DFeatures)
            {
                cudaFree(H);
            }
            H = DOutput;
            CurrentDim = OutputDim;

            // allocate new output for next layer
            cudaMalloc(&DOutput, NumNodes * LayerSizes_[Layer + 2] * sizeof(float));
        }
    }

    // copy final results back to host
    std::vector<float> FlatOutput(NumNodes * OutputDim_);
    cudaMemcpy(FlatOutput.data(), DOutput,
               NumNodes * OutputDim_ * sizeof(float), cudaMemcpyDeviceToHost);

    // reshape output
    OutputFeatures.resize(NumNodes);
    for (int i = 0; i < NumNodes; ++i)
    {
        OutputFeatures[i].resize(OutputDim_);
        for (int j = 0; j < OutputDim_; ++j)
        {
            OutputFeatures[i][j] = FlatOutput[i * OutputDim_ + j];
        }
    }

    // cleanup gpu memory
    cudaFree(DFeatures);
    cudaFree(DAdjacency);
    cudaFree(DOutput);
    if (H != DFeatures)
    {
        cudaFree(H);
    }
}

// calculate loss based on task type
float GNN::calculateLoss(const std::vector<std::vector<float>> &predictions,
                         const std::vector<std::vector<float>> &labels)
{
    int num_nodes = predictions.size();
    int pred_size = predictions[0].size();

    // flatten data for gpu processing
    std::vector<float> flat_predictions(num_nodes * pred_size);
    std::vector<float> flat_labels(num_nodes * pred_size);

    for (int i = 0; i < num_nodes; ++i)
    {
        for (int j = 0; j < pred_size; ++j)
        {
            flat_predictions[i * pred_size + j] = predictions[i][j];
            flat_labels[i * pred_size + j] = labels[i][j];
        }
    }

    // allocate gpu memory
    float *d_predictions, *d_labels, *d_loss;
    cudaMalloc(&d_predictions, num_nodes * pred_size * sizeof(float));
    cudaMalloc(&d_labels, num_nodes * pred_size * sizeof(float));
    cudaMalloc(&d_loss, num_nodes * sizeof(float));

    // copy data to gpu
    cudaMemcpy(d_predictions, flat_predictions.data(),
               num_nodes * pred_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, flat_labels.data(),
               num_nodes * pred_size * sizeof(float), cudaMemcpyHostToDevice);

    // compute loss based on task type
    int blockSize = 256;
    int numBlocks = (num_nodes + blockSize - 1) / blockSize;

    if (TaskType_ == "classification")
    {
        crossEntropyLossKernel<<<numBlocks, blockSize>>>(
            d_predictions, d_labels, d_loss, num_nodes, pred_size);
    }
    else if (TaskType_ == "regression")
    {
        mseLossKernel<<<numBlocks, blockSize>>>(
            d_predictions, d_labels, d_loss, num_nodes * pred_size);
    }
    else if (TaskType_ == "property_prediction")
    {
        binaryCrossEntropyLossKernel<<<numBlocks, blockSize>>>(
            d_predictions, d_labels, d_loss, num_nodes * pred_size);
    }

    cudaDeviceSynchronize();

    // compute mean loss
    std::vector<float> host_loss(num_nodes);
    cudaMemcpy(host_loss.data(), d_loss, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    float total_loss = 0.0f;
    for (float loss : host_loss)
    {
        total_loss += loss;
    }
    total_loss /= num_nodes;

    // cleanup
    cudaFree(d_predictions);
    cudaFree(d_labels);
    cudaFree(d_loss);

    return total_loss;
}

// train the gnn using proper graph-aware backpropagation
void GNN::train(const std::vector<std::vector<float>> &node_features,
                const std::vector<std::vector<float>> &adjacency_matrix,
                const std::vector<std::vector<float>> &labels, int epochs, bool verbose)
{

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // forward pass - get both predictions and activations
        std::vector<std::vector<float>> predictions;
        forward(node_features, adjacency_matrix, predictions);

        // calculate loss
        float current_loss = calculateLoss(predictions, labels);

        // print progress
        if (verbose && (epoch % 20 == 0 || epoch == epochs - 1))
        {
            std::cout << "epoch " << epoch << "/" << epochs << ", loss: " << current_loss << std::endl;
        }

        // compute gradients using sophisticated graph-aware backpropagation
        computeGraphGradients(node_features, adjacency_matrix, labels, predictions);
    }

    IsTrained_ = true;
}

// use the trained gnn to predict node features or classifications
std::vector<std::vector<float>> GNN::predict(const std::vector<std::vector<float>> &node_features,
                                             const std::vector<std::vector<float>> &adjacency_matrix)
{
    if (!IsTrained_)
    {
        std::cout << "model not trained yet." << std::endl;
        return {};
    }

    std::vector<std::vector<float>> predictions;
    forward(node_features, adjacency_matrix, predictions);
    return predictions;
}

// sophisticated graph-aware backpropagation implementation
// similar to python's _compute_graph_gradients method
void GNN::computeGraphGradients(const std::vector<std::vector<float>> &NodeFeatures,
                                const std::vector<std::vector<float>> &AdjacencyMatrix,
                                const std::vector<std::vector<float>> &Labels,
                                const std::vector<std::vector<float>> &Predictions)
{

    int NumNodes = NodeFeatures.size();
    int FeatureDim = NodeFeatures[0].size();

    // flatten data for gpu processing
    std::vector<float> FlatFeatures(NumNodes * FeatureDim);
    std::vector<float> FlatAdjacency(NumNodes * NumNodes);
    std::vector<float> FlatLabels(NumNodes * OutputDim_);
    std::vector<float> FlatPredictions(NumNodes * OutputDim_);

    for (int i = 0; i < NumNodes; ++i)
    {
        for (int j = 0; j < FeatureDim; ++j)
        {
            FlatFeatures[i * FeatureDim + j] = NodeFeatures[i][j];
        }
        for (int j = 0; j < NumNodes; ++j)
        {
            FlatAdjacency[i * NumNodes + j] = AdjacencyMatrix[i][j];
        }
        for (int j = 0; j < OutputDim_; ++j)
        {
            FlatLabels[i * OutputDim_ + j] = Labels[i][j];
            FlatPredictions[i * OutputDim_ + j] = Predictions[i][j];
        }
    }

    // allocate gpu memory
    float *DFeatures, *DAdjacency, *DLabels, *DPredictions;
    float *DGradOutput, *DGradInput, *DGradWeights, *DGradBiases;

    cudaMalloc(&DFeatures, NumNodes * FeatureDim * sizeof(float));
    cudaMalloc(&DAdjacency, NumNodes * NumNodes * sizeof(float));
    cudaMalloc(&DLabels, NumNodes * OutputDim_ * sizeof(float));
    cudaMalloc(&DPredictions, NumNodes * OutputDim_ * sizeof(float));
    cudaMalloc(&DGradOutput, NumNodes * OutputDim_ * sizeof(float));
    cudaMalloc(&DGradInput, NumNodes * FeatureDim * sizeof(float));

    // copy data to gpu
    cudaMemcpy(DFeatures, FlatFeatures.data(), NumNodes * FeatureDim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DAdjacency, FlatAdjacency.data(), NumNodes * NumNodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DLabels, FlatLabels.data(), NumNodes * OutputDim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DPredictions, FlatPredictions.data(), NumNodes * OutputDim_ * sizeof(float), cudaMemcpyHostToDevice);

    // compute output layer gradient based on task type
    int BlockSize = 256;
    int NumBlocks = (NumNodes + BlockSize - 1) / BlockSize;

    if (TaskType_ == "classification")
    {
        // softmax + cross-entropy gradient
        int ClassBlocks = (NumNodes + BlockSize - 1) / BlockSize;
        softmaxDerivativeKernel<<<ClassBlocks, BlockSize>>>(
            DPredictions, DLabels, DGradOutput, NumNodes, OutputDim_);
    }
    else if (TaskType_ == "regression")
    {
        // mse gradient: grad = 2 * (pred - true)
        // use MatrixOps subtract method
        MatrixOps::subtract(DPredictions, DLabels, DGradOutput, NumNodes * OutputDim_);
        // multiply by 2 for MSE gradient
        int RegBlocks = (NumNodes * OutputDim_ + BlockSize - 1) / BlockSize;
        multiplyByTwoKernel<<<RegBlocks, BlockSize>>>(DGradOutput, NumNodes * OutputDim_);
    }
    else if (TaskType_ == "property_prediction")
    {
        // sigmoid + binary cross-entropy gradient
        // use MatrixOps subtract method
        MatrixOps::subtract(DPredictions, DLabels, DGradOutput, NumNodes * OutputDim_);
    }

    cudaDeviceSynchronize();

    // backpropagate through layers
    float *CurrentGrad = DGradOutput;

    // backprop through final layer
    cudaMalloc(&DGradWeights, LayerSizes_[LayerSizes_.size() - 2] * OutputDim_ * sizeof(float));
    cudaMalloc(&DGradBiases, OutputDim_ * sizeof(float));

    dim3 GridSize((NumNodes + 15) / 16, (LayerSizes_[LayerSizes_.size() - 2] + 15) / 16);
    dim3 BlockSize2D(16, 16);

    graphConvolutionGradientKernel<<<GridSize, BlockSize2D>>>(
        CurrentGrad, DAdjacency, DWeights_[DWeights_.size() - 1], DGradInput, DGradWeights,
        NumNodes, LayerSizes_[LayerSizes_.size() - 2], OutputDim_);

    biasGradientKernel<<<NumBlocks, BlockSize>>>(CurrentGrad, DGradBiases, NumNodes, OutputDim_);

    // update final layer weights and biases
    updateWeightsAndBiases(DWeights_[DWeights_.size() - 1], DBiases_[DBiases_.size() - 1],
                           DGradWeights, DGradBiases, LayerSizes_[LayerSizes_.size() - 2], OutputDim_);

    // backprop through hidden layers
    for (int layer = LayerSizes_.size() - 2; layer > 0; --layer)
    {
        int InputDim = LayerSizes_[layer - 1];
        int OutputDim = LayerSizes_[layer];

        // apply relu derivative
        reluDerivativeKernel<<<NumBlocks, BlockSize>>>(CurrentGrad, CurrentGrad, NumNodes * OutputDim);

        // allocate gradients for this layer
        float *DGradWeightsLayer, *DGradBiasesLayer, *DGradInputLayer;
        cudaMalloc(&DGradWeightsLayer, InputDim * OutputDim * sizeof(float));
        cudaMalloc(&DGradBiasesLayer, OutputDim * sizeof(float));
        cudaMalloc(&DGradInputLayer, NumNodes * InputDim * sizeof(float));

        // compute gradients
        dim3 GridSizeLayer((NumNodes + 15) / 16, (InputDim + 15) / 16);
        graphConvolutionGradientKernel<<<GridSizeLayer, BlockSize2D>>>(
            CurrentGrad, DAdjacency, DWeights_[layer - 1], DGradInputLayer, DGradWeightsLayer,
            NumNodes, InputDim, OutputDim);

        biasGradientKernel<<<NumBlocks, BlockSize>>>(CurrentGrad, DGradBiasesLayer, NumNodes, OutputDim);

        // update weights and biases
        updateWeightsAndBiases(DWeights_[layer - 1], DBiases_[layer - 1],
                               DGradWeightsLayer, DGradBiasesLayer, InputDim, OutputDim);

        // prepare for next layer
        cudaFree(CurrentGrad);
        CurrentGrad = DGradInputLayer;

        // cleanup
        cudaFree(DGradWeightsLayer);
        cudaFree(DGradBiasesLayer);
    }

    // cleanup
    cudaFree(DFeatures);
    cudaFree(DAdjacency);
    cudaFree(DLabels);
    cudaFree(DPredictions);
    cudaFree(DGradOutput);
    cudaFree(DGradInput);
    cudaFree(DGradWeights);
    cudaFree(DGradBiases);
    if (CurrentGrad != DGradOutput)
    {
        cudaFree(CurrentGrad);
    }
}

// helper function to update weights and biases with computed gradients
void GNN::updateWeightsAndBiases(float *Weights, float *Biases,
                                 const float *GradWeights, const float *GradBiases,
                                 int InputDim, int OutputDim)
{
    int WeightSize = InputDim * OutputDim;
    int BiasSize = OutputDim;

    // update weights
    std::vector<float> HostWeights(WeightSize);
    std::vector<float> HostGradWeights(WeightSize);
    cudaMemcpy(HostWeights.data(), Weights, WeightSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(HostGradWeights.data(), GradWeights, WeightSize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < WeightSize; ++i)
    {
        HostWeights[i] -= LearningRate_ * HostGradWeights[i];
    }

    cudaMemcpy(Weights, HostWeights.data(), WeightSize * sizeof(float), cudaMemcpyHostToDevice);

    // update biases
    std::vector<float> HostBiases(BiasSize);
    std::vector<float> HostGradBiases(BiasSize);
    cudaMemcpy(HostBiases.data(), Biases, BiasSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(HostGradBiases.data(), GradBiases, BiasSize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < BiasSize; ++i)
    {
        HostBiases[i] -= LearningRate_ * HostGradBiases[i];
    }

    cudaMemcpy(Biases, HostBiases.data(), BiasSize * sizeof(float), cudaMemcpyHostToDevice);
}
