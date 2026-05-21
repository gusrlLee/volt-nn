#ifndef VOLT_NEURAL_NETWORK_GRAPH_NEURAL_NETWORK_CUDA_HEADER
#define VOLT_NEURAL_NETWORK_GRAPH_NEURAL_NETWORK_CUDA_HEADER

#include <vector>
#include <iostream>
#include <cuda_runtime.h>

class GNN
{
public:
    // Constructor that sets up the graph neural network with the specified architecture and hyperparameters
    GNN(int input_dim, const std::vector<int> &hidden_dim, int output_dim, const std::string &task_type, float learning_late = 0.01f);

    // Clean gpu memory when the graph neural network is destoryed memory
    ~GNN();

    void train(
        const std::vector<std::vector<float>> &node_features,
        const std::vector<std::vector<float>> &adjacency_matrix,
        const std::vector<std::vector<float>> &labels,
        int epochs = 100,
        bool verbose = true);

    // Use the trained gnn to predict node features or classifications, etc.
    std::vector<std::vector<float>> predict(
        const std::vector<std::vector<float>> &node_features,
        const std::vector<std::vector<float>> &adjacency_matrix);

    // forward pass through the gnn (made public for testing and inspection)
    void forward(
        const std::vector<std::vector<float>> &node_features,
        const std::vector<std::vector<float>> &adjacency_matrix,
        std::vector<std::vector<float>> &output_features);

    // calculate loss based on task type
    float calculateLoss(
        const std::vector<std::vector<float>> &predictions,
        const std::vector<std::vector<float>> &labels);

    const std::vector<int> &getLayerSizes() const { return m_LayerSizes; }
    const std::string &getTaskType() const { return m_TaskType; }
    float getLearningRate() const { return m_LearningRate; }
    bool isTrained() const { return m_IsTrained; }

private:
    int m_InputDim;
    std::vector<int> m_HiddenDims;
    int m_OutputDim;
    std::string m_TaskType;
    float m_LearningRate;
    bool m_IsTrained;

    std::vector<int> m_LayerSizes;

    std::vector<float *> m_DWeights;
    std::vector<float *> m_DBiases;

    void initWeights();
    void applyActivation(const float *input, float *output, const std::string &activation, int size);
    void computeGraphGradient(
        const std::vector<std::vector<float>> &node_features,
        const std::vector<std::vector<float>> &adjacency_matrix,
        const std::vector<std::vector<float>> &labels,
        const std::vector<std::vector<float>> &predictions);
    void updateWeightsAndBiases(
        float *weights,
        float *biases,
        const float *grad_weights,
        const float *grad_biases,
        int input_dim,
        int output_dim);
};

#endif // VOLT_NEURAL_NETWORK_GRAPH_NEURAL_NETWORK_CUDA_HEADER