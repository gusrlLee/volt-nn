#include "gnn.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

GNN::GNN(int input_dim, const std::vector<int>& hidden_dim, int output_dim, const std::string& task_type, float learning_late)
    : m_InputDim(input_dim), m_HiddenDims(hidden_dim), m_OutputDim(output_dim), m_TaskType(task_type), m_LearningRate(learning_late), m_IsTrained(false)
{
    m_LayerSizes.push_back(input_dim);
    for (int hidden_size : hidden_dim)
    {
        m_LayerSizes.push_back(hidden_size);
    }

    m_LayerSizes.push_back(output_dim);

    initWeights();
}

GNN::~GNN()
{
    for (auto& W : m_DWeights)
    {
        if (W) cudaFree(W);
    }
    
    for (auto& B : m_DBiases)
    {
        if (B) cudaFree(B);
    }
}

void GNN::train(
    const std::vector<std::vector<float>>& node_features,
    const std::vector<std::vector<float>>& adjacency_matrix,
    const std::vector<std::vector<float>>& labels, 
    int epochs = 100,
    bool verbose = true
)
{

}

void GNN::initWeights()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    // -1 becuase output size
    for (size_t i = 0; i < m_LayerSizes.size() - 1; i++)
    {

    }
}