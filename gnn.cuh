#pragma once

#include <vector>
#include <string>
#include <stdexcept>

class GNN {
public:
    // constructor that sets up the graph neural network with the specified architecture and hyperparameters
    // similar to python version: GNN(input_dim, hidden_dims, output_dim, task_type, learning_rate)
    GNN(int InputDim, const std::vector<int>& HiddenDims, int OutputDim, 
        const std::string& TaskType = "classification", float LearningRate = 0.01f);
    
    // cleanup gpu memory when the graph neural network is destroyed
    ~GNN();
    
    // train the gnn on graph data using message passing and backpropagation
    // similar to python version: gnn.train(node_features, adjacency_matrix, labels, epochs, verbose)
    void train(const std::vector<std::vector<float>>& NodeFeatures,
              const std::vector<std::vector<float>>& AdjacencyMatrix,
              const std::vector<std::vector<float>>& Labels, int Epochs = 100, bool Verbose = true);
    
    // use the trained gnn to predict node features or classifications
    // similar to python version: gnn.predict(node_features, adjacency_matrix)
    std::vector<std::vector<float>> predict(const std::vector<std::vector<float>>& NodeFeatures,
                                          const std::vector<std::vector<float>>& AdjacencyMatrix);
    
    // forward pass through the gnn (made public for testing and inspection)
    // similar to python version: gnn.forward(node_features, adjacency_matrix)
    void forward(const std::vector<std::vector<float>>& NodeFeatures,
                const std::vector<std::vector<float>>& AdjacencyMatrix,
                std::vector<std::vector<float>>& OutputFeatures);
    
    // calculate loss based on task type
    // similar to python version: gnn._calculate_loss(y_true, y_pred)
    float calculateLoss(const std::vector<std::vector<float>>& Predictions,
                       const std::vector<std::vector<float>>& Labels);
    
    // functions to inspect the gnns current configuration and training state
    // similar to python version properties
    const std::vector<int>& getLayerSizes() const { return LayerSizes_; }
    const std::string& getTaskType() const { return TaskType_; }
    float getLearningRate() const { return LearningRate_; }
    bool isTrained() const { return IsTrained_; }

private:
    // architecture parameters - similar to python version
    int InputDim_;
    std::vector<int> HiddenDims_;
    int OutputDim_;
    std::string TaskType_;
    float LearningRate_;
    bool IsTrained_;
    
    // layer sizes for internal use
    std::vector<int> LayerSizes_;
    
    // pointers to gpu memory for storing learnable parameters that get updated during training
    // similar to python version: self.w, self.b
    std::vector<float*> DWeights_;
    std::vector<float*> DBiases_;
    
    // internal functions that implement the core graph neural network operations
    // similar to python version methods
    void initializeWeights();
    void applyActivation(const float* Input, float* Output, const std::string& Activation, int Size);
    
    // sophisticated backpropagation methods
    void computeGraphGradients(const std::vector<std::vector<float>>& NodeFeatures,
                              const std::vector<std::vector<float>>& AdjacencyMatrix,
                              const std::vector<std::vector<float>>& Labels,
                              const std::vector<std::vector<float>>& Predictions);
    void updateWeightsAndBiases(float* Weights, float* Biases, 
                               const float* GradWeights, const float* GradBiases,
                               int InputDim, int OutputDim);
};



