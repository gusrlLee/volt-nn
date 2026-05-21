#include "gnn.cuh"
#include "utils.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>

// function to create sample graph data for testing the gnn
std::vector<std::vector<float>> createSampleNodeFeatures(int NumNodes, int FeatureDim)
{
    std::vector<std::vector<float>> NodeFeatures(NumNodes);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < NumNodes; ++i)
    {
        NodeFeatures[i].resize(FeatureDim);
        for (int j = 0; j < FeatureDim; ++j)
        {
            NodeFeatures[i][j] = dist(gen);
        }
    }

    return NodeFeatures;
}

// function to create a random adjacency matrix for testing
std::vector<std::vector<float>> createSampleAdjacencyMatrix(int NumNodes, float ConnectionProb = 0.3f)
{
    std::vector<std::vector<float>> AdjacencyMatrix(NumNodes, std::vector<float>(NumNodes, 0.0f));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < NumNodes; ++i)
    {
        for (int j = i + 1; j < NumNodes; ++j)
        {
            if (dist(gen) < ConnectionProb)
            {
                AdjacencyMatrix[i][j] = 1.0f;
                AdjacencyMatrix[j][i] = 1.0f; // make it symmetric
            }
        }
    }

    return AdjacencyMatrix;
}

// function to create sample labels for different task types
std::vector<std::vector<float>> createSampleLabels(int NumNodes, const std::string &TaskType, int NumClasses = 3)
{
    std::vector<std::vector<float>> Labels(NumNodes);
    std::random_device rd;
    std::mt19937 gen(rd());

    if (TaskType == "classification")
    {
        std::uniform_int_distribution<int> classDist(0, NumClasses - 1);
        for (int i = 0; i < NumNodes; ++i)
        {
            Labels[i].resize(NumClasses, 0.0f);
            int ClassLabel = classDist(gen);
            Labels[i][ClassLabel] = 1.0f; // one-hot encoding
        }
    }
    else if (TaskType == "regression")
    {
        std::uniform_real_distribution<float> valueDist(0.0f, 10.0f);
        for (int i = 0; i < NumNodes; ++i)
        {
            Labels[i].resize(1);
            Labels[i][0] = valueDist(gen);
        }
    }
    else if (TaskType == "property_prediction")
    {
        std::uniform_int_distribution<int> binaryDist(0, 1);
        for (int i = 0; i < NumNodes; ++i)
        {
            Labels[i].resize(1);
            Labels[i][0] = static_cast<float>(binaryDist(gen));
        }
    }

    return Labels;
}

// function to calculate accuracy for classification tasks
float calculateClassificationAccuracy(const std::vector<std::vector<float>> &Predictions,
                                      const std::vector<std::vector<float>> &Labels)
{
    int Correct = 0;
    int Total = Predictions.size();

    for (size_t i = 0; i < Predictions.size(); ++i)
    {
        // find predicted class
        int PredictedClass = 0;
        float MaxProb = Predictions[i][0];
        for (size_t j = 1; j < Predictions[i].size(); ++j)
        {
            if (Predictions[i][j] > MaxProb)
            {
                MaxProb = Predictions[i][j];
                PredictedClass = j;
            }
        }

        // find true class
        int TrueClass = 0;
        for (size_t j = 0; j < Labels[i].size(); ++j)
        {
            if (Labels[i][j] == 1.0f)
            {
                TrueClass = j;
                break;
            }
        }

        if (PredictedClass == TrueClass)
        {
            Correct++;
        }
    }

    return static_cast<float>(Correct) / Total;
}

// function to monitor gpu memory usage
void printGPUMemoryUsage()
{
    size_t FreeMemory, TotalMemory;
    cudaMemGetInfo(&FreeMemory, &TotalMemory);

    std::cout << "GPU Memory Usage:" << std::endl;
    std::cout << "  Total Memory: " << TotalMemory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Free Memory: " << FreeMemory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Used Memory: " << (TotalMemory - FreeMemory) / (1024 * 1024) << " MB" << std::endl;
    std::cout << std::endl;
}

int main()
{
    std::cout << "=" << std::string(80, '=') << std::endl;
    std::cout << "GRAPH NEURAL NETWORK DEMONSTRATION" << std::endl;
    std::cout << "implementing node classification, regression, and property prediction" << std::endl;
    std::cout << std::endl;
    std::cout << "note: this cuda implementation follows the same logic as the python version" << std::endl;
    std::cout << "but uses gpu acceleration for faster computation on large graphs" << std::endl;
    std::cout << "=" << std::string(80, '=') << std::endl;
    std::cout << std::endl;

    // display gpu information
    std::cout << "GPU INFORMATION:" << std::endl;
    std::cout << "================" << std::endl;
    Utils::printDeviceInfo();
    printGPUMemoryUsage();

    // set random seed for reproducible results
    Utils::setRandomSeed(42);

    // create sample graph data
    int NumNodes = 1980;
    int FeatureDim = 10;
    int NumClasses = 3;

    std::cout << "creating sample graph with " << NumNodes << " nodes and " << FeatureDim << " features each..." << std::endl;
    auto NodeFeatures = createSampleNodeFeatures(NumNodes, FeatureDim);
    auto AdjacencyMatrix = createSampleAdjacencyMatrix(NumNodes, 0.3f);

    std::cout << "graph has " << NumNodes << " users with " << FeatureDim << " features each" << std::endl;
    std::cout << std::endl;

    // test node classification
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "NODE CLASSIFICATION EXAMPLE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "classifying users into " << NumClasses << " categories" << std::endl;
    std::cout << std::endl;

    auto ClassificationLabels = createSampleLabels(NumNodes, "classification", NumClasses);

    // create and train gnn for classification
    GNN ClassificationGNN(FeatureDim, {8, 6}, NumClasses, "classification", 0.01f);

    std::cout << "training gnn for node classification..." << std::endl;

    // measure training time
    auto StartTime = std::chrono::high_resolution_clock::now();
    ClassificationGNN.train(NodeFeatures, AdjacencyMatrix, ClassificationLabels, 50, true);
    auto EndTime = std::chrono::high_resolution_clock::now();
    auto TrainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(EndTime - StartTime);

    std::cout << "training completed in " << TrainingTime.count() << " ms" << std::endl;

    // make predictions and calculate accuracy
    auto ClassificationPredictions = ClassificationGNN.predict(NodeFeatures, AdjacencyMatrix);
    float ClassificationAccuracy = calculateClassificationAccuracy(ClassificationPredictions, ClassificationLabels);

    std::cout << "node classification accuracy: " << std::fixed << std::setprecision(4)
              << ClassificationAccuracy << " (" << ClassificationAccuracy * 100 << "%)" << std::endl;
    std::cout << std::endl;

    // test node regression
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "NODE REGRESSION EXAMPLE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "predicting continuous values for nodes" << std::endl;
    std::cout << std::endl;

    auto RegressionLabels = createSampleLabels(NumNodes, "regression");

    // create and train gnn for regression
    GNN RegressionGNN(FeatureDim, {8, 6}, 1, "regression", 0.01f);

    std::cout << "training gnn for node regression..." << std::endl;
    RegressionGNN.train(NodeFeatures, AdjacencyMatrix, RegressionLabels, 50, true);

    // make predictions
    auto RegressionPredictions = RegressionGNN.predict(NodeFeatures, AdjacencyMatrix);

    // calculate mean squared error
    float TotalError = 0.0f;
    for (size_t i = 0; i < RegressionPredictions.size(); ++i)
    {
        float Error = RegressionPredictions[i][0] - RegressionLabels[i][0];
        TotalError += Error * Error;
    }
    float MSE = TotalError / RegressionPredictions.size();

    std::cout << "node regression mean squared error: " << std::fixed << std::setprecision(6) << MSE << std::endl;
    std::cout << std::endl;

    // test property prediction
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "PROPERTY PREDICTION EXAMPLE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "predicting binary properties for nodes" << std::endl;
    std::cout << std::endl;

    auto PropertyLabels = createSampleLabels(NumNodes, "property_prediction");

    // create and train gnn for property prediction
    GNN PropertyGNN(FeatureDim, {8, 6}, 1, "property_prediction", 0.01f);

    std::cout << "training gnn for property prediction..." << std::endl;
    PropertyGNN.train(NodeFeatures, AdjacencyMatrix, PropertyLabels, 100, true);

    // make predictions
    auto PropertyPredictions = PropertyGNN.predict(NodeFeatures, AdjacencyMatrix);

    // calculate binary accuracy
    int CorrectPredictions = 0;
    for (size_t i = 0; i < PropertyPredictions.size(); ++i)
    {
        float PredictedValue = PropertyPredictions[i][0];
        float TrueValue = PropertyLabels[i][0];

        // threshold at 0.5 for binary classification
        if ((PredictedValue > 0.5f && TrueValue > 0.5f) || (PredictedValue <= 0.5f && TrueValue <= 0.5f))
        {
            CorrectPredictions++;
        }
    }
    float PropertyAccuracy = static_cast<float>(CorrectPredictions) / PropertyPredictions.size();

    std::cout << "property prediction accuracy: " << std::fixed << std::setprecision(4)
              << PropertyAccuracy << " (" << PropertyAccuracy * 100 << "%)" << std::endl;
    std::cout << std::endl;

    // display gnn architecture info
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "GNN ARCHITECTURE SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "input dimension: " << FeatureDim << std::endl;
    std::cout << "hidden dimensions: [8, 6]" << std::endl;
    std::cout << "learning rate: 0.01" << std::endl;
    std::cout << "task types supported: classification, regression, property_prediction" << std::endl;
    std::cout << "gpu acceleration: enabled via cuda kernels" << std::endl;
    std::cout << std::endl;

    std::cout << "=" << std::string(80, '=') << std::endl;
    std::cout << "PERFORMANCE SUMMARY" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << "total nodes processed: " << NumNodes << std::endl;
    std::cout << "features per node: " << FeatureDim << std::endl;
    std::cout << "total training epochs: 150 (50 per task)" << std::endl;
    std::cout << "gpu acceleration: enabled via cuda kernels" << std::endl;
    std::cout << "memory management: automatic gpu allocation/deallocation" << std::endl;
    std::cout << std::endl;

    std::cout << "=" << std::string(80, '=') << std::endl;
    std::cout << "GNN DEMONSTRATION COMPLETED SUCCESSFULLY!" << std::endl;
    std::cout << "=" << std::string(80, '=') << std::endl;

    return 0;
}