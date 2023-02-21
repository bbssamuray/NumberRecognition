#include <string>

class neurolib {

   public:
    struct neuron {
        float* weights;  // Array of weights connecting to the next layer
        float bias;
        float value;
        float derivative;       // For training only
        float* weightBatchSum;  // For training only, array
        float biasBatchSum;     // For training only
    };

    struct layer {
        int size;         // Number of neurons in this layer
        neuron* neurons;  // Array of neurons
    };

    const char magicBytes[12] = "Neuro Model";
    // 4e6575726f204d6f64656c00 in hex
    // These are the first bytes for a model file

    const float stepSize = 0.1;  // Learning rate

    int numOfLayers;
    int trainingSinceLastBatch = 0;
    layer* layers;  // Array of all of the layers

    neurolib(std::string);  // Used for loading models
    neurolib(int layerSizes[], int numOfLayers);
    ~neurolib();
    void softMax(float* inputs, int inputSize = 0);
    void runModel(float inputs[], float outputs[]);
    void trainModel(float* inputs, int truth, float outputs[] = NULL);
    void applyBatch();
    int saveModel(std::string modelName);
    void printWeightInfo();

   private:
    float randF(float min, float max);
    float actFunc(float x);
    float actFuncDer(float x);
};
