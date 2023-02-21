#include "neurolib.h"

#include <bits/stdc++.h>
#include <stdio.h>

#include <cmath>
#include <fstream>

neurolib::neurolib(std::string modelName) {
    // For loading models

    std::ifstream modelFile(modelName, std::ios::in | std::ios::binary);

    // No error checking is done in here
    // That needs to be taken care of

    char* buffer = new char[4];

    // Would be nice if we checked the magic bytes
    // We will just skip over them for now
    modelFile.read(buffer, 4);
    modelFile.read(buffer, 4);
    modelFile.read(buffer, 4);

    modelFile.read(reinterpret_cast<char*>(&numOfLayers), 4);
    layers = new layer[numOfLayers];  // Initialize layers

    for (int layerId = 0; layerId < numOfLayers; layerId++) {
        modelFile.read(reinterpret_cast<char*>(&layers[layerId].size), 4);
    }

    for (int layerId = 0; layerId < numOfLayers; layerId++) {
        layer* currentLayer = &(layers[layerId]);

        currentLayer->neurons = new neuron[currentLayer->size];
        // Initialize this layer's neurons

        for (int neuronId = 0; neuronId < currentLayer->size; neuronId++) {
            neuron* currentNeuron = &(currentLayer->neurons[neuronId]);

            modelFile.read(reinterpret_cast<char*>(&currentNeuron->bias), sizeof(float));
            // Read biases

            if (layerId == numOfLayers - 1) {
                currentNeuron->weights = NULL;
                currentNeuron->weightBatchSum = NULL;
                // output layer doesn't need any weights
                continue;
            }

            const int weightCount = layers[layerId + 1].size;
            currentNeuron->weights = new float[weightCount];
            // Initialize weights
            currentNeuron->weightBatchSum = new float[weightCount];
            // Initialize weight batches

            for (int weightId = 0; weightId < weightCount; weightId++) {

                modelFile.read(reinterpret_cast<char*>(&currentNeuron->weights[weightId]), sizeof(float));
                // Read weights

                currentNeuron->weightBatchSum[weightId] = 0.0;
            }
        }
    }

    modelFile.close();

    delete[] buffer;
}

neurolib::neurolib(int layerSizes[], int numOfLayers) {

    this->numOfLayers = numOfLayers;
    const float startingValue = 0.2;  // All of the weights and biases will be between +startingValue and -startingValue when initializing

    trainingSinceLastBatch = 0;

    srand(0);  // Change the seed later

    layers = new layer[numOfLayers];  // Initialize layers
    for (int layerId = 0; layerId < numOfLayers; layerId++) {
        layer* currentLayer = &(layers[layerId]);

        currentLayer->size = layerSizes[layerId];

        currentLayer->neurons = new neuron[layerSizes[layerId]];  // Initialize neurons in the said layer
        for (int neuronId = 0; neuronId < layerSizes[layerId]; neuronId++) {
            neuron* currentNeuron = &(currentLayer->neurons[neuronId]);

            // randomize biases
            currentNeuron->bias = randF(-startingValue, startingValue);
            // Set bias batch to 0
            currentNeuron->biasBatchSum = 0.0;

            if (layerId == numOfLayers - 1) {
                currentNeuron->weights = NULL;
                currentNeuron->weightBatchSum = NULL;
                // output layer doesn't need any weights
                continue;
            }

            currentNeuron->weights = new float[layerSizes[layerId + 1]];         // Initialize weights connecting to the next layer
            currentNeuron->weightBatchSum = new float[layerSizes[layerId + 1]];  // Init weight batches

            for (int weightId = 0; weightId < layerSizes[layerId + 1]; weightId++) {
                // Randomize weights
                currentNeuron->weights[weightId] = randF(-startingValue, startingValue);
                // Set weight batch to 0
                currentNeuron->weightBatchSum[weightId] = 0.0;
            }
        }
    }
}

neurolib::~neurolib() {
    for (int layerId = 0; layerId < numOfLayers; layerId++) {
        for (int neuronId = 0; neuronId < layers[layerId].size; neuronId++) {
            if (layers[layerId].neurons[neuronId].weights != NULL)
                delete[] layers[layerId].neurons[neuronId].weights;
            if (layers[layerId].neurons[neuronId].weightBatchSum != NULL)
                delete[] layers[layerId].neurons[neuronId].weightBatchSum;
        }
        if (layers[layerId].neurons != NULL)
            delete[] layers[layerId].neurons;
    }
    if (layers != NULL)
        delete[] layers;
}

float neurolib::randF(float min, float max) {
    // Gives a random float value between a range
    return (float)(rand()) / (float)(RAND_MAX) * (max - min) + (min);
}

#define FUNCRELU  // Comment this line to use sigmoid, not quite sure how well it works though

#if !defined FUNCRELU
#define FUNCSIGMOID
#endif

inline float neurolib::actFunc(float x) {

#ifdef FUNCRELU
    // Leaky RelU
    if (x > 0) {
        return x;
    } else {
        return x * 0.1;
    }
#endif

#ifdef FUNCSIGMOID
    // Sigmoid
    return 1 / (1 + expf(-x));
#endif
}

inline float neurolib::actFuncDer(float x) {
    // Derivative of the activation function

#ifdef FUNCRELU
    // Leaky RelU derivative
    if (x >= 0) {
        return 1.0;
    } else {
        return 0.1;
    }
#endif

#ifdef FUNCSIGMOID
    // Sigmoid derivative
    return x * (1 - x);
#endif
}

void neurolib::softMax(float* inputs, int inputSize) {
    // Applies soft max to inputs and writes it into the same array

    // Default value of inputSize is 0

    if (inputSize <= 0) {
        // Use output layer's size if no size is given
        // Overload this with no int argument?
        inputSize = layers[numOfLayers - 1].size;
    }

    float sum = 0.0;

    for (int i = 0; i < inputSize; i++) {
        sum += expf(inputs[i]);
    }

    for (int i = 0; i < inputSize; i++) {
        inputs[i] = expf(inputs[i]) / sum;
    }
}

void neurolib::runModel(float inputs[], float outputs[]) {
    // Runs the model with the given inputs
    // Results will be written to outputs
    // outputs must be initialized before passing it here

    // Set input layer's values
    for (int inputId = 0; inputId < layers[0].size; inputId++) {
        layers[0].neurons[inputId].value = inputs[inputId];
    }

    // Reset all of other neuron values to their bias
    for (int layerId = 1; layerId < numOfLayers; layerId++) {
        for (int neuronId = 0; neuronId < layers[layerId].size; neuronId++) {
            layers[layerId].neurons[neuronId].value = layers[layerId].neurons[neuronId].bias;
        }
    }

    for (int layerId = 0; layerId < numOfLayers - 1; layerId++) {
        layer* currentLayer = &(layers[layerId]);
        layer* nextLayer = &(layers[layerId + 1]);

        // multiply current neuron's value with the corresponding weight and add it to the neuron that weight is connected to
        for (int neuronId = 0; neuronId < layers[layerId].size; neuronId++) {
            neuron* currentNeuron = &(currentLayer->neurons[neuronId]);

            for (int weightId = 0; weightId < nextLayer->size; weightId++) {
                nextLayer->neurons[weightId].value += currentNeuron->weights[weightId] * currentNeuron->value;
            }
        }

        // Run every neuron of the next layer through the activation function
        for (int neuronId = 0; neuronId < layers[layerId + 1].size; neuronId++) {
            nextLayer->neurons[neuronId].value = actFunc(nextLayer->neurons[neuronId].value);
        }
    }

    for (int i = 0; i < layers[numOfLayers - 1].size; i++) {
        outputs[i] = layers[numOfLayers - 1].neurons[i].value;
    }
}

void neurolib::trainModel(float inputs[], int truth, float outputs[]) {
    // Calculate derivatives and add tweaks to the batch sums

    // truth is the ID of output neuron that should be 1.0

    trainingSinceLastBatch++;

    float* smaxResults;

    if (outputs == NULL) {
        smaxResults = new float[layers[numOfLayers - 1].size];
    } else {
        smaxResults = outputs;
    }
    runModel(inputs, smaxResults);
    softMax(smaxResults);

    // Set up output layer's derivatives and biases
    for (int neuronId = 0; neuronId < layers[numOfLayers - 1].size; neuronId++) {
        neuron* currentNeuron = &(layers[numOfLayers - 1].neurons[neuronId]);
        if (neuronId == truth) {
            // Need to pass values before the activation function
            // Doesn't really matter for RelU or sigmoid though
            const float derivative = (smaxResults[neuronId] - 1) * actFuncDer(currentNeuron->value);
            currentNeuron->derivative = derivative;
            currentNeuron->biasBatchSum += derivative;
        } else {
            // Same as above
            const float derivative = (smaxResults[neuronId]) * actFuncDer(currentNeuron->value);
            currentNeuron->derivative = derivative;
            currentNeuron->biasBatchSum += derivative;
        }
    }

    for (int layerId = numOfLayers - 2; layerId >= 0; layerId--) {
        layer* currentLayer = &(layers[layerId]);
        layer* nextLayer = &(layers[layerId + 1]);

        for (int neuronId = 0; neuronId < currentLayer->size; neuronId++) {
            neuron* currentNeuron = &(currentLayer->neurons[neuronId]);

            // Calculate the derivative
            float derivative = 0;
            for (int nextNeuronId = 0; nextNeuronId < nextLayer->size; nextNeuronId++) {
                // Sum all of the next layer's neurons multiplied by the corresponding weight
                neuron* nextNeuron = &(nextLayer->neurons[nextNeuronId]);
                derivative += nextNeuron->derivative * currentNeuron->weights[nextNeuronId];
            }
            derivative *= actFuncDer(currentNeuron->value);
            currentNeuron->derivative = derivative;

            // Tweak the bias batch
            currentNeuron->biasBatchSum += derivative;

            // Tweak the weight batch
            for (int weightId = 0; weightId < layers[layerId + 1].size; weightId++) {
                currentNeuron->weightBatchSum[weightId] += nextLayer->neurons[weightId].derivative * currentNeuron->value;
            }
        }
    }

    if (outputs == NULL) {
        delete[] smaxResults;
    }
}

void neurolib::applyBatch() {
    // Averages the weight and bias addition sums in the current batch and adds it to the model

    if (trainingSinceLastBatch == 0) {
        // Nothing to do...
        return;
    }

    for (int layerId = 0; layerId < numOfLayers; layerId++) {
        layer* currentLayer = &(layers[layerId]);
        const int weightCount = layers[layerId + 1].size;  // Todo: Funny bug here

        for (int neuronId = 0; neuronId < currentLayer->size; neuronId++) {
            neuron* currentNeuron = &(currentLayer->neurons[neuronId]);

            // Apply bias batch
            currentNeuron->bias -= currentNeuron->biasBatchSum / trainingSinceLastBatch * stepSize;

            currentNeuron->biasBatchSum = 0.0;

            // Last layer Doesn't have any weights
            if (layerId == numOfLayers - 1) {
                continue;
            }

            // Apply weight batch
            for (int weightId = 0; weightId < weightCount; weightId++) {
                currentNeuron->weights[weightId] -= currentNeuron->weightBatchSum[weightId] / (float)trainingSinceLastBatch * stepSize;
                currentNeuron->weightBatchSum[weightId] = 0.0;
            }
        }
    }

    trainingSinceLastBatch = 0;
}

int neurolib::saveModel(std::string modelName) {
    // Saves model to the given path

    // Todo: No error checking is done in this function
    // and that is bad

    std::ofstream modelFile(modelName, std::ios::out | std::ios::binary | std::ios::ate);

    float modelFileWeight = 0.5;

    modelFile.write(magicBytes, sizeof(magicBytes));
    // Write magic bytes

    modelFile.write(reinterpret_cast<const char*>(&numOfLayers), sizeof(int));
    // Write the number of layers

    for (int layerId = 0; layerId < numOfLayers; layerId++) {
        modelFile.write(reinterpret_cast<const char*>(&layers[layerId].size), sizeof(int));
        // Write each layer's size
    }

    for (int layerId = 0; layerId < numOfLayers - 1; layerId++) {
        layer* currentLayer = &(layers[layerId]);
        const int weightCount = layers[layerId + 1].size;

        for (int neuronId = 0; neuronId < currentLayer->size; neuronId++) {
            neuron* currentNeuron = &(currentLayer->neurons[neuronId]);

            modelFile.write(reinterpret_cast<const char*>(&currentNeuron->bias), sizeof(float));
            // Write biases

            for (int weightId = 0; weightId < weightCount; weightId++) {
                modelFile.write(reinterpret_cast<const char*>(&currentNeuron->weights[weightId]), sizeof(float));
                // Write weights
            }
        }
    }

    layer* currentLayer = &(layers[numOfLayers - 1]);
    for (int neuronId = 0; neuronId < currentLayer->size; neuronId++) {
        neuron* currentNeuron = &(currentLayer->neurons[neuronId]);
        modelFile.write(reinterpret_cast<const char*>(&currentNeuron->bias), sizeof(float));
        // Write output layer's biases
    }

    modelFile.close();

    return 0;
}

void neurolib::printWeightInfo() {
    // Prints some debug info

    bool printBatchSum = false;
    if (trainingSinceLastBatch != 0) {
        printBatchSum = true;
        printf("\nTraining since last batch apply: %d\n", trainingSinceLastBatch);
    } else {
        printf("\n");
    }

    for (int layerId = 0; layerId < numOfLayers; layerId++) {
        layer* currentLayer = &(layers[layerId]);
        const int weightCount = layers[layerId + 1].size;  // Todo: copy pasted funny bug

        printf(" Layer %d:\n", layerId);

        for (int neuronId = 0; neuronId < currentLayer->size; neuronId++) {
            neuron* currentNeuron = &(currentLayer->neurons[neuronId]);

            printf("   Neuron %d:\n", neuronId);
            printf("      B : %+f", currentNeuron->bias);
            if (printBatchSum)
                printf("   BBSum: %+f\n      --------------------------------\n", currentNeuron->biasBatchSum);
            else
                printf("\n      -------------\n");

            // Last layer Doesn't have any weights
            if (layerId == numOfLayers - 1) {
                continue;
            }

            for (int weightId = 0; weightId < weightCount; weightId++) {
                printf("     %2d : %+f", weightId, currentNeuron->weights[weightId]);
                if (printBatchSum) printf("   WBSum: %+f", currentNeuron->weightBatchSum[weightId]);
                printf("\n");
            }
        }
    }
}