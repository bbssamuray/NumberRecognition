
#include <stdio.h>

#include <fstream>

#include "neurolib.h"

int changeEndian(unsigned char* buffer) {

    return (int)buffer[3] | (int)buffer[2] << 8 | (int)buffer[1] << 16 | (int)buffer[0] << 24;
}

char giveLabel(std::ifstream& labelFile) {
    // Take file stream and return label of the next image

    char temp;
    labelFile.read(&temp, 1);

    if (labelFile.eof()) {
        // Go back to the beginning if the file ended
        labelFile.clear();
        labelFile.seekg(8, labelFile.beg);  // Skip metadata
        return giveLabel(labelFile);
    }

    return temp;
}

// If defined, shifts training and test images in a random direction
#define imageShift  // uncomment this line to enable image shifting

// Side note about imageShift:
// Implemented the shift after I realized it didn't recognize some of my handwriting
// (especially with 9, which I have seen other people also say their NN struggled with too)
// Even though it makes it harder for it to learn
// it helped a lot when it was trying to recognize my handwriting
// Success rate on my own small curated dataset went from %55~ to %90~

void giveImage(std::ifstream& imFile, float* pixelBuffer) {
    // Take file stream and buffer
    // Fill the buffer with the next image's pixel data
    // 1.0 is white, 0.0 is black

    const int imageSize = 28 * 28;
    char temp;

    for (int pixelId = 0; pixelId < imageSize; pixelId++) {
        imFile.read(reinterpret_cast<char*>(&temp), 1);
        pixelBuffer[pixelId] = (unsigned char)temp / 255.0;
    }

    if (imFile.eof()) {
        // Go back to the beginning if the file ended
        imFile.clear();
        imFile.seekg(16, imFile.beg);  // Skip metadata
        giveImage(imFile, pixelBuffer);
    }

    float tempFloat;

    // Dataset images are flipped for some reason?
    for (int y = 0; y < 13; y++) {  // Half of 28. So we don't flip it twice
        for (int x = 0; x < 28; x++) {
            tempFloat = pixelBuffer[y * 28 + x];
            pixelBuffer[y * 28 + x] = pixelBuffer[(27 - y) * 28 + x];
            pixelBuffer[(27 - y) * 28 + x] = tempFloat;
        }
    }

#ifdef imageShift
    // Shift image in a random direction

    float oldImage[imageSize];

    std::copy(pixelBuffer, pixelBuffer + imageSize, oldImage);

    const int shiftMax = 2;  // Maximum amount of shifting to be done

    // Get random values between +shiftMax and -shiftMax
    const int shiftX = rand() % (shiftMax * 2 + 1) - shiftMax;
    const int shiftY = rand() % (shiftMax * 2 + 1) - shiftMax;

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {

            const int xCord = x + shiftX;
            const int yCord = y + shiftY;
            if (xCord >= 0 && xCord <= 27 && yCord >= 0 && yCord <= 27) {
                pixelBuffer[y * 28 + x] = oldImage[yCord * 28 + xCord];
            } else {
                pixelBuffer[y * 28 + x] = 0.0;
            }
        }
    }

#endif
}

void testModel(std::string traininigFileName, std::string labelFileName, std::string modelFileName) {

    neurolib neuronNet = neurolib(modelFileName);

    std::ifstream labelFile(labelFileName, std::ios::in | std::ios::binary);  // Test label file

    if (!labelFile.is_open()) {
        printf("Couldn't find test label file! Aborting testing.\n");
        return;
    }
    int labelNum;
    char label;

    labelFile.read(reinterpret_cast<char*>(&labelNum), 4);  // Magic bytes, not important
    labelNum = changeEndian(reinterpret_cast<unsigned char*>(&labelNum));
    labelFile.read(reinterpret_cast<char*>(&labelNum), 4);  // Number of labels
    labelNum = changeEndian(reinterpret_cast<unsigned char*>(&labelNum));

    std::ifstream imFile(traininigFileName, std::ios::in | std::ios::binary);  // Test image file

    if (!imFile.is_open()) {
        printf("Couldn't find test image file! Aborting testing.\n");
        return;
    }

    int imNum, imRow, imCol;

    imFile.read(reinterpret_cast<char*>(&imNum), 4);  // Magic byte, not important
    imFile.read(reinterpret_cast<char*>(&imNum), 4);  // Number of images
    imNum = changeEndian(reinterpret_cast<unsigned char*>(&imNum));
    imFile.read(reinterpret_cast<char*>(&imRow), 4);  // Number of rows in image
    imRow = changeEndian(reinterpret_cast<unsigned char*>(&imRow));
    imFile.read(reinterpret_cast<char*>(&imCol), 4);  // Number of columns in image
    imCol = changeEndian(reinterpret_cast<unsigned char*>(&imCol));

    if (imNum != labelNum) {
        return;
    }

    const int outLayerSize = neuronNet.layers[neuronNet.numOfLayers - 1].size;

    float* inputs = new float[neuronNet.layers[0].size];
    float* outputs = new float[outLayerSize];

    int biggestOut = 0;
    int correctGuesses = 0;
    bool correct = false;

    for (int imId = 0; imId < imNum; imId++) {
        correct = false;

        giveImage(imFile, inputs);
        label = giveLabel(labelFile);

        neuronNet.runModel(inputs, outputs);

        biggestOut = 0;
        for (int i = 0; i < outLayerSize; i++) {
            if (outputs[biggestOut] < outputs[i]) biggestOut = i;
        }

        if (biggestOut == label) {
            correctGuesses++;
            correct = true;
        }
    }

    printf("Number of correct guesses in test dataset: %d  - ", correctGuesses);
    printf("Success rate: %%%2.2f\n\n", (float)correctGuesses / (float)imNum * 100);

    labelFile.close();
    imFile.close();
    delete[] outputs;
    delete[] inputs;
}

void startTraining(std::string traininigFileName, std::string labelFileName, std::string modelFileName) {

    std::ifstream labelFile(labelFileName, std::ios::in | std::ios::binary);  // Training label file

    if (!labelFile.is_open()) {
        printf("Couldn't find training label file! Aborting training.\n");
        return;
    }
    int labelNum;
    char label;

    labelFile.read(reinterpret_cast<char*>(&labelNum), 4);  // Magic bytes, not important
    labelNum = changeEndian(reinterpret_cast<unsigned char*>(&labelNum));
    labelFile.read(reinterpret_cast<char*>(&labelNum), 4);  // Number of labels
    labelNum = changeEndian(reinterpret_cast<unsigned char*>(&labelNum));

    std::ifstream imFile(traininigFileName, std::ios::in | std::ios::binary);  // Training image file

    if (!imFile.is_open()) {
        printf("Couldn't find training image file! Aborting training.\n");
        return;
    }

    int imNum, imRow, imCol;

    imFile.read(reinterpret_cast<char*>(&imNum), 4);  // Magic byte, not important
    imFile.read(reinterpret_cast<char*>(&imNum), 4);  // Number of images
    imNum = changeEndian(reinterpret_cast<unsigned char*>(&imNum));
    imFile.read(reinterpret_cast<char*>(&imRow), 4);  // Number of rows in image
    imRow = changeEndian(reinterpret_cast<unsigned char*>(&imRow));
    imFile.read(reinterpret_cast<char*>(&imCol), 4);  // Number of columns in image
    imCol = changeEndian(reinterpret_cast<unsigned char*>(&imCol));

    if (imNum != labelNum) {
        printf("Label file doesn't match image file. Aborting.\n");
        return;
    }

    int layerSizes[] = {imCol * imRow, 400, 300, 200, 100, 10};
    int numOfLayers = sizeof(layerSizes) / sizeof(int);
    float* inputs = new float[layerSizes[0]];
    float* outputs = new float[layerSizes[numOfLayers - 1]];

    neurolib neuronNet(layerSizes, numOfLayers);

    int correctGuesses = 0;

    const int epoch = 50;
    const int batchSize = 100;

    for (int trLoop = 1; trLoop < imNum * epoch; trLoop++) {

        giveImage(imFile, inputs);
        label = giveLabel(labelFile);

        neuronNet.trainModel(inputs, (int)label, outputs);

        // Check if the prediction is correct
        int biggestOut = 0;
        for (int i = 0; i < layerSizes[numOfLayers - 1]; i++) {
            if (outputs[biggestOut] < outputs[i]) biggestOut = i;
        }

        if (biggestOut == label) {
            correctGuesses++;
        }

        if (trLoop % imNum == 0) {
            // Test model every epoch
            correctGuesses = 0;
            testModel("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", modelFileName);
        }

        if (trLoop % 7001 == 0) {
            // Print debug info
            if (trLoop % imNum != 0)
                printf("trloop: %6d %6d, current loop accuracy: %%%2.2f , image with label %6d outputs:\n", trLoop / imNum, trLoop % imNum, (float)correctGuesses / (trLoop % imNum) * 100, label);
            for (int i = 0; i < layerSizes[numOfLayers - 1]; i++) {
                printf(" %d: %f", i, outputs[i]);
            }
            printf("\n");
        }

        if (trLoop % batchSize == 0) {
            // Apply training every 100 image
            neuronNet.applyBatch();
            neuronNet.saveModel(modelFileName);
        }
    }

    neuronNet.applyBatch();
    neuronNet.saveModel(modelFileName);

    labelFile.close();
    imFile.close();
    delete[] outputs;
    delete[] inputs;
}

int guessNumberFromBMP(std::string bmpFileName, std::string modelFileName) {

    std::ifstream bmpFile(bmpFileName, std::ios::in | std::ios::binary);
    if (!bmpFile.is_open()) {
        printf("%s file doesn't exist!\n Quitting.", bmpFileName.c_str());
        return -1;
    }

    char* header = new char[54];

    bmpFile.read(header, 54);

    int width = *(int*)&header[18];
    int height = *(int*)&header[22];
    int rowPadded = (width * 3 + 3) & (~3);
    // Rounds up to multiples of 4

    char* pixelData = new char[rowPadded];
    float* inputs = new float[width * height];
    float* outputs = new float[10];

    if (width != 28 || height != 28) {
        printf("width: %d\nHeight: %d\n", width, height);
        printf("WARNING:\n This binary only supports 28x28 pixel images.");
    }

    for (int y = 0; y < height; y++) {
        bmpFile.read(pixelData, rowPadded);
        // Read a row of pixels into the pixelData array

        for (int x = 0; x < width * 3; x += 3) {
            inputs[y * width + x / 3] = ((unsigned char)pixelData[x] + (unsigned char)pixelData[x + 1] + (unsigned char)pixelData[x + 2]) / 3.0 / 255.0;
            // Get the average of RGB values and put it into the input array
        }
    }

    neurolib neuroNet(modelFileName);

    neuroNet.runModel(inputs, outputs);
    neuroNet.softMax(outputs, 10);

    printf("Results:\n");

    int biggestOutput = 0;

    for (int i = 0; i < 10; i++) {
        if (outputs[biggestOutput] < outputs[i]) {
            biggestOutput = i;
        }
        printf(" %d : %f\n", i, outputs[i]);  // Uncomment this
    }

    delete[] pixelData;
    delete[] header;

    delete[] inputs;
    delete[] outputs;

    return biggestOutput;
}

int main(int argc, char** argv) {

    std::string modelFileName;

    if (argc < 2) {
        modelFileName = "numberModel.o";
    } else {
        for (int i = 1; i < argc; i++) {
            modelFileName += argv[i];
        }
    }

    printf("Using model %s\n", modelFileName.c_str());

    // There probably is a cleaner way of checking if a file exists
    std::ifstream modelFile(modelFileName, std::ios::in | std::ios::binary);  // Model file

    if (!modelFile.is_open()) {
        // Don't train if there already is a model file
        startTraining("train-images-idx3-ubyte", "train-labels-idx1-ubyte", modelFileName);
        return 0;
    }
    modelFile.close();

    // Test model if the test datasets exist
    testModel("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", modelFileName);

    int guess = guessNumberFromBMP("number.bmp", modelFileName);

    printf("Your number is %d.\n", guess);

    return 0;
}