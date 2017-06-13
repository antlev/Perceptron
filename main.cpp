#include <iostream>
#include "MLP.h"

int main() {
    int nbLayer = 3;
    int modelStruct[3] = { 2, 2, 1 };
    int inputSize = 2;
    int outputSize = 1;

    int nbData =3;

    double* inputs = new double[inputSize*nbData];
    double* expectedOutputs = new double[outputSize*nbData];
    double* oneInput = new double[inputSize];
    double* oneOutput = new double[outputSize];

    MLP* regressionPerceptron = new MLP(modelStruct,nbLayer);

    inputs[0] = 0;
    inputs[1] = 0;

    inputs[2] = 0;
    inputs[3] = 1;

    inputs[4] = 1;
    inputs[5] = 1;

    expectedOutputs[0] = 0;
    expectedOutputs[1] = 0;
    expectedOutputs[2] = 0.5;

    std::cout << "Fitting regression model..." << std::endl;
    regressionPerceptron->fitRegression(inputs, inputSize, inputSize*nbData, expectedOutputs, outputSize);

    std::cout << "Testing Regression" << std::endl;
    oneInput[0] = 0;
    oneInput[1] = 0;
    regressionPerceptron->predict(oneInput,inputSize,&oneOutput,outputSize);
    std::cout << "Response for input = [" << oneInput[0] << "][" << oneInput[1] << "] ->" << oneOutput[0] << "<" << std::endl;

    oneInput[0] = 0;
    oneInput[1] = 1;
    regressionPerceptron->predict(oneInput,inputSize,&oneOutput,outputSize);
    std::cout << "Response for input = [" << oneInput[0] << "][" << oneInput[1] << "] ->" << oneOutput[0] << "<" << std::endl;

    oneInput[0] = 1;
    oneInput[1] = 1;
    regressionPerceptron->predict(oneInput,inputSize,&oneOutput,outputSize);
    std::cout << "Response for input = [" << oneInput[0] << "][" << oneInput[1] << "] ->" << oneOutput[0] << "<" << std::endl;


    MLP* classificationPerceptron = new MLP(modelStruct,nbLayer);

    inputs[0] = 0;
    inputs[1] = 0;

    inputs[2] = 0;
    inputs[3] = 1;

    inputs[4] = 1;
    inputs[5] = 1;

    expectedOutputs[0] = 0;
    expectedOutputs[1] = 0;
    expectedOutputs[2] = 1;

    std::cout << "Fitting Classification model..." << std::endl;
    regressionPerceptron->fitClassification(inputs, inputSize, inputSize*nbData, expectedOutputs, outputSize);

    std::cout << "Testing Classification" << std::endl;
    oneInput[0] = 0;
    oneInput[1] = 0;
    regressionPerceptron->classify(oneInput,inputSize,&oneOutput,outputSize);
    std::cout << "Response for input = [" << oneInput[0] << "][" << oneInput[1] << "] ->" << oneOutput[0] << "<" << std::endl;

    oneInput[0] = 0;
    oneInput[1] = 1;
    regressionPerceptron->classify(oneInput,inputSize,&oneOutput,outputSize);
    std::cout << "Response for input = [" << oneInput[0] << "][" << oneInput[1] << "] ->" << oneOutput[0] << "<" << std::endl;

    oneInput[0] = 1;
    oneInput[1] = 1;
    regressionPerceptron->classify(oneInput,inputSize,&oneOutput,outputSize);
    std::cout << "Response for input = [" << oneInput[0] << "][" << oneInput[1] << "] ->" << oneOutput[0] << "<" << std::endl;

    return 1;
}