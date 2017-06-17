#include <iostream>
#include "MLP.h"

int main() {
     std::cout << "Testing MLP Classification" << std::endl;

    int testClassifMLP_nbLayer = 2;
    int testClassifMLP_modelStruct[2] = { 2, 1 };
    int testClassifMLP_inputSize = 2;
    int testClassifMLP_outputSize = 1;
    int testClassifMLP_nbData =4;

    double* testClassifMLP_inputs = new double[testClassifMLP_inputSize*testClassifMLP_nbData];
    double* testClassifMLP_expectedOutputs = new double[testClassifMLP_outputSize*testClassifMLP_nbData];
    double* testClassifMLP_oneInput = new double[testClassifMLP_inputSize];

    MLP* testClassifMLP = new MLP(testClassifMLP_modelStruct,testClassifMLP_nbLayer);

    testClassifMLP_inputs[0] = 0;
    testClassifMLP_inputs[1] = 0;

    testClassifMLP_inputs[2] = 0;
    testClassifMLP_inputs[3] = 1;

    testClassifMLP_inputs[4] = 1;
    testClassifMLP_inputs[5] = 1;

    testClassifMLP_inputs[6] = 1;
    testClassifMLP_inputs[7] = 0;

    testClassifMLP_expectedOutputs[0] = 1;
    testClassifMLP_expectedOutputs[1] = 1;
    testClassifMLP_expectedOutputs[2] = -1;
    testClassifMLP_expectedOutputs[3] = -1;

    std::cout << "Fitting Classification model with linear inputs" << std::endl;
    testClassifMLP->fitClassification(testClassifMLP_inputs, testClassifMLP_inputSize, testClassifMLP_inputSize*testClassifMLP_nbData,
                                      testClassifMLP_expectedOutputs, testClassifMLP_outputSize);

    testClassifMLP_oneInput[0] = 0;
    testClassifMLP_oneInput[1] = 0;
    testClassifMLP->classify(testClassifMLP_oneInput,testClassifMLP_inputSize);
    std::cout << "Response for input = [" << testClassifMLP_oneInput[0] << "][" << testClassifMLP_oneInput[1] <<
                    "] ->" << testClassifMLP->getOutputsforClassif() << "< expected : 1" << std::endl;

    testClassifMLP_oneInput[0] = 0;
    testClassifMLP_oneInput[1] = 1;
    testClassifMLP->classify(testClassifMLP_oneInput,testClassifMLP_inputSize);
    std::cout << "Response for input = [" << testClassifMLP_oneInput[0] << "][" << testClassifMLP_oneInput[1] <<
              "] ->" << testClassifMLP->getOutputsforClassif() << "< expected : 1" << std::endl;
    testClassifMLP_oneInput[0] = 1;
    testClassifMLP_oneInput[1] = 1;
    testClassifMLP->classify(testClassifMLP_oneInput,testClassifMLP_inputSize);
    std::cout << "Response for input = [" << testClassifMLP_oneInput[0] << "][" << testClassifMLP_oneInput[1] <<
              "] ->" << testClassifMLP->getOutputsforClassif() << "< expected : -1" << std::endl;
    testClassifMLP_oneInput[0] = 1;
    testClassifMLP_oneInput[1] = 0;
    testClassifMLP->classify(testClassifMLP_oneInput,testClassifMLP_inputSize);
    std::cout << "Response for input = [" << testClassifMLP_oneInput[0] << "][" << testClassifMLP_oneInput[1] <<
              "] ->" << testClassifMLP->getOutputsforClassif() << "< expected : -1" << std::endl;



    int testClassifMLP2_nbLayer = 3;
    int testClassifMLP2_modelStruct[3] = { 2, 2, 1 };
    int testClassifMLP2_inputSize = 2;
    int testClassifMLP2_outputSize = 1;
    int testClassifMLP2_nbData =4;

    double* testClassifMLP2_inputs = new double[testClassifMLP_inputSize*testClassifMLP_nbData];
    double* testClassifMLP2_expectedOutputs = new double[testClassifMLP_outputSize*testClassifMLP_nbData];
    double* testClassifMLP2_oneInput = new double[testClassifMLP_inputSize];

    MLP* testClassifMLP2 = new MLP(testClassifMLP2_modelStruct,testClassifMLP2_nbLayer);

    testClassifMLP2_inputs[0] = 0;
    testClassifMLP2_inputs[1] = 0;

    testClassifMLP2_inputs[2] = 1;
    testClassifMLP2_inputs[3] = 1;

    testClassifMLP2_inputs[4] = 0;
    testClassifMLP2_inputs[5] = 1;

    testClassifMLP2_inputs[6] = 1;
    testClassifMLP2_inputs[7] = 0;

    testClassifMLP2_expectedOutputs[0] = 1;
    testClassifMLP2_expectedOutputs[1] = 1;
    testClassifMLP2_expectedOutputs[2] = -1;
    testClassifMLP2_expectedOutputs[3] = -1;

    std::cout << "Fitting Classification model with XOR inputs" << std::endl;
    testClassifMLP2->fitClassification(testClassifMLP2_inputs, testClassifMLP2_inputSize, testClassifMLP2_inputSize*testClassifMLP2_nbData,
                                      testClassifMLP2_expectedOutputs, testClassifMLP2_outputSize);

    testClassifMLP2_oneInput[0] = 0;
    testClassifMLP2_oneInput[1] = 0;
    testClassifMLP2->classify(testClassifMLP2_oneInput,testClassifMLP2_inputSize);
    std::cout << "Response for input = [" << testClassifMLP2_oneInput[0] << "][" << testClassifMLP2_oneInput[1] <<
              "] ->" << testClassifMLP2->getOutputsforClassif() << "< expected : 1" << std::endl;

    testClassifMLP2_oneInput[0] = 1;
    testClassifMLP2_oneInput[1] = 1;
    testClassifMLP2->classify(testClassifMLP2_oneInput,testClassifMLP2_inputSize);
    std::cout << "Response for input = [" << testClassifMLP2_oneInput[0] << "][" << testClassifMLP2_oneInput[1] <<
              "] ->" << testClassifMLP2->getOutputsforClassif() << "< expected : 1" << std::endl;
    testClassifMLP2_oneInput[0] = 0;
    testClassifMLP2_oneInput[1] = 1;
    testClassifMLP2->classify(testClassifMLP2_oneInput,testClassifMLP2_inputSize);
    std::cout << "Response for input = [" << testClassifMLP2_oneInput[0] << "][" << testClassifMLP2_oneInput[1] <<
              "] ->" << testClassifMLP2->getOutputsforClassif() << "< expected : -1" << std::endl;
    testClassifMLP2_oneInput[0] = 1;
    testClassifMLP2_oneInput[1] = 0;
    testClassifMLP2->classify(testClassifMLP2_oneInput,testClassifMLP2_inputSize);
    std::cout << "Response for input = [" << testClassifMLP2_oneInput[0] << "][" << testClassifMLP2_oneInput[1] <<
              "] ->" << testClassifMLP2->getOutputsforClassif() << "< expected : -1" << std::endl;


    std::cout << "Testing MLP Regression" << std::endl;

    int testRegressionMLP_nbLayer = 2;
    int testRegressionMLP_modelStruct[2] = { 2, 1 };
    int testRegressionMLP_inputSize = 2;
    int testRegressionMLP_outputSize = 1;
    int testRegressionMLP_nbData =3;

    double* testRegressionMLP_inputs = new double[testRegressionMLP_inputSize*testRegressionMLP_nbData];
    double* testRegressionMLP_expectedOutputs = new double[testRegressionMLP_outputSize*testRegressionMLP_nbData];
    double* testRegressionMLP_oneInput = new double[testRegressionMLP_inputSize];

    MLP* testRegressionMLP = new MLP(testRegressionMLP_modelStruct,testRegressionMLP_nbLayer);

    testRegressionMLP_inputs[0] = 0;
    testRegressionMLP_inputs[1] = 0;

    testRegressionMLP_inputs[2] = 0;
    testRegressionMLP_inputs[3] = 1;

    testRegressionMLP_inputs[4] = 1;
    testRegressionMLP_inputs[5] = 1;

    testRegressionMLP_expectedOutputs[0] = 0;
    testRegressionMLP_expectedOutputs[1] = 0;
    testRegressionMLP_expectedOutputs[2] = 0.5;

    std::cout << "Fitting regression model..." << std::endl;
    testRegressionMLP->fitRegression(testRegressionMLP_inputs, testRegressionMLP_inputSize, testRegressionMLP_inputSize*testRegressionMLP_nbData,
                                     testRegressionMLP_expectedOutputs, testRegressionMLP_outputSize);

    testRegressionMLP_oneInput[0] = 0;
    testRegressionMLP_oneInput[1] = 0;
    testRegressionMLP->predict(testRegressionMLP_oneInput,testRegressionMLP_inputSize);
    std::cout << "Response for input = [" << testRegressionMLP_oneInput[0] << "][" << testRegressionMLP_oneInput[1] << "] "
            "->" << testRegressionMLP->getOutputsforRegression() << "< expected : 0" << std::endl;

    testRegressionMLP_oneInput[0] = 0;
    testRegressionMLP_oneInput[1] = 1;
    testRegressionMLP->predict(testRegressionMLP_oneInput,testRegressionMLP_inputSize);
    std::cout << "Response for input = [" << testRegressionMLP_oneInput[0] << "][" << testRegressionMLP_oneInput[1] << "] "
            "->" << testRegressionMLP->getOutputsforRegression() << "< expected : 0" << std::endl;

    testRegressionMLP_oneInput[0] = 1;
    testRegressionMLP_oneInput[1] = 1;
    testRegressionMLP->predict(testRegressionMLP_oneInput,testRegressionMLP_inputSize);
    std::cout << "Response for input = [" << testRegressionMLP_oneInput[0] << "][" << testRegressionMLP_oneInput[1] << "] "
            "->" << testRegressionMLP->getOutputsforRegression() << "< expected : 0.5" << std::endl;
    return 1;
}