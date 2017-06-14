#pragma once
#include "Eigen/Dense"
//
// Created by antoine on 14/06/2017.
//
#define DllExport   __declspec( dllexport )

class DllExport ILinearPerceptron
{
public:
    virtual void linear_remove_model(double *model) = 0;
    virtual int linear_fit_classification_rosenblatt(double *model, double *inputs, int inputsSize, int inputSize,
                                             double *expectedOutputs, int outputSize, int iterationMax, double step) = 0;
    virtual Eigen::MatrixXd* linear_fit_regression(double *inputs, int inputsSize, int inputSize,
                                           double *expectedOutputs, int outputSize) = 0;
    virtual void linear_classify(double *model, double* input, int inputSize, double* output, int outputDimension) = 0;
    virtual void linearPredict(Eigen::MatrixXd* model, double* input, int inputSize, double* output, int outputSize) = 0;
};

class DllExport LinearPerceptron: public ILinearPerceptron {
public:
    LinearPerceptron(int inputDimension, int outputDimension);

    void* getPointer(int inputDimension, int outputDimension);
    void linear_remove_model(double *model);
    int linear_fit_classification_rosenblatt(double *model, double *inputs, int inputsSize, int inputSize, double *expectedOutputs, int outputSize, int iterationMax, double step);
    Eigen::MatrixXd* linear_fit_regression(double *inputs, int inputsSize, int inputSize, double *expectedOutputs, int outputSize);
    void linear_classify(double *model, double* input, int inputSize, double* output, int outputDimension);
    void linearPredict(Eigen::MatrixXd* model, double* input, int inputSize, double* output, int outputSize);

private:
    double* addBiasToInput(double *input, int inputSize);
    double* addBiasToInputs(double *inputs, int *inputsSize, int *inputSize) ;
    void tabToMatrix(Eigen::MatrixXd* matrix, double* tab, int nbRow, int nbCols);
    void matrixToTab(Eigen::MatrixXd matrix, double *tab, int nbRow, int nbCols);
};
//__declspec(dllexport) ILinearPerceptron *Create(int inputDimension, int outputDimension)
//{
//    return new LinearPerceptron(inputDimension, outputDimension);
//}
