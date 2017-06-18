#pragma once
//
// Created by antoine on 14/06/2017.
//
#ifndef PERCEPTRON_RBF_H
#define PERCEPTRON_RBF_H
#include "Eigen/Dense"


class RBF {
public:


    static double* lloydAlgorithm(double* inputs, int inputSize, int nbData, int nbRepresentatives);
    static Eigen::MatrixXd learnModel(int nbExamples, double gamma, double* X, Eigen::MatrixXd Y);
    static Eigen::MatrixXd naiveLearnWeights(int nbExamples, double gamma, double* X, int inputSize, double* Y) ;
    static void getRBFResponse(Eigen::MatrixXd weights, double gamma, double* input, int inputSize, double* output, double* X, int nbExamples) ;


    int nbRepresentatives;
private:
    double* representatives;

    double distance(double * A, double* B, int inputSize) ;
    void showRepresentative(int inputSize){
        for(int i=0;i<RBF::nbRepresentatives;i+=inputSize){
            std::cout << "Representant " << i << " = (" << representatives[i] << ";" << representatives[i+1] << ")" << std::endl;
        }
    }

};


#endif //PERCEPTRON_RBF_H
