////
//// Created by antoine on 14/06/2017.
////
//#include <cstdlib>
//#include <tgmath.h>
//#include "RBF.h"
//#include "Eigen/src/Core/Matrix.h"
//#include "Eigen/Dense"
//class MatrixXd;
//
////double** initRbf(int inputSize, int nbExamples){
////    double** weights = new double*[inputSize];
////    for (int input = 0; input < inputSize; ++input) {
////        weights[input] = new double[nbExamples];
////        for (int example = 0; example < nbExamples; ++example) {
////            weights[input][example] = ((float) rand()) / ((float) RAND_MAX) * 2.0 - 1.0;
////        }
////    }
////    return weights;
////}
//
//int lloydAlgorithm(double* inputs, int inputsSize, double* representative){
//    representative = new double[inputsSize];
//    int nbRepresentatives=0;
//    while(1){
//        // elect n representatives
//    }
//    return nbRepresentatives;
//}
////void classify(double* input, int inputSize, int* outputs, int outputSize, int gamma, double** weights, int nbNeurons){
////    double calc=0;
////    for(int i=0; i<outputSize;i++){
////        for (int j = 1; j < nbNeurons; ++j) {
////            calc += weights[j][?]* exp(-gamma*(X-X[nbNeurons])*(X-X[nbNeurons]));
////        }
////        outputs[i] = (calc < 0) ? -1 : 1;
////        calc=0;
////    }
////}
////void predict(double* input, int inputSize, int* outputs, int outputSize, int gamma, double** weights, int nbNeurons){
////    double calc=0;
////    for(int i=0; i<outputSize;i++){
////        for (int j = 1; j < nbNeurons; ++j) {
////            calc += weights[j][?]* exp(-gamma*(X-X[nbNeurons])*(X-X[nbNeurons]));
////        }
////        outputs[i] = calc;
////        calc=0;
////    }
////}
//Eigen::MatrixXd* learnModel(int nbExamples, double gamma, double* X, Eigen::MatrixXd Y){
//    Eigen::MatrixXd teta(nbExamples,nbExamples);
//    for (int i = 0; i < nbExamples; ++i) {
//        for (int j = 0; j < nbExamples; ++j) {
//            teta(i,j) = exp(-gamma*(X[i]-X[j])*(X[i]-X[j]));
//        }
//    }
//
//    return new Eigen::MatrixXd = teta.inverse()*Y;
//}
//Eigen::MatrixXd* naiveLearnModel(int nbExamples, double gamma, double* X, Eigen::MatrixXd Y){
//    Eigen::MatrixXd teta(nbExamples,nbExamples);
//    for (int i = 0; i < nbExamples; ++i) {
//        for (int j = 0; j < nbExamples; ++j) {
//            teta(i,j) = exp(-gamma*(X[i]-X[j])*(X[i]-X[j]));
//        }
//    }
//
//    return new Eigen::MatrixXd = teta.inverse()*Y;
//}
