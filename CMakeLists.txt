cmake_minimum_required(VERSION 3.7)
project(Perceptron)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp MLP.cpp MLP.h linearPerceptron.cpp linearPerceptron.h RBF.cpp RBF.h)
add_executable(Perceptron ${SOURCE_FILES})