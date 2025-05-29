#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stddef.h>

typedef struct {
    size_t n_inputs;
    size_t n_hidden;
    size_t n_outputs;
    double learning_rate;

    double *weights_input_hidden;  // size: n_inputs * n_hidden
    double *weights_hidden_output; // size: n_hidden * n_outputs
    double *bias_hidden;           // size: n_hidden
    double *bias_output;           // size: n_outputs

    double *hidden_layer;          // size: n_hidden
    double *output_layer;          // size: n_outputs
} NeuralNetwork;

// Initialize network with random weights/biases
void nn_init(NeuralNetwork *nn, size_t n_inputs,
             size_t n_hidden, size_t n_outputs,
             double learning_rate);

// Free allocated memory
void nn_free(NeuralNetwork *nn);

// Forward pass
void nn_feedforward(const NeuralNetwork *nn, const double *input);

// Backpropagation update using single target vector
void nn_backpropagate(NeuralNetwork *nn,
                      const double *input,
                      const double *target);

// Utility: sigmoid and derivative
static inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
static inline double sigmoid_derivative(double y) {
    return y * (1.0 - y);
}

#endif // NEURAL_NETWORK_H