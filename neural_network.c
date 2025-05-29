#include "../include/neural_network.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

static double rand_weight() {
    return ((double)rand() / RAND_MAX) - 0.5;
}

void nn_init(NeuralNetwork *nn, size_t n_inputs,
             size_t n_hidden, size_t n_outputs,
             double learning_rate) {
    nn->n_inputs = n_inputs;
    nn->n_hidden = n_hidden;
    nn->n_outputs = n_outputs;
    nn->learning_rate = learning_rate;

    size_t ih = n_inputs * n_hidden;
    size_t ho = n_hidden * n_outputs;

    nn->weights_input_hidden = malloc(sizeof(double) * ih);
    nn->weights_hidden_output = malloc(sizeof(double) * ho);
    nn->bias_hidden = malloc(sizeof(double) * n_hidden);
    nn->bias_output = malloc(sizeof(double) * n_outputs);
    nn->hidden_layer = malloc(sizeof(double) * n_hidden);
    nn->output_layer = malloc(sizeof(double) * n_outputs);

    for (size_t i = 0; i < ih; ++i)
        nn->weights_input_hidden[i] = rand_weight();
    for (size_t i = 0; i < ho; ++i)
        nn->weights_hidden_output[i] = rand_weight();
    for (size_t i = 0; i < n_hidden; ++i)
        nn->bias_hidden[i] = rand_weight();
    for (size_t i = 0; i < n_outputs; ++i)
        nn->bias_output[i] = rand_weight();
}

void nn_free(NeuralNetwork *nn) {
    free(nn->weights_input_hidden);
    free(nn->weights_hidden_output);
    free(nn->bias_hidden);
    free(nn->bias_output);
    free(nn->hidden_layer);
    free(nn->output_layer);
}

void nn_feedforward(const NeuralNetwork *nn, const double *input) {
    // Input ==> Hidden
    for (size_t i = 0; i < nn->n_hidden; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < nn->n_inputs; ++j)
            sum += input[j] * nn->weights_input_hidden[j * nn->n_hidden + i];
        sum += nn->bias_hidden[i];
        nn->hidden_layer[i] = sigmoid(sum);
    }

    // Hidden ==> Output
    for (size_t i = 0; i < nn->n_outputs; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < nn->n_hidden; ++j)
            sum += nn->hidden_layer[j] * nn->weights_hidden_output[j * nn->n_outputs + i];
        sum += nn->bias_output[i];
        nn->output_layer[i] = sigmoid(sum);
    }
}

void nn_backpropagate(NeuralNetwork *nn,
                      const double *input,
                      const double *target) {
    // Feedforward to update nn->hidden_layer and nn->output_layer
    nn_feedforward(nn, input);

    // Output errors
    double *output_errors = malloc(sizeof(double) * nn->n_outputs);
    for (size_t i = 0; i < nn->n_outputs; ++i) {
        double o = nn->output_layer[i];
        output_errors[i] = (target[i] - o) * sigmoid_derivative(o);
    }

    // Hidden errors
    double *hidden_errors = malloc(sizeof(double) * nn->n_hidden);
    for (size_t i = 0; i < nn->n_hidden; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < nn->n_outputs; ++j)
            sum += output_errors[j] * nn->weights_hidden_output[i * nn->n_outputs + j];
        hidden_errors[i] = sum * sigmoid_derivative(nn->hidden_layer[i]);
    }

    // Update weights & biases
    for (size_t i = 0; i < nn->n_hidden; ++i)
        for (size_t j = 0; j < nn->n_outputs; ++j)
            nn->weights_hidden_output[i * nn->n_outputs + j] +=
                nn->learning_rate * output_errors[j] * nn->hidden_layer[i];

    for (size_t i = 0; i < nn->n_outputs; ++i)
        nn->bias_output[i] += nn->learning_rate * output_errors[i];

    // Update input→hidden weights & biases
    for (size_t i = 0; i < nn->n_inputs; ++i)
        for (size_t j = 0; j < nn->n_hidden; ++j)
            nn->weights_input_hidden[i * nn->n_hidden + j] +=
                nn->learning_rate * hidden_errors[j] * input[i];

    for (size_t i = 0; i < nn->n_hidden; ++i)
        nn->bias_hidden[i] += nn->learning_rate * hidden_errors[i];

    free(output_errors);
    free(hidden_errors);
}