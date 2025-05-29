#include <stdio.h>
#include <stdlib.h>
#include "../include/neural_network.h"

#define MAX_SAMPLES 1000

int main(void) {
    srand(42);

    FILE *fp = fopen("data/training_data_sample.txt", "r");
    if (!fp) { perror("Failed to open training_data.txt"); return 1; }

    size_t sample_count = 0;
    double inputs[MAX_SAMPLES][2];
    double targets[MAX_SAMPLES][1];
    char line[256];

    // Skip first line since it's an header
    if (fgets(line, sizeof line, fp) == NULL) return 1;

    while (fgets(line, sizeof line, fp) && sample_count < MAX_SAMPLES) {
        double in0, in1, out0;
        if (sscanf(line, "%lf,%lf,%lf", &in0, &in1, &out0) == 3) {
            inputs[sample_count][0] = in0;
            inputs[sample_count][1] = in1;
            targets[sample_count][0] = out0;
            sample_count++;
        }
    }
    fclose(fp);

    // Network creation
    NeuralNetwork nn;
    nn_init(&nn, 2, 2, 1, 0.5);

    for (size_t epoch = 0; epoch < 10000; ++epoch) {
        size_t idx = epoch % sample_count;
        nn_backpropagate(&nn, inputs[idx], targets[idx]);
    }

    // Test on loaded data
    printf("Results from training data:
");
    for (size_t i = 0; i < sample_count; ++i) {
        nn_feedforward(&nn, inputs[i]);
        printf("Input: %g, %g -> %.3f (target: %.0f)
",
               inputs[i][0], inputs[i][1], nn.output_layer[0], targets[i][0]);
    }

    nn_free(&nn);
    return 0;
}

