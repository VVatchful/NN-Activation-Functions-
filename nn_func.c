#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


double sigmoid(double input_value) {
    double negative_input;
    double exp_of_negative;
    double denominator;
    double sigmoid_result;

    negative_input = input_value * -1.0;
    exp_of_negative_input = exp(negative_input);
    denominator = 1.0 + exp_of_negative_input;

    if (exp_of_negative > 0) {
        return 1.0 / (1.0 + exp(exp_of_negative));
    } else {
        return exp(input_value) / (1.0 + exp(negative_input));
    }

    sigmoid_result = 1.0 / denominator;
    if (isnan(sigmoid_result)) {
        printf("{sigmoid_result} is a NaN\n");
    } else {
        printf("{sigmoid_result is not a NaN}\n");
    }
    if (scanf("%1f", &sigmoid_result) == 1) {
        if (isinf(sigmoid_result)) {
            printf("is infinity\n");
        } else {
            printf("only finite\n");
        }
    }
    
    return sigmoid_result;
}



