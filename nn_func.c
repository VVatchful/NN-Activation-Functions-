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
    exp_of_negative = exp(negative_input);
    denominator = 1.0 + exp_of_negative;
    sigmoid_result = 1.0 / denominator;
    return sigmoid_result;
}

double sigmoid_error_handl(double input_value) {
    double negative_input;
    double exp_of_negative;
    double denominator;
    double sigmoid_result;

    if (isnan(input_value)) {
        printf("%%1f Error: Input is NaN\n");
        return NAN;
    }
    if (isinf(input_value)) {
        if (input_value > 0.0) {
            return 1.0;
        } else {
            return 0.0;
        }
    }
    if (input_value > 20.0) {
        return 1.0;
    }
    if (input_value < -20.0) {
        return 0.0;
    }

    negative_input = input_value * -1.0;
    exp_of_negative = exp(negative_input);
    denominator = 1.0 + exp_of_negative;
    sigmoid_result = 1.0 / denominator;

    return sigmoid_result;
}

double sigmoid_derivative(double input_value) {
    double sig_val = sigmoid(input_value);
    return sig_val * (1.0 - sig_val);
}   

double tanh_activation(double input_value) {
    double tanh_result;
    tanh_result = tanh(input_value);
    return tanh_result;

}

double tanh_derivative(double input_value) {
    double tanh_val;
    double tanh_squared;
    double derivative_result;

    tanh_val = tanh_activation(input_value);

    tanh_squared = tanh_val * tanh_val;
    derivative_result = 1.0 - tanh_squared;
    
    return derivative_result;
}


