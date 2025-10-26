/*

FUNCTION sigmoid(input_value)
    PURPOSE: Transform any real number to a value between 0 and 1
    
    STEP 1: Handle the exponential calculation
        NEGATE the input value (multiply by -1)
        STORE as negative_input
        
    STEP 2: Calculate e^(-x)
        CALL exponential function from math library
        PASS negative_input to it
        STORE result as exp_of_negative_input
        
    STEP 3: Calculate denominator
        ADD 1.0 to exp_of_negative_input
        STORE as denominator
        
    STEP 4: Calculate final result
        DIVIDE 1.0 by denominator
        STORE as sigmoid_result
        
    STEP 5: Return the result
        RETURN sigmoid_result
        
    EDGE CASES TO CONSIDER:
        - Very large positive input → result approaches 1
        - Very large negative input → result approaches 0
        - Input of 0 → result is exactly 0.5
END FUNCTION


FUNCTION sigmoid_derivative(input_value)
    PURPOSE: Calculate rate of change of sigmoid at a point
    
    STEP 1: Calculate sigmoid first
        CALL sigmoid function with input_value
        STORE result as sig_val
        
    STEP 2: Calculate (1 - sigmoid)
        SUBTRACT sig_val from 1.0
        STORE as one_minus_sig
        
    STEP 3: Multiply sigmoid by (1 - sigmoid)
        MULTIPLY sig_val by one_minus_sig
        STORE as derivative_result
        
    STEP 4: Return the derivative
        RETURN derivative_result
        
    WHY THIS FORMULA:
        - Derivative is maximum when sigmoid(x) = 0.5
        - Derivative approaches 0 for large positive/negative x
        - This is used in backpropagation for learning
END FUNCTION
```

## 2. Tanh Function - Detailed
```
FUNCTION tanh_activation(input_value)
    PURPOSE: Transform any real number to a value between -1 and 1
    
    METHOD 1 (From scratch):
    
    STEP 1: Calculate positive exponential
        CALL exponential function with input_value
        STORE as exp_positive
        
    STEP 2: Calculate negative exponential
        NEGATE input_value
        CALL exponential function with negated value
        STORE as exp_negative
        
    STEP 3: Calculate numerator
        SUBTRACT exp_negative from exp_positive
        STORE as numerator
        (This gives: e^x - e^(-x))
        
    STEP 4: Calculate denominator
        ADD exp_positive and exp_negative
        STORE as denominator
        (This gives: e^x + e^(-x))
        
    STEP 5: Calculate final result
        DIVIDE numerator by denominator
        STORE as tanh_result
        
    STEP 6: Return result
        RETURN tanh_result
        
    METHOD 2 (Using library):
    
    STEP 1: Call built-in tanh
        CALL hyperbolic tangent function from math library
        PASS input_value
        RETURN the result directly
        
    EDGE CASES:
        - Input of 0 → result is exactly 0
        - Large positive input → approaches 1
        - Large negative input → approaches -1
END FUNCTION


FUNCTION tanh_derivative(input_value)
    PURPOSE: Calculate rate of change of tanh at a point
    
    STEP 1: Calculate tanh first
        CALL tanh_activation with input_value
        STORE result as tanh_val
        
    STEP 2: Square the tanh value
        MULTIPLY tanh_val by itself
        STORE as tanh_squared
        
    STEP 3: Subtract from 1
        SUBTRACT tanh_squared from 1.0
        STORE as derivative_result
        (Formula: 1 - tanh²(x))
        
    STEP 4: Return the derivative
        RETURN derivative_result
        
    WHY THIS FORMULA:
        - Derivative is maximum (equals 1) when x = 0
        - Derivative approaches 0 for large magnitude x
        - Used in backpropagation for gradient descent
END FUNCTION
```

## 3. Matrix Operations - Detailed
```
STRUCTURE Matrix
    FIELD: number_of_rows (integer)
    FIELD: number_of_columns (integer)
    FIELD: data (2D array of floating-point numbers)
END STRUCTURE


FUNCTION create_matrix(rows, cols)
    PURPOSE: Allocate memory for a new matrix
    
    STEP 1: Allocate memory for matrix structure
        REQUEST memory for Matrix structure
        STORE pointer as new_matrix
        
    STEP 2: Set dimensions
        SET new_matrix.number_of_rows to rows
        SET new_matrix.number_of_columns to cols
        
    STEP 3: Allocate memory for row pointers
        REQUEST memory for array of row pointers (size = rows)
        STORE in new_matrix.data
        
    STEP 4: Allocate memory for each row
        FOR each row from 0 to rows - 1
            REQUEST memory for array of columns (size = cols)
            STORE in new_matrix.data[row]
        END FOR
        
    STEP 5: Initialize values to zero
        FOR each row from 0 to rows - 1
            FOR each column from 0 to cols - 1
                SET new_matrix.data[row][column] to 0.0
            END FOR
        END FOR
        
    STEP 6: Return the matrix
        RETURN new_matrix
END FUNCTION


FUNCTION apply_function_elementwise(input_matrix, function_to_apply)
    PURPOSE: Apply any function to every element in a matrix
    
    INPUT VALIDATION:
        IF input_matrix is NULL
            PRINT error message
            RETURN NULL
        END IF
        
    STEP 1: Get dimensions
        GET number_of_rows from input_matrix
        GET number_of_columns from input_matrix
        STORE these values
        
    STEP 2: Create output matrix
        CALL create_matrix with same dimensions
        STORE as output_matrix
        
    STEP 3: Process each element
        FOR row_index from 0 to number_of_rows - 1
            FOR col_index from 0 to number_of_columns - 1
                
                GET input value at [row_index][col_index]
                STORE as current_value
                
                CALL function_to_apply with current_value
                STORE result as transformed_value
                
                SET output_matrix.data[row_index][col_index] 
                    to transformed_value
                    
            END FOR
        END FOR
        
    STEP 4: Return result
        RETURN output_matrix
        
    MEMORY NOTE: Caller is responsible for freeing output_matrix
END FUNCTION


FUNCTION apply_sigmoid_to_matrix(input_matrix)
    PURPOSE: Convenience function for sigmoid on matrices
    
    STEP 1: Call general elementwise function
        CALL apply_function_elementwise
        PASS input_matrix
        PASS sigmoid function as second parameter
        STORE result as sigmoid_matrix
        
    STEP 2: Return result
        RETURN sigmoid_matrix
END FUNCTION


FUNCTION apply_tanh_to_matrix(input_matrix)
    PURPOSE: Convenience function for tanh on matrices
    
    STEP 1: Call general elementwise function
        CALL apply_function_elementwise
        PASS input_matrix
        PASS tanh_activation function as second parameter
        STORE result as tanh_matrix
        
    STEP 2: Return result
        RETURN tanh_matrix
END FUNCTION


FUNCTION apply_sigmoid_derivative_to_matrix(input_matrix)
    PURPOSE: Apply sigmoid derivative to entire matrix
    
    STEP 1: Call general elementwise function
        CALL apply_function_elementwise
        PASS input_matrix
        PASS sigmoid_derivative function as second parameter
        STORE result as derivative_matrix
        
    STEP 2: Return result
        RETURN derivative_matrix
END FUNCTION


FUNCTION apply_tanh_derivative_to_matrix(input_matrix)
    PURPOSE: Apply tanh derivative to entire matrix
    
    STEP 1: Call general elementwise function
        CALL apply_function_elementwise
        PASS input_matrix
        PASS tanh_derivative function as second parameter
        STORE result as derivative_matrix
        
    STEP 2: Return result
        RETURN derivative_matrix
END FUNCTION


FUNCTION free_matrix(matrix)
    PURPOSE: Clean up memory allocated for matrix
    
    VALIDATION:
        IF matrix is NULL
            RETURN immediately
        END IF
        
    STEP 1: Free each row
        FOR row_index from 0 to matrix.number_of_rows - 1
            IF matrix.data[row_index] is not NULL
                FREE memory at matrix.data[row_index]
            END IF
        END FOR
        
    STEP 2: Free row pointer array
        IF matrix.data is not NULL
            FREE memory at matrix.data
        END IF
        
    STEP 3: Free matrix structure
        FREE memory at matrix
END FUNCTION
```

## 4. Testing Functions - Detailed
```
FUNCTION test_sigmoid_basic()
    PURPOSE: Test sigmoid with known values
    
    DECLARE tolerance as 0.0001 (acceptable error margin)
    DECLARE test_passed_count as 0
    DECLARE test_failed_count as 0
    
    TEST 1: Sigmoid of zero
        CALL sigmoid(0.0)
        STORE result
        CALCULATE absolute difference from 0.5
        IF difference is less than tolerance
            PRINT "✓ Test 1 passed: sigmoid(0) = 0.5"
            INCREMENT test_passed_count
        ELSE
            PRINT "✗ Test 1 failed: sigmoid(0) = " + result
            INCREMENT test_failed_count
        END IF
        
    TEST 2: Sigmoid of large positive number
        CALL sigmoid(10.0)
        STORE result
        IF result is greater than 0.99
            PRINT "✓ Test 2 passed: sigmoid(10) ≈ 1"
            INCREMENT test_passed_count
        ELSE
            PRINT "✗ Test 2 failed: sigmoid(10) = " + result
            INCREMENT test_failed_count
        END IF
        
    TEST 3: Sigmoid of large negative number
        CALL sigmoid(-10.0)
        STORE result
        IF result is less than 0.01
            PRINT "✓ Test 3 passed: sigmoid(-10) ≈ 0"
            INCREMENT test_passed_count
        ELSE
            PRINT "✗ Test 3 failed: sigmoid(-10) = " + result
            INCREMENT test_failed_count
        END IF
        
    TEST 4: Sigmoid symmetry
        CALL sigmoid(2.0)
        STORE as result_pos
        CALL sigmoid(-2.0)
        STORE as result_neg
        CALCULATE sum as result_pos + result_neg
        CALCULATE absolute difference from 1.0
        IF difference is less than tolerance
            PRINT "✓ Test 4 passed: sigmoid symmetry holds"
            INCREMENT test_passed_count
        ELSE
            PRINT "✗ Test 4 failed: symmetry broken"
            INCREMENT test_failed_count
        END IF
        
    PRINT summary
        PRINT "Sigmoid tests: " + test_passed_count + " passed, " 
              + test_failed_count + " failed"
END FUNCTION


FUNCTION test_tanh_basic()
    PURPOSE: Test tanh with known values
    
    DECLARE tolerance as 0.0001
    DECLARE test_passed_count as 0
    DECLARE test_failed_count as 0
    
    TEST 1: Tanh of zero
        CALL tanh_activation(0.0)
        STORE result
        CALCULATE absolute value of result
        IF absolute value is less than tolerance
            PRINT "✓ Test 1 passed: tanh(0) = 0"
            INCREMENT test_passed_count
        ELSE
            PRINT "✗ Test 1 failed: tanh(0) = " + result
            INCREMENT test_failed_count
        END IF
        
    TEST 2: Tanh of 1
        CALL tanh_activation(1.0)
        STORE result
        CALCULATE absolute difference from 0.7616
        IF difference is less than 0.001
            PRINT "✓ Test 2 passed: tanh(1) ≈ 0.7616"
            INCREMENT test_passed_count
        ELSE
            PRINT "✗ Test 2 failed: tanh(1) = " + result
            INCREMENT test_failed_count
        END IF
        
    TEST 3: Tanh antisymmetry
        CALL tanh_activation(2.0)
        STORE as result_pos
        CALL tanh_activation(-2.0)
        STORE as result_neg
        CALCULATE sum as result_pos + result_neg
        CALCULATE absolute value of sum
        IF absolute value is less than tolerance
            PRINT "✓ Test 3 passed: tanh antisymmetry holds"
            INCREMENT test_passed_count
        ELSE
            PRINT "✗ Test 3 failed: antisymmetry broken"
            INCREMENT test_failed_count
        END IF
        
    TEST 4: Tanh bounds
        CALL tanh_activation(100.0)
        STORE result
        IF result is between 0.99 and 1.01
            PRINT "✓ Test 4 passed: tanh approaches 1"
            INCREMENT test_passed_count
        ELSE
            PRINT "✗ Test 4 failed: tanh(100) = " + result
            INCREMENT test_failed_count
        END IF
        
    PRINT summary
        PRINT "Tanh tests: " + test_passed_count + " passed, " 
              + test_failed_count + " failed"
END FUNCTION


FUNCTION test_derivative_numerically(test_function, derivative_function, x_value)
    PURPOSE: Verify derivative formula using numerical approximation
    
    STEP 1: Set up parameters
        SET step_size to 0.00001 (very small h)
        SET tolerance to 0.001 (acceptable error)
        
    STEP 2: Calculate numerical derivative
        CALL test_function(x_value + step_size)
        STORE as f_plus
        
        CALL test_function(x_value - step_size)
        STORE as f_minus
        
        CALCULATE difference as (f_plus - f_minus)
        CALCULATE divisor as (2.0 * step_size)
        DIVIDE difference by divisor
        STORE as numerical_derivative
        
    STEP 3: Calculate analytical derivative
        CALL derivative_function(x_value)
        STORE as analytical_derivative
        
    STEP 4: Compare results
        CALCULATE absolute difference between the two derivatives
        STORE as error
        
        IF error is less than tolerance
            PRINT "✓ Derivative correct at x = " + x_value
            PRINT "  Numerical: " + numerical_derivative
            PRINT "  Analytical: " + analytical_derivative
            PRINT "  Error: " + error
            RETURN true
        ELSE
            PRINT "✗ Derivative mismatch at x = " + x_value
            PRINT "  Numerical: " + numerical_derivative
            PRINT "  Analytical: " + analytical_derivative
            PRINT "  Error: " + error
            RETURN false
        END IF
        
    EXPLANATION OF METHOD:
        The numerical derivative approximates f'(x) using:
        f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        This is called the central difference method
        It's more accurate than forward/backward differences
END FUNCTION


FUNCTION test_all_derivatives()
    PURPOSE: Test derivatives at multiple points
    
    DECLARE test_points array = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    DECLARE passed_count as 0
    DECLARE total_tests as 0
    
    PRINT "Testing Sigmoid Derivatives:"
    FOR each x_value in test_points
        INCREMENT total_tests
        CALL test_derivative_numerically(sigmoid, sigmoid_derivative, x_value)
        IF result is true
            INCREMENT passed_count
        END IF
    END FOR
    
    PRINT blank line
    PRINT "Testing Tanh Derivatives:"
    FOR each x_value in test_points
        INCREMENT total_tests
        CALL test_derivative_numerically(tanh_activation, tanh_derivative, x_value)
        IF result is true
            INCREMENT passed_count
        END IF
    END FOR
    
    PRINT summary
        PRINT total_tests + " derivative tests run"
        PRINT passed_count + " tests passed"
        PRINT (total_tests - passed_count) + " tests failed"
END FUNCTION


FUNCTION test_matrix_operations()
    PURPOSE: Test that matrix operations work correctly
    
    STEP 1: Create test matrix
        CALL create_matrix(3, 3)
        STORE as test_matrix
        
    STEP 2: Fill with test values
        SET test_matrix.data[0][0] to -2.0
        SET test_matrix.data[0][1] to -1.0
        SET test_matrix.data[0][2] to 0.0
        SET test_matrix.data[1][0] to 1.0
        SET test_matrix.data[1][1] to 2.0
        SET test_matrix.data[1][2] to 3.0
        SET test_matrix.data[2][0] to -0.5
        SET test_matrix.data[2][1] to 0.5
        SET test_matrix.data[2][2] to 1.5
        
    STEP 3: Test sigmoid on matrix
        PRINT "Testing sigmoid matrix operation..."
        CALL apply_sigmoid_to_matrix(test_matrix)
        STORE as sigmoid_result
        
        Verify spot checks:
        CALL sigmoid(0.0) for comparison
        COMPARE with sigmoid_result.data[0][2]
        IF values match within tolerance
            PRINT "✓ Sigmoid matrix center element correct"
        ELSE
            PRINT "✗ Sigmoid matrix operation failed"
        END IF
        
    STEP 4: Test tanh on matrix
        PRINT "Testing tanh matrix operation..."
        CALL apply_tanh_to_matrix(test_matrix)
        STORE as tanh_result
        
        Verify spot checks:
        CALL tanh_activation(0.0) for comparison
        COMPARE with tanh_result.data[0][2]
        IF values match within tolerance
            PRINT "✓ Tanh matrix center element correct"
        ELSE
            PRINT "✗ Tanh matrix operation failed"
        END IF
        
    STEP 5: Test derivatives on matrix
        PRINT "Testing derivative matrix operations..."
        CALL apply_sigmoid_derivative_to_matrix(test_matrix)
        STORE as sigmoid_deriv_result
        
        CALL apply_tanh_derivative_to_matrix(test_matrix)
        STORE as tanh_deriv_result
        
        Verify that all values are positive (derivatives should be)
        SET all_positive to true
        FOR each row
            FOR each column
                IF sigmoid_deriv_result.data[row][col] is not positive
                    SET all_positive to false
                END IF
                IF tanh_deriv_result.data[row][col] is not positive
                    SET all_positive to false
                END IF
            END FOR
        END FOR
        
        IF all_positive is true
            PRINT "✓ All derivative values are positive as expected"
        ELSE
            PRINT "✗ Some derivative values are negative"
        END IF
        
    STEP 6: Clean up memory
        CALL free_matrix(test_matrix)
        CALL free_matrix(sigmoid_result)
        CALL free_matrix(tanh_result)
        CALL free_matrix(sigmoid_deriv_result)
        CALL free_matrix(tanh_deriv_result)
        
    PRINT "Matrix operation tests complete"
END FUNCTION


FUNCTION main_test_runner()
    PURPOSE: Run all tests in organized fashion
    
    PRINT "========================================="
    PRINT "Activation Functions Test Suite"
    PRINT "========================================="
    PRINT blank line
    
    PRINT "PART 1: Basic Function Tests"
    PRINT "-----------------------------------------"
    CALL test_sigmoid_basic()
    PRINT blank line
    CALL test_tanh_basic()
    PRINT blank line
    
    PRINT "PART 2: Derivative Verification"
    PRINT "-----------------------------------------"
    CALL test_all_derivatives()
    PRINT blank line
    
    PRINT "PART 3: Matrix Operations"
    PRINT "-----------------------------------------"
    CALL test_matrix_operations()
    PRINT blank line
    
    PRINT "========================================="
    PRINT "All tests complete!"
    PRINT "========================================="
END FUNCTION
```

## 5. Helper Functions
```
FUNCTION print_matrix(matrix, label)
    PURPOSE: Display matrix contents for debugging
    
    PRINT label + ":"
    FOR row from 0 to matrix.number_of_rows - 1
        PRINT "  ["
        FOR col from 0 to matrix.number_of_columns - 1
            PRINT matrix.data[row][col] with formatting
            IF col is not the last column
                PRINT ", "
            END IF
        END FOR
        PRINT "]"
    END FOR
    PRINT blank line
END FUNCTION


FUNCTION absolute_value(number)
    PURPOSE: Get absolute value of a number
    
    IF number is less than 0
        RETURN negative of number
    ELSE
        RETURN number
    END IF
END FUNCTION


FUNCTION compare_floats(value1, value2, tolerance)
    PURPOSE: Compare two floating-point numbers with tolerance
    
    CALCULATE difference as value1 - value2
    CALL absolute_value on difference
    STORE as abs_difference
    
    IF abs_difference is less than or equal to tolerance
        RETURN true (they are equal within tolerance)
    ELSE
        RETURN false (they are different)
    END IF
END FUNCTION

*/
