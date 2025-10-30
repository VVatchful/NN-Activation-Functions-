/*


FUNCTION apply_activation_elementwise(matrix, activation_function):
    result_matrix = create_matrix(matrix.rows, matrix.cols)
    
    FOR i FROM 0 TO matrix.rows - 1:
        FOR j FROM 0 TO matrix.cols - 1:
            result_matrix.data[i][j] = activation_function(matrix.data[i][j])
        END FOR
    END FOR
    
    RETURN result_matrix
END FUNCTION
```

## **Element-wise Derivative Application**
```
FUNCTION apply_derivative_elementwise(matrix, derivative_function):
    result_matrix = create_matrix(matrix.rows, matrix.cols)
    
    FOR i FROM 0 TO matrix.rows - 1:
        FOR j FROM 0 TO matrix.cols - 1:
            result_matrix.data[i][j] = derivative_function(matrix.data[i][j])
        END FOR
    END FOR
    
    RETURN result_matrix
END FUNCTION
```

## **Hadamard Product (Element-wise Multiplication)**
```
FUNCTION hadamard_product(A, B):
    IF A.rows != B.rows OR A.cols != B.cols THEN
        PRINT "Matrices must have same dimensions"
        RETURN NULL
    END IF
    
    result = create_matrix(A.rows, A.cols)
    
    FOR i FROM 0 TO A.rows - 1:
        FOR j FROM 0 TO A.cols - 1:
            result.data[i][j] = A.data[i][j] * B.data[i][j]
        END FOR
    END FOR
    
    RETURN result
END FUNCTION
```

## **Softmax for Vector (Single Row/Column)**
```
FUNCTION apply_softmax_to_vector(matrix):
    // Assumes matrix is either 1xN or Nx1
    IF matrix.rows != 1 AND matrix.cols != 1 THEN
        PRINT "Input must be a vector"
        RETURN NULL
    END IF
    
    length = MAX(matrix.rows, matrix.cols)
    result = create_matrix(matrix.rows, matrix.cols)
    
    // Find max value for numerical stability
    max_val = matrix.data[0][0]
    FOR i FROM 0 TO length - 1:
        current_val = GET_ELEMENT(matrix, i)
        IF current_val > max_val THEN
            max_val = current_val
        END IF
    END FOR
    
    // Calculate sum of exponentials
    sum_exp = 0.0
    FOR i FROM 0 TO length - 1:
        current_val = GET_ELEMENT(matrix, i)
        exp_val = exp(current_val - max_val)
        sum_exp = sum_exp + exp_val
    END FOR
    
    // Calculate softmax values
    FOR i FROM 0 TO length - 1:
        current_val = GET_ELEMENT(matrix, i)
        exp_val = exp(current_val - max_val)
        softmax_val = exp_val / sum_exp
        SET_ELEMENT(result, i, softmax_val)
    END FOR
    
    RETURN result
END FUNCTION
```

## **Batch Softmax (for multiple vectors)**
```
FUNCTION apply_softmax_batch(matrix, axis):
    // axis = 0 means apply along columns (each column is a sample)
    // axis = 1 means apply along rows (each row is a sample)
    
    result = create_matrix(matrix.rows, matrix.cols)
    
    IF axis == 1 THEN
        // Apply softmax to each row
        FOR i FROM 0 TO matrix.rows - 1:
            row_vector = extract_row(matrix, i)
            softmax_row = apply_softmax_to_vector(row_vector)
            insert_row(result, i, softmax_row)
            dealloc_matrix(row_vector)
            dealloc_matrix(softmax_row)
        END FOR
    ELSE
        // Apply softmax to each column
        FOR j FROM 0 TO matrix.cols - 1:
            col_vector = extract_column(matrix, j)
            softmax_col = apply_softmax_to_vector(col_vector)
            insert_column(result, j, softmax_col)
            dealloc_matrix(col_vector)
            dealloc_matrix(softmax_col)
        END FOR
    END IF
    
    RETURN result
END FUNCTION
```

## **Helper Functions Needed**
```
FUNCTION extract_row(matrix, row_index):
    result = create_matrix(1, matrix.cols)
    FOR j FROM 0 TO matrix.cols - 1:
        result.data[0][j] = matrix.data[row_index][j]
    END FOR
    RETURN result
END FUNCTION

FUNCTION extract_column(matrix, col_index):
    result = create_matrix(matrix.rows, 1)
    FOR i FROM 0 TO matrix.rows - 1:
        result.data[i][0] = matrix.data[i][col_index]
    END FOR
    RETURN result
END FUNCTION

FUNCTION insert_row(dest_matrix, row_index, source_row):
    FOR j FROM 0 TO dest_matrix.cols - 1:
        dest_matrix.data[row_index][j] = source_row.data[0][j]
    END FOR
END FUNCTION

FUNCTION insert_column(dest_matrix, col_index, source_col):
    FOR i FROM 0 TO dest_matrix.rows - 1:
        dest_matrix.data[i][col_index] = source_col.data[i][0]
    END FOR
END FUNCTION
```

## **Clipping Function (for gradient stability)**
```
FUNCTION clip_matrix(matrix, min_val, max_val):
    result = create_matrix(matrix.rows, matrix.cols)
    
    FOR i FROM 0 TO matrix.rows - 1:
        FOR j FROM 0 TO matrix.cols - 1:
            value = matrix.data[i][j]
            IF value < min_val THEN
                result.data[i][j] = min_val
            ELSE IF value > max_val THEN
                result.data[i][j] = max_val
            ELSE
                result.data[i][j] = value
            END IF
        END FOR
    END FOR
    
    RETURN result
END FUNCTION

*/
