/*

FUNCTION sigmoid(input_value)
    PURPOSE: Transform any real number to a value between 0 and 1
    MATHEMATICAL FORMULA: σ(x) = 1 / (1 + e^(-x))
    
    INPUT PARAMETERS:
        input_value: A floating-point number (double precision recommended)
                    Can be any real number from -∞ to +∞
    
    RETURN VALUE:
        A floating-point number strictly between 0 and 1 (exclusive)
        Never exactly 0 or 1 due to floating-point arithmetic
    
    VARIABLE DECLARATIONS:
        DECLARE negative_input as double precision floating-point

        DECLARE denominator as double precision floating-point
        DECLARE sigmoid_result as double precision floating-point
    
    STEP 1: Negate the input value
        PURPOSE: Prepare for exponential calculation
        RATIONALE: The sigmoid formula uses e^(-x), so we need -x
        
        CALCULATE negative_input = input_value * -1.0
        
        ALTERNATIVE APPROACH:
            negative_input = 0.0 - input_value
        
        NOTE: Using -1.0 (not -1) ensures floating-point multiplication
        
    STEP 2: Calculate exponential of negative input
        PURPOSE: Compute e^(-x) where e ≈ 2.71828
        
        CALL exp() function from <math.h> library
        PASS negative_input as argument
        STORE returned value in exp_of_negative_input
        
        IMPLEMENTATION NOTE:
            In C: exp_of_negative_input = exp(negative_input)
            The exp() function is part of standard math library
            
        NUMERICAL CONSIDERATION:
            - For very large positive input_value (e.g., x = 100)
              negative_input will be -100
              exp(-100) will be extremely close to 0
              This is numerically stable
              
            - For very large negative input_value (e.g., x = -100)
              negative_input will be 100
              exp(100) will be approximately 2.688×10^43
              This can cause overflow issues
              
    STEP 3: Calculate the denominator
        PURPOSE: Complete the formula's denominator (1 + e^(-x))
        
        CALCULATE denominator = 1.0 + exp_of_negative_input
        
        NOTE: Use 1.0 (not 1) to ensure floating-point arithmetic
        
        SPECIAL CASES:
            - If exp_of_negative_input ≈ 0 (large positive x)
              Then denominator ≈ 1.0
              
            - If exp_of_negative_input is very large (large negative x)
              Then denominator will be dominated by exp_of_negative_input
              
    STEP 4: Calculate the final sigmoid value
        PURPOSE: Complete the division to get final result
        
        CALCULATE sigmoid_result = 1.0 / denominator
        
        MATHEMATICAL INSIGHT:
            When x is large positive:
                e^(-x) ≈ 0, so 1/(1+0) = 1
            When x is large negative:
                e^(-x) is huge, so 1/(1+huge) ≈ 0
            When x = 0:
                e^0 = 1, so 1/(1+1) = 0.5
                
    STEP 5: Return the computed value
        RETURN sigmoid_result
        
    ERROR HANDLING CONSIDERATIONS:
        - Check for NaN (Not a Number) input
        - Check for infinity input
        - Handle potential overflow in exp() calculation
        
    OPTIONAL ENHANCED VERSION WITH OVERFLOW PROTECTION:
        
        IF input_value > 20.0
            COMMENT: For x > 20, sigmoid(x) is so close to 1.0
            COMMENT: that we can return 1.0 directly
            RETURN 1.0
        END IF
        
        IF input_value < -20.0
            COMMENT: For x < -20, sigmoid(x) is so close to 0.0
            COMMENT: that we can return 0.0 directly
            RETURN 0.0
        END IF
        
        THEN proceed with normal calculation
        
    ALTERNATIVE IMPLEMENTATION FOR NUMERICAL STABILITY:
        
        COMMENT: This version avoids overflow for negative inputs
        
        IF input_value >= 0.0
            COMMENT: Use standard formula for non-negative inputs
            CALCULATE exp_neg_x = exp(-input_value)
            CALCULATE result = 1.0 / (1.0 + exp_neg_x)
            RETURN result
        ELSE
            COMMENT: Use algebraically equivalent form for negative inputs
            COMMENT: σ(x) = e^x / (1 + e^x) when x < 0
            CALCULATE exp_x = exp(input_value)
            CALCULATE result = exp_x / (1.0 + exp_x)
            RETURN result
        END IF
        
    OPTIMIZATION NOTES:
        - The exp() function is computationally expensive
        - For batch processing, consider vectorization
        - Modern CPUs may have SIMD instructions for exponentials
        - Lookup tables can be used for approximate but faster results
        
    EDGE CASE TESTING VALUES:
        Test with x = 0: Should return exactly 0.5
        Test with x = 1: Should return ≈ 0.7310585786
        Test with x = -1: Should return ≈ 0.2689414214
        Test with x = 10: Should return ≈ 0.9999546021
        Test with x = -10: Should return ≈ 0.0000453979
        Test with x = 100: Should return ≈ 1.0 (within machine precision)
        Test with x = -100: Should return ≈ 0.0 (within machine precision)
        
    MATHEMATICAL PROPERTIES TO VERIFY:
        1. Symmetry: σ(x) + σ(-x) = 1.0
        2. Monotonicity: If x1 < x2, then σ(x1) < σ(x2)
        3. Range: 0 < σ(x) < 1 for all real x
        4. Midpoint: σ(0) = 0.5 exactly
        
    MEMORY CONSIDERATIONS:
        - All variables should be automatic (stack-allocated)
        - Total stack space: approximately 32 bytes (4 doubles)
        - No dynamic memory allocation needed
        - Function is thread-safe (no global state)
        
    PERFORMANCE CHARACTERISTICS:
        - Time complexity: O(1) - constant time
        - Primary cost: one exp() function call
        - Typical execution: ~20-50 CPU cycles (varies by processor)
        - No branching in basic version (good for pipelining)
        
END FUNCTION
```

## Additional Context and Considerations
```
MATHEMATICAL BACKGROUND:
    The sigmoid function is also known as:
    - Logistic function
    - Expit function
    - Soft step function
    
    Historical context:
    - Introduced by Pierre François Verhulst in 1838
    - Used for modeling population growth
    - Now central to neural networks and logistic regression
    
    Why values between 0 and 1:
    - Useful for probability interpretation
    - Output can represent confidence or activation level
    - Smooth gradient everywhere (differentiable)
    
RELATIONSHIP TO OTHER FUNCTIONS:
    - Inverse: logit function (log-odds)
    - Related: tanh(x) = 2·σ(2x) - 1
    - Softmax is multi-class generalization
    
PRECISION CONSIDERATIONS:
    For single precision (float):
    - Noticeable precision loss beyond |x| > 15
    
    For double precision (double):
    - Maintains precision up to |x| ≈ 35
    - Beyond this, function is effectively 0 or 1
    
    Recommended: Always use double precision for training
                 Can use float for inference if memory constrained

===================================================================

FUNCTION sigmoid_derivative(input_value)
    PURPOSE: Calculate the rate of change (gradient) of sigmoid at a specific point
    MATHEMATICAL FORMULA: σ'(x) = σ(x) · (1 - σ(x))
    
    INPUT PARAMETERS:
        input_value: A floating-point number (double precision recommended)
                    Can be any real number from -∞ to +∞
    
    RETURN VALUE:
        A floating-point number between 0 and 0.25 (inclusive)
        Maximum value of 0.25 occurs at x = 0
        Approaches 0 as |x| increases
    
    VARIABLE DECLARATIONS:
        DECLARE sig_val as double precision floating-point
        DECLARE one_minus_sig as double precision floating-point
        DECLARE derivative_result as double precision floating-point
    
    STEP 1: Calculate the sigmoid value
        PURPOSE: Get σ(x) which is needed for the derivative formula
        
        CALL sigmoid(input_value)
        STORE returned value in sig_val
        
        RATIONALE:
            The derivative formula requires knowing the sigmoid value itself
            This is computationally efficient because σ'(x) = σ(x)·(1-σ(x))
            We avoid having to recalculate exponentials
            
        PERFORMANCE NOTE:
            If sigmoid(x) has already been computed elsewhere,
            consider passing it as a parameter to avoid redundant calculation
            
        NUMERICAL RANGE:
            sig_val will always be in range (0, 1)
            Never exactly 0 or 1 due to floating-point arithmetic
    
    STEP 2: Calculate the complement (1 - sigmoid)
        PURPOSE: Get the second factor in the derivative formula
        
        CALCULATE one_minus_sig = 1.0 - sig_val
        
        MATHEMATICAL INSIGHT:
            When sig_val is close to 1 (large positive x):
                one_minus_sig is close to 0
                Product will be close to 0
            When sig_val is close to 0 (large negative x):
                one_minus_sig is close to 1
                Product will still be close to 0
            When sig_val = 0.5 (x = 0):
                one_minus_sig = 0.5
                Product will be 0.25 (maximum derivative)
                
        NUMERICAL CONSIDERATION:
            Use 1.0 (not 1) to ensure floating-point subtraction
            No risk of negative values since 0 < sig_val < 1
    
    STEP 3: Multiply sigmoid by its complement
        PURPOSE: Complete the derivative calculation
        
        CALCULATE derivative_result = sig_val * one_minus_sig
        
        MATHEMATICAL PROPERTIES:
            This product has a parabolic shape when plotted
            Symmetric around x = 0
            Maximum at x = 0 where derivative = 0.25
            
        WHY THIS FORMULA WORKS:
            Starting from σ(x) = 1/(1 + e^(-x))
            Apply quotient rule: d/dx[1/f(x)] = -f'(x)/f(x)²
            After algebraic manipulation:
            σ'(x) = σ(x) · (1 - σ(x))
            
        COMPUTATIONAL ADVANTAGE:
            Only requires one multiplication after sigmoid is computed
            Much faster than recalculating from scratch with exponentials
            Crucial for backpropagation where this is computed millions of times
    
    STEP 4: Return the derivative value
        RETURN derivative_result
        
    ALTERNATIVE COMPUTATION METHODS:
        
        METHOD A: Direct from input (less efficient)
            CALCULATE exp_neg_x = exp(-input_value)
            CALCULATE denom = 1.0 + exp_neg_x
            CALCULATE derivative = exp_neg_x / (denom * denom)
            RETURN derivative
            NOTE: This requires exponential calculation, slower
        
        METHOD B: From pre-computed sigmoid (most efficient)
            IF sigmoid value is already available:
                PASS sigmoid_value as additional parameter
                SKIP Step 1, use passed value directly
                Saves one function call
    
    GRADIENT PROPERTIES FOR NEURAL NETWORKS:
        
        VANISHING GRADIENT PROBLEM:
            When |x| is large (e.g., > 5):
                sigmoid(x) approaches 0 or 1
                derivative approaches 0
                Gradients become very small ("vanish")
                Learning slows down or stops
                
            This is why sigmoid is less popular in deep networks
            Modern networks often use ReLU or other activations
            
        OPTIMAL LEARNING RANGE:
            Derivative is largest when |x| < 2
            Keep activations in this range for effective learning
            Use proper weight initialization (e.g., Xavier/He)
            
        BACKPROPAGATION USAGE:
            In neural networks, this derivative is multiplied by
            the error signal from the next layer
            Chain rule: ∂L/∂x = ∂L/∂σ · σ'(x)
    
    NUMERICAL STABILITY CONSIDERATIONS:
        
        FOR VERY LARGE POSITIVE x (e.g., x > 20):
            sig_val ≈ 1.0
            one_minus_sig ≈ 0.0
            derivative ≈ 0.0
            Numerically stable, no issues
            
        FOR VERY LARGE NEGATIVE x (e.g., x < -20):
            sig_val ≈ 0.0
            one_minus_sig ≈ 1.0
            derivative ≈ 0.0
            Numerically stable, no issues
            
        ENHANCED VERSION WITH SHORTCUTS:
            IF input_value > 20.0 OR input_value < -20.0
                RETURN 0.0
                COMMENT: Derivative is effectively zero
            END IF
    
    TESTING VALUES:
        x = 0:     derivative = 0.25 (maximum)
        x = 1:     derivative ≈ 0.1966
        x = -1:    derivative ≈ 0.1966 (symmetric)
        x = 2:     derivative ≈ 0.1050
        x = -2:    derivative ≈ 0.1050
        x = 5:     derivative ≈ 0.0066
        x = -5:    derivative ≈ 0.0066
        x = 10:    derivative ≈ 0.000045
        
    VERIFICATION OF CORRECTNESS:
        1. Symmetry: σ'(x) = σ'(-x) for all x
        2. Non-negativity: σ'(x) ≥ 0 for all x
        3. Maximum: max{σ'(x)} = 0.25 at x = 0
        4. Bounds: 0 ≤ σ'(x) ≤ 0.25
        5. Monotonicity: 
           - Increasing on (-∞, 0)
           - Decreasing on (0, ∞)
    
    MEMORY CONSIDERATIONS:
        Stack space: ~24 bytes (3 doubles)
        No heap allocation
        No static/global variables
        Thread-safe
        
    PERFORMANCE CHARACTERISTICS:
        Time complexity: O(1) if sigmoid value is pre-computed
                        O(exp) if sigmoid must be calculated
        One sigmoid call + two arithmetic operations
        Typical execution: ~25-60 CPU cycles
        
    COMMON PITFALLS TO AVOID:
        1. Don't confuse with σ'(σ(x)) - wrong composition!
        2. Ensure sigmoid is called with correct input value
        3. Don't assume derivative is always positive and large
        4. Remember derivative can be very close to 0
        
END FUNCTION
```

## Expanded Tanh Activation Function
```
FUNCTION tanh_activation(input_value)
    PURPOSE: Transform any real number to a value between -1 and 1
    MATHEMATICAL FORMULA: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    ALTERNATIVE FORM: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    
    INPUT PARAMETERS:
        input_value: A floating-point number (double precision recommended)
                    Can be any real number from -∞ to +∞
    
    RETURN VALUE:
        A floating-point number strictly between -1 and 1 (exclusive)
        Centered at 0, unlike sigmoid which is centered at 0.5
        
    VARIABLE DECLARATIONS (METHOD 1):
        DECLARE exp_positive as double precision floating-point
        DECLARE exp_negative as double precision floating-point
        DECLARE numerator as double precision floating-point
        DECLARE denominator as double precision floating-point
        DECLARE tanh_result as double precision floating-point
        DECLARE negated_input as double precision floating-point
        
    ═══════════════════════════════════════════════════════════
    METHOD 1: FROM SCRATCH (Manual Implementation)
    ═══════════════════════════════════════════════════════════
    
    STEP 1: Calculate positive exponential
        PURPOSE: Compute e^x
        
        CALL exp() function from <math.h>
        PASS input_value as argument
        STORE returned value in exp_positive
        
        RATIONALE:
            We need e^x for the numerator (e^x - e^(-x))
            Also needed for denominator (e^x + e^(-x))
            
        NUMERICAL RANGE:
            For input_value = 0: exp_positive = 1.0
            For input_value > 0: exp_positive grows exponentially
            For input_value < 0: exp_positive shrinks toward 0
            
        OVERFLOW RISK:
            Large positive input_value (> 20) may cause issues
            exp(20) ≈ 4.85 × 10^8 (manageable)
            exp(100) ≈ 2.69 × 10^43 (potential overflow)
            exp(700) will overflow on most systems
    
    STEP 2: Calculate negative exponential
        PURPOSE: Compute e^(-x)
        
        SUBSTEP 2a: Negate the input
            CALCULATE negated_input = -1.0 * input_value
            ALTERNATIVE: negated_input = 0.0 - input_value
            
        SUBSTEP 2b: Calculate exponential
            CALL exp() function
            PASS negated_input as argument
            STORE returned value in exp_negative
            
        MATHEMATICAL RELATIONSHIP:
            exp_negative = 1 / exp_positive (when no overflow)
            This relationship can be used for optimization
            
        NUMERICAL CONSIDERATION:
            For very large positive input_value:
                exp_negative will be extremely small (→ 0)
            For very large negative input_value:
                exp_negative will be extremely large (overflow risk)
    
    STEP 3: Calculate numerator
        PURPOSE: Compute e^x - e^(-x) (hyperbolic sine scaled by 2)
        
        CALCULATE numerator = exp_positive - exp_negative
        
        MATHEMATICAL PROPERTIES:
            When x = 0:
                numerator = 1 - 1 = 0
            When x > 0:
                numerator > 0 (exp_positive dominates)
            When x < 0:
                numerator < 0 (exp_negative dominates)
                
        SIGN INTERPRETATION:
            The sign of numerator determines sign of final result
            This makes tanh an odd function: tanh(-x) = -tanh(x)
            
        CANCELLATION ERROR:
            When |x| is very small (near 0):
                exp_positive ≈ exp_negative ≈ 1
                Subtracting similar values loses precision
                Not a major issue in practice for tanh
    
    STEP 4: Calculate denominator
        PURPOSE: Compute e^x + e^(-x) (hyperbolic cosine scaled by 2)
        
        CALCULATE denominator = exp_positive + exp_negative
        
        MATHEMATICAL PROPERTIES:
            Always positive for all x
            Minimum value is 2.0 at x = 0
            Symmetric around x = 0
            
        RANGE:
            For x = 0: denominator = 2.0
            For |x| large: denominator ≈ max(exp_positive, exp_negative)
            
        NUMERICAL STABILITY:
            Addition is more stable than subtraction
            No cancellation errors
            Denominator never zero, so division is safe
    
    STEP 5: Calculate final result
        PURPOSE: Complete the tanh calculation
        
        CALCULATE tanh_result = numerator / denominator
        
        ALGEBRAIC SIMPLIFICATION:
            tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
            Can also write as: (e^(2x) - 1) / (e^(2x) + 1)
            Or: 2·σ(2x) - 1 (relation to sigmoid)
            
        RANGE VERIFICATION:
            Since |numerator| < denominator for all x
            Result is always in range (-1, 1)
            
        BEHAVIOR:
            When x → +∞: tanh(x) → 1
            When x → -∞: tanh(x) → -1
            When x = 0: tanh(x) = 0
    
    STEP 6: Return result
        RETURN tanh_result
        
    ═══════════════════════════════════════════════════════════
    METHOD 2: USING LIBRARY (Recommended for Production)
    ═══════════════════════════════════════════════════════════
    
    STEP 1: Call built-in hyperbolic tangent
        PURPOSE: Use optimized library implementation
        
        CALL tanh() function from <math.h>
        PASS input_value as argument
        RETURN the result directly
        
        ADVANTAGES OF LIBRARY METHOD:
            - Highly optimized (often uses CPU intrinsics)
            - Handles edge cases automatically
            - Numerically stable for all inputs
            - May use lookup tables or polynomial approximations
            - Tested extensively across platforms
            - Shorter code, less chance for bugs
            
        WHEN TO USE LIBRARY:
            - Production code (always prefer this)
            - When performance matters
            - When you trust the standard library
            
        WHEN TO USE MANUAL IMPLEMENTATION:
            - Educational purposes
            - Embedded systems without math library
            - When you need custom behavior
            - Verifying library correctness
    
    ═══════════════════════════════════════════════════════════
    NUMERICAL STABILITY ENHANCEMENT
    ═══════════════════════════════════════════════════════════
    
    IMPROVED METHOD (Overflow-Safe):
        
        IF input_value >= 0.0
            COMMENT: For non-negative inputs, use standard formula
            CALCULATE exp_pos = exp(input_value)
            CALCULATE exp_neg = exp(-input_value)
            CALCULATE result = (exp_pos - exp_neg) / (exp_pos + exp_neg)
            RETURN result
        ELSE
            COMMENT: For negative inputs, factor out e^x to avoid overflow
            CALCULATE exp_2x = exp(2.0 * input_value)
            CALCULATE result = (exp_2x - 1.0) / (exp_2x + 1.0)
            RETURN result
        END IF
        
        WHY THIS WORKS:
            For x < 0, e^x is small and e^(-x) is large
            Standard formula: (small - large) / (small + large)
            Can lose precision or overflow
            
            Alternative: multiply top and bottom by e^x
            Gets: (e^(2x) - 1) / (e^(2x) + 1)
            Now e^(2x) is small for negative x (safe)
    
    AGGRESSIVE CLAMPING (Optional):
        
        IF input_value > 20.0
            RETURN 1.0
            COMMENT: tanh(20) ≈ 0.9999999999 (effectively 1)
        END IF
        
        IF input_value < -20.0
            RETURN -1.0
            COMMENT: tanh(-20) ≈ -0.9999999999 (effectively -1)
        END IF
        
        PROCEED with normal calculation
    
    ═══════════════════════════════════════════════════════════
    EDGE CASES AND SPECIAL VALUES
    ═══════════════════════════════════════════════════════════
    
    INPUT: x = 0
        EXPECTED: tanh(0) = 0 exactly
        REASON: Numerator = 1 - 1 = 0
        
    INPUT: x = 1
        EXPECTED: tanh(1) ≈ 0.7615941559557649
        CALCULATION: (e - 1/e) / (e + 1/e)
        
    INPUT: x = -1
        EXPECTED: tanh(-1) ≈ -0.7615941559557649
        PROPERTY: tanh(-x) = -tanh(x) (odd function)
        
    INPUT: x → +∞
        EXPECTED: tanh(x) → 1.0
        REASON: e^(-x) → 0, so (e^x - 0)/(e^x + 0) = 1
        
    INPUT: x → -∞
        EXPECTED: tanh(x) → -1.0
        REASON: e^x → 0, so (0 - e^(-x))/(0 + e^(-x)) = -1
        
    INPUT: x = NaN
        EXPECTED: Return NaN
        HANDLING: Check with isnan() if necessary
        
    INPUT: x = +Infinity
        EXPECTED: Return 1.0 (or NaN depending on implementation)
        
    INPUT: x = -Infinity
        EXPECTED: Return -1.0 (or NaN depending on implementation)
    
    ═══════════════════════════════════════════════════════════
    MATHEMATICAL PROPERTIES
    ═══════════════════════════════════════════════════════════
    
    SYMMETRY PROPERTIES:
        1. Odd function: tanh(-x) = -tanh(x)
        2. Point symmetry around origin (0,0)
        3. tanh(0) = 0 (passes through origin)
        
    RELATIONSHIPS TO OTHER FUNCTIONS:
        1. tanh(x) = 2·sigmoid(2x) - 1
        2. tanh(x) = (1 - e^(-2x)) / (1 + e^(-2x))
        3. sinh(x) = (numerator) / 2
        4. cosh(x) = (denominator) / 2
        5. tanh(x) = sinh(x) / cosh(x)
        
    CALCULUS PROPERTIES:
        1. Domain: (-∞, +∞)
        2. Range: (-1, +1)
        3. Continuous everywhere
        4. Differentiable everywhere
        5. Monotonically increasing
        6. Horizontal asymptotes at y = ±1
        7. Inflection point at (0, 0)
        
    SERIES EXPANSION (for small x):
        tanh(x) = x - x³/3 + 2x⁵/15 - 17x⁷/315 + ...
        Useful for |x| < 1
        Can be used for polynomial approximation
    
    ═══════════════════════════════════════════════════════════
    ADVANTAGES OVER SIGMOID FOR NEURAL NETWORKS
    ═══════════════════════════════════════════════════════════
    
    ZERO-CENTERED:
        - Output ranges from -1 to 1 (not 0 to 1)
        - Mean activation is 0, not 0.5
        - Helps with gradient flow in deep networks
        - Reduces bias in weight updates
        
    STEEPER GRADIENT:
        - Maximum derivative is 1.0 (vs 0.25 for sigmoid)
        - Stronger gradients = faster learning
        - Less prone to vanishing gradient (but still occurs)
        
    SYMMETRIC SATURATION:
        - Saturates at both ends symmetrically
        - Better for handling negative inputs
        
    STILL HAS LIMITATIONS:
        - Still suffers from vanishing gradients for |x| > 3
        - Exponential calculations are slow
        - Modern networks often prefer ReLU family
    
    ═══════════════════════════════════════════════════════════
    TESTING VALUES FOR VERIFICATION
    ═══════════════════════════════════════════════════════════
    
    x = 0.0:    tanh(x) = 0.0 exactly
    x = 0.5:    tanh(x) ≈ 0.46211715726
    x = 1.0:    tanh(x) ≈ 0.76159415595
    x = 2.0:    tanh(x) ≈ 0.96402758007
    x = 5.0:    tanh(x) ≈ 0.99990920426
    x = 10.0:   tanh(x) ≈ 0.99999999587
    x = -1.0:   tanh(x) ≈ -0.76159415595
    x = -2.0:   tanh(x) ≈ -0.96402758007
    
    ═══════════════════════════════════════════════════════════
    PERFORMANCE CONSIDERATIONS
    ═══════════════════════════════════════════════════════════
    
    COMPUTATIONAL COST:
        Method 1: Two exp() calls + 3 arithmetic ops
        Method 2: One tanh() call (typically optimized)
        
        Library version is usually 2-3x faster
        
    MEMORY USAGE:
        Stack: ~40 bytes for Method 1 (5 doubles)
        Stack: ~16 bytes for Method 2 (2 doubles)
        No heap allocation
        
    OPTIMIZATION STRATEGIES:
        1. Use library function when available
        2. For batch processing, use SIMD instructions
        3. Consider lookup table for fixed-point arithmetic
        4. Pre-compute for repeated same values
        5. Use approximation polynomials for speed-critical code
        
    APPROXIMATION ALTERNATIVES:
        Fast tanh (less accurate but faster):
            IF |x| > 2.5
                RETURN sign(x) * 1.0
            ELSE
                RETURN x / (1 + |x|)
            END IF
        
        Rational approximation:
            tanh(x) ≈ x(27 + x²) / (27 + 9x²)
            Valid for |x| < 1.6, error < 0.003
        
END FUNCTION

FUNCTION tanh_derivative(input_value)
    PURPOSE: Calculate the rate of change (gradient) of tanh at a specific point
    MATHEMATICAL FORMULA: tanh'(x) = 1 - tanh²(x) = sech²(x)
    ALTERNATIVE FORM: tanh'(x) = 4 / (e^x + e^(-x))²
    
    INPUT PARAMETERS:
        input_value: A floating-point number (double precision recommended)
                    Can be any real number from -∞ to +∞
    
    RETURN VALUE:
        A floating-point number between 0 and 1 (inclusive)
        Maximum value of 1.0 occurs at x = 0
        Approaches 0 as |x| increases
        Always positive (strictly positive for finite x)
    
    VARIABLE DECLARATIONS:
        DECLARE tanh_val as double precision floating-point
        DECLARE tanh_squared as double precision floating-point
        DECLARE derivative_result as double precision floating-point
    
    ═══════════════════════════════════════════════════════════
    STEP-BY-STEP COMPUTATION
    ═══════════════════════════════════════════════════════════
    
    STEP 1: Calculate tanh value
        PURPOSE: Get tanh(x) which is needed for the derivative formula
        
        CALL tanh_activation(input_value)
        STORE returned value in tanh_val
        
        RATIONALE:
            The derivative formula uses tanh²(x)
            Much more efficient than computing from exponentials
            Formula: tanh'(x) = 1 - tanh²(x)
            
        ALTERNATIVE APPROACH:
            If tanh(x) is already computed in forward pass:
                Consider passing it as parameter
                Saves redundant function call
                Common in neural network backpropagation
                
        NUMERICAL RANGE:
            tanh_val will be in range (-1, 1)
            Can be positive, negative, or zero
            
        MEMORY CONSIDERATION:
            If doing both forward and backward pass:
                Cache tanh_val from forward pass
                Reuse in backward pass
                Significant performance improvement
    
    STEP 2: Square the tanh value
        PURPOSE: Compute tanh²(x) for the derivative formula
        
        CALCULATE tanh_squared = tanh_val * tanh_val
        
        ALTERNATIVE NOTATIONS:
            tanh_squared = tanh_val² 
            tanh_squared = pow(tanh_val, 2.0)  // Less efficient
            
        WHY MULTIPLICATION INSTEAD OF POW:
            Multiplication is much faster than pow()
            pow() has function call overhead
            pow() handles general exponents (overkill for squaring)
            
        NUMERICAL PROPERTIES:
            tanh_squared is always non-negative
            Range: [0, 1) (can be 0, approaches 1 but never reaches it)
            
            When tanh_val = 0 (at x = 0):
                tanh_squared = 0
                
            When tanh_val ≈ ±1 (large |x|):
                tanh_squared ≈ 1
                
            When tanh_val = ±0.5:
                tanh_squared = 0.25
                
        PRECISION CONSIDERATION:
            Squaring can amplify or reduce floating-point errors
            Small values become very small
            Values near 1 stay near 1
            Generally stable operation
    
    STEP 3: Subtract from 1
        PURPOSE: Complete the derivative calculation
        
        CALCULATE derivative_result = 1.0 - tanh_squared
        
        MATHEMATICAL INSIGHT:
            This formula comes from the chain rule:
            d/dx[tanh(x)] = d/dx[(e^x - e^(-x))/(e^x + e^(-x))]
            Using quotient rule and simplification:
            = sech²(x) = 1/cosh²(x) = 1 - tanh²(x)
            
        WHY 1.0 NOT 1:
            Using 1.0 ensures floating-point arithmetic
            Compiler treats it as double
            Prevents integer arithmetic confusion
            
        NUMERICAL BEHAVIOR:
            When tanh_squared ≈ 0 (x near 0):
                derivative_result ≈ 1.0 (maximum)
                
            When tanh_squared ≈ 1 (large |x|):
                derivative_result ≈ 0.0 (minimum)
                
            When tanh_squared = 0.25:
                derivative_result = 0.75
                
        CANCELLATION CONCERN:
            When tanh_squared is very close to 1:
                Subtracting similar values loses precision
                Result may have few significant digits
                Not usually a problem in practice
                If tanh_squared > 0.9999, derivative < 0.0001
    
    STEP 4: Return the derivative
        RETURN derivative_result
        
    ═══════════════════════════════════════════════════════════
    WHY THIS FORMULA WORKS (MATHEMATICAL DERIVATION)
    ═══════════════════════════════════════════════════════════
    
    STARTING POINT:
        tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        
    APPLY QUOTIENT RULE:
        If f(x) = g(x)/h(x), then f'(x) = [g'h - gh']/h²
        
        Let g(x) = e^x - e^(-x)
        Let h(x) = e^x + e^(-x)
        
    COMPUTE DERIVATIVES:
        g'(x) = e^x + e^(-x)  // Note: derivative of -e^(-x) is +e^(-x)
        h'(x) = e^x - e^(-x)
        
    APPLY FORMULA:
        tanh'(x) = [(e^x + e^(-x))(e^x + e^(-x)) - (e^x - e^(-x))(e^x - e^(-x))]
                   / (e^x + e^(-x))²
        
    SIMPLIFY NUMERATOR:
        = [(e^x + e^(-x))² - (e^x - e^(-x))²] / (e^x + e^(-x))²
        = [e^(2x) + 2 + e^(-2x) - e^(2x) + 2 - e^(-2x)] / (e^x + e^(-x))²
        = 4 / (e^x + e^(-x))²
        
    ALTERNATIVE FORM:
        = 1 / cosh²(x) = sech²(x)
        
    RELATE TO TANH:
        Since cosh²(x) - sinh²(x) = 1
        And tanh(x) = sinh(x)/cosh(x)
        Therefore: sech²(x) = 1 - tanh²(x)
        
    CONCLUSION:
        tanh'(x) = 1 - tanh²(x) ✓
    
    ═══════════════════════════════════════════════════════════
    ALTERNATIVE IMPLEMENTATION METHODS
    ═══════════════════════════════════════════════════════════
    
    METHOD A: From exponentials (less efficient)
        CALCULATE exp_x = exp(input_value)
        CALCULATE exp_neg_x = exp(-input_value)
        CALCULATE cosh_x = (exp_x + exp_neg_x) / 2.0
        CALCULATE sech_x = 1.0 / cosh_x
        CALCULATE derivative = sech_x * sech_x
        RETURN derivative
        
        COST: 2 exp() calls + multiple arithmetic operations
        
    METHOD B: From pre-computed tanh (most efficient)
        IF tanh_value is already known:
            ACCEPT tanh_value as parameter instead of input_value
            SKIP Step 1
            CALCULATE tanh_squared = tanh_value * tanh_value
            CALCULATE derivative = 1.0 - tanh_squared
            RETURN derivative
            
        COST: 2 multiplications + 1 subtraction (very fast)
        
    METHOD C: Using identity (for verification)
        CALCULATE tanh_val = tanh_activation(input_value)
        CALCULATE sinh_val = (exp(input_value) - exp(-input_value)) / 2.0
        CALCULATE cosh_val = (exp(input_value) + exp(-input_value)) / 2.0
        VERIFY: tanh_val = sinh_val / cosh_val
        CALCULATE derivative = 1.0 / (cosh_val * cosh_val)
        RETURN derivative
        
    ═══════════════════════════════════════════════════════════
    PROPERTIES OF THE DERIVATIVE
    ═══════════════════════════════════════════════════════════
    
    RANGE AND BOUNDS:
        - Minimum value: 0 (at x = ±∞)
        - Maximum value: 1 (at x = 0)
        - Range: (0, 1] (exclusive of 0, inclusive of 1)
        - Always positive for finite x
        
SYMMETRY (continued):
        - Symmetric around x = 0
        - tanh'(2) = tanh'(-2)
        - Mirror image across y-axis when plotted
        
    MONOTONICITY:
        - Strictly decreasing on (0, ∞)
        - Strictly increasing on (-∞, 0)
        - Maximum at x = 0
        
    RATE OF DECAY:
        - Decays exponentially as |x| increases
        - Much faster decay than sigmoid derivative
        - By x = ±3, derivative < 0.01
        - By x = ±5, derivative < 0.0001
        
    COMPARISON WITH SIGMOID DERIVATIVE:
        Sigmoid maximum: 0.25 at x = 0
        Tanh maximum: 1.0 at x = 0
        
        Tanh derivative is 4x larger at origin
        Better gradient flow in neural networks
        Still suffers from vanishing gradient problem
    
    ═══════════════════════════════════════════════════════════
    NEURAL NETWORK IMPLICATIONS
    ═══════════════════════════════════════════════════════════
    
    VANISHING GRADIENT PROBLEM:
        
        CRITICAL RANGE FOR LEARNING:
            |x| < 2: derivative > 0.07 (decent learning)
            |x| > 3: derivative < 0.01 (very slow learning)
            |x| > 5: derivative < 0.0001 (effectively no learning)
            
        IMPACT ON DEEP NETWORKS:
            As gradients backpropagate through layers:
                Each layer multiplies by tanh'(x)
                If tanh'(x) < 1, gradient shrinks
                Deep networks: gradient ≈ (tanh'(x))^n where n = layers
                With n=10 and tanh'(x)=0.5: gradient ≈ 0.001
                
        MITIGATION STRATEGIES:
            1. Careful weight initialization (Xavier/Glorot)
            2. Batch normalization
            3. Residual connections (skip connections)
            4. Switch to ReLU or variants
            5. Gradient clipping
            
    BACKPROPAGATION USAGE:
        
        FORWARD PASS:
            Store activation: a = tanh(z)
            where z is weighted sum
            
        BACKWARD PASS:
            Receive gradient from next layer: dL/da
            Compute local gradient: da/dz = tanh'(z)
            Chain rule: dL/dz = (dL/da) × (da/dz)
            
        EFFICIENCY TRICK:
            Since we have 'a' from forward pass:
            da/dz = 1 - a²
            No need to recompute tanh(z)
            Just square the stored activation
            
        VECTORIZED COMPUTATION:
            For batch of examples:
                A = tanh(Z)  // Forward pass, store A
                dZ = dA * (1 - A²)  // Backward pass, element-wise
                
    ADVANTAGES OVER SIGMOID:
        1. Larger maximum gradient (1.0 vs 0.25)
        2. Zero-centered (helps optimization)
        3. Steeper around origin (faster initial learning)
        
    DISADVANTAGES COMPARED TO ReLU:
        1. Exponential calculations (slower)
        2. Still has vanishing gradient
        3. Not sparse (always non-zero)
        4. Saturates on both sides
    
    ═══════════════════════════════════════════════════════════
    NUMERICAL STABILITY CONSIDERATIONS
    ═══════════════════════════════════════════════════════════
    
    POTENTIAL ISSUES:
        
        ISSUE 1: tanh_val very close to ±1
            When |input_value| > 5:
                tanh_val ≈ ±0.99999...
                tanh_squared ≈ 0.99999...
                1 - tanh_squared ≈ 0.00001...
                
            Problem: Catastrophic cancellation
            Impact: Loss of precision in result
            
            Solution (if needed):
                IF tanh_val > 0.99999 OR tanh_val < -0.99999
                    RETURN 0.0
                END IF
                
        ISSUE 2: Very large |input_value|
            Computing tanh becomes unnecessary
            Result will be ≈ 0 anyway
            
            Optimization:
                IF input_value > 10.0 OR input_value < -10.0
                    RETURN 0.0
                    COMMENT: Derivative effectively zero
                END IF
                
        ISSUE 3: NaN propagation
            If input_value is NaN:
                tanh_val will be NaN
                tanh_squared will be NaN
                derivative_result will be NaN
                
            Handling (if needed):
                IF isnan(input_value)
                    RETURN NaN
                    OR handle error appropriately
                END IF
                
    FLOATING-POINT PRECISION:
        
        SINGLE PRECISION (float):
            Precision: ~7 decimal digits
            Issues when |x| > 4
            Acceptable for inference
            
        DOUBLE PRECISION (double):
            Precision: ~15 decimal digits
            Accurate for |x| < 10
            Recommended for training
            
        EXTENDED PRECISION (long double):
            Rarely needed for neural networks
            Overkill for most applications
    
    ═══════════════════════════════════════════════════════════
    TESTING AND VERIFICATION
    ═══════════════════════════════════════════════════════════
    
    EXACT TEST VALUES:
        
        x = 0.0:
            tanh(0) = 0
            tanh²(0) = 0
            derivative = 1.0 (maximum)
            
        x = 1.0:
            tanh(1) ≈ 0.7616
            tanh²(1) ≈ 0.5800
            derivative ≈ 0.4200
            
        x = -1.0:
            tanh(-1) ≈ -0.7616
            tanh²(-1) ≈ 0.5800 (same as x=1, even function)
            derivative ≈ 0.4200 (same as x=1)
            
        x = 2.0:
            tanh(2) ≈ 0.9640
            tanh²(2) ≈ 0.9293
            derivative ≈ 0.0707
            
        x = 3.0:
            tanh(3) ≈ 0.9951
            tanh²(3) ≈ 0.9902
            derivative ≈ 0.0098
            
        x = 5.0:
            tanh(5) ≈ 0.9999
            tanh²(5) ≈ 0.9998
            derivative ≈ 0.0002
            
    PROPERTY VERIFICATION TESTS:
        
        TEST 1: Even function property
            FOR various x:
                COMPUTE deriv_pos = tanh_derivative(x)
                COMPUTE deriv_neg = tanh_derivative(-x)
                VERIFY: deriv_pos = deriv_neg (within tolerance)
            END FOR
            
        TEST 2: Maximum at origin
            COMPUTE deriv_0 = tanh_derivative(0.0)
            VERIFY: deriv_0 = 1.0
            
            FOR x from -5.0 to 5.0 in small steps:
                COMPUTE deriv_x = tanh_derivative(x)
                VERIFY: deriv_x ≤ deriv_0
            END FOR
            
        TEST 3: Positive values only
            FOR x from -10.0 to 10.0:
                COMPUTE deriv = tanh_derivative(x)
                VERIFY: deriv > 0
            END FOR
            
        TEST 4: Monotonicity
            FOR x from 0.0 to 5.0 in small steps:
                COMPUTE deriv_current = tanh_derivative(x)
                COMPUTE deriv_next = tanh_derivative(x + 0.01)
                VERIFY: deriv_next < deriv_current (decreasing)
            END FOR
            
        TEST 5: Numerical derivative comparison
            Use finite difference method:
                h = 0.00001
                tanh_plus = tanh_activation(x + h)
                tanh_minus = tanh_activation(x - h)
                numerical_deriv = (tanh_plus - tanh_minus) / (2*h)
                analytical_deriv = tanh_derivative(x)
                VERIFY: |numerical_deriv - analytical_deriv| < 0.0001
    
    ═══════════════════════════════════════════════════════════
    OPTIMIZATION TECHNIQUES
    ═══════════════════════════════════════════════════════════
    
    CACHING STRATEGY (Critical for Neural Networks):
        
        PROBLEM:
            Both tanh(x) and tanh'(x) are needed
            Computing both separately is wasteful
            
        SOLUTION:
            DURING FORWARD PASS:
                Compute and store: a = tanh(z)
                Store in activation cache
                
            DURING BACKWARD PASS:
                Retrieve: a from cache
                Compute: da/dz = 1 - a²
                No tanh function call needed!
                
        MEMORY COST:
            One float/double per activation
            Typical: few MB for moderate networks
            
        SPEED BENEFIT:
            Eliminates ~50% of tanh computations
            Replaces exp() calls with simple arithmetic
            Can be 5-10x faster for backward pass
            
    VECTORIZATION (For Batch Processing):
        
        USING SIMD INSTRUCTIONS:
            Modern CPUs have vector units
            Can process 4-8 values simultaneously
            Especially effective for tanh derivative
            
            PSEUDO-VECTOR CODE:
                LOAD 4 tanh_values into vector register
                MULTIPLY vector by itself (squared)
                BROADCAST 1.0 to vector
                SUBTRACT squared from ones
                STORE result vector
                
        USING GPU (CUDA/OpenCL):
            Each thread computes one derivative
            Thousands of threads in parallel
            Essential for deep learning frameworks
            
    LOOKUP TABLE APPROXIMATION:
        
        FOR EMBEDDED SYSTEMS OR SPEED-CRITICAL CODE:
            
            STEP 1: Pre-compute table
                ALLOCATE table[1000]
                FOR i from 0 to 999:
                    x = (i - 500) * 0.01  // Range: -5 to 5
                    table[i] = 1.0 - tanh(x)²
                END FOR
                
            STEP 2: During runtime
                IF |input_value| > 5.0
                    RETURN 0.0
                END IF
                
                index = round((input_value + 5.0) * 100)
                CLAMP index to [0, 999]
                RETURN table[index]
                
            TRADE-OFFS:
                + Very fast (just array lookup)
                + Predictable timing
                - Less accurate (discretization error)
                - Uses memory for table
                - Limited input range
    
    POLYNOMIAL APPROXIMATION:
        
        FOR MODERATE ACCURACY NEEDS:
            
            Valid for |x| < 2:
                derivative ≈ 1 - x²/3 + 2x⁴/15 - 17x⁶/315
                
            Valid for |x| < 1:
                derivative ≈ 1 - x²
                (First two terms of Taylor series)
                
            Fast but less accurate
            Good for embedded systems
    
    ═══════════════════════════════════════════════════════════
    MEMORY CONSIDERATIONS
    ═══════════════════════════════════════════════════════════
    
    STACK USAGE:
        Basic implementation: 3 doubles = 24 bytes
        With additional temp variables: ~40 bytes
        Negligible for modern systems
        
    CACHE BEHAVIOR:
        Function is small (fits in instruction cache)
        No data cache issues (no arrays)
        Branch predictor friendly (no conditionals in basic version)
        
    MEMORY ACCESS PATTERN:
        Sequential when processing arrays
        Cache-friendly for vectorized operations
        No random access
        
    FOR NEURAL NETWORK LAYER:
        Forward pass: Store n activations (n × 8 bytes)
        Backward pass: Compute from stored activations
        Total: n doubles stored temporarily
        
        Example: Layer with 1000 neurons
            Cache size: 8,000 bytes = 8 KB
            Easily fits in L1 cache (typically 32-64 KB)
    
    ═══════════════════════════════════════════════════════════
    PERFORMANCE CHARACTERISTICS
    ═══════════════════════════════════════════════════════════
    
    TIME COMPLEXITY:
        O(1) - constant time per evaluation
        Independent of input magnitude
        
    TYPICAL EXECUTION TIME:
        With tanh call: 30-60 CPU cycles
        From cached value: 5-10 CPU cycles
        
    COMPARED TO OTHER OPERATIONS:
        Faster than: exp, log, trig functions
        Slower than: ReLU derivative (just comparison)
        Similar to: sigmoid derivative
        
    BOTTLENECKS:
        1. tanh() function call (if not cached)
        2. Function call overhead itself
        3. Memory access for loading variables
        
    SCALABILITY:
        Parallelizes perfectly
        No dependencies between different inputs
        Ideal for GPU acceleration
        SIMD-friendly
    
    ═══════════════════════════════════════════════════════════
    COMMON IMPLEMENTATION MISTAKES
    ═══════════════════════════════════════════════════════════
    
    MISTAKE 1: Using tanh'(tanh(x)) instead of tanh'(x)
        WRONG:
            tanh_val = tanh_activation(input_value)
            derivative = tanh_derivative(tanh_val)  // NO!
            
        CORRECT:
            derivative = tanh_derivative(input_value)
            
        WHY IT'S WRONG:
            We want derivative at x, not at tanh(x)
            Composition of functions, not derivative of composition
            
    MISTAKE 2: Forgetting to square
        WRONG:
            derivative = 1.0 - tanh_val  // Missing square
            
        CORRECT:
            derivative = 1.0 - tanh_val * tanh_val
            
    MISTAKE 3: Using integer 1 instead of 1.0
        WRONG:
            derivative = 1 - tanh_squared  // Integer arithmetic
            
        CORRECT:
            derivative = 1.0 - tanh_squared
            
    MISTAKE 4: Inefficient repeated computation
        WRONG:
            FOR each training iteration:
                forward_val = tanh(x)
                backward_deriv = 1 - tanh(x)²  // Recomputes tanh
                
        CORRECT:
            forward_val = tanh(x)
            backward_deriv = 1 - forward_val²  // Reuse
            
    MISTAKE 5: Not handling edge cases
        Missing checks for:
            - Very large |x| (unnecessary computation)
            - NaN inputs
            - Infinity inputs
    
    ═══════════════════════════════════════════════════════════
    DEBUGGING AND TROUBLESHOOTING
    ═══════════════════════════════════════════════════════════
    
    SYMPTOM: Derivative is negative
        CAUSE: Bug in implementation (should never happen)
        CHECK: Is 1.0 coming before tanh_squared in subtraction?
        FIX: Ensure correct order: 1.0 - tanh_squared
        
    SYMPTOM: Derivative greater than 1
        CAUSE: Logic error (mathematically impossible)
        CHECK: Is tanh_squared actually squared?
        FIX: Use tanh_val * tanh_val, not just tanh_val
        
    SYMPTOM: Derivative is NaN
        CAUSE: NaN input propagating through
        CHECK: Is input_value NaN? Is tanh returning NaN?
        FIX: Add input validation
        
    SYMPTOM: Learning is extremely slow
        CAUSE: Activations might be saturated (|x| large)
        CHECK: Print histogram of activation values
        FIX: Adjust weight initialization, use batch normalization
        
    SYMPTOM: Inconsistent with numerical derivative
        CAUSE: Implementation error
        CHECK: Compare against finite difference
        FIX: Review formula: should be 1 - tanh²(x)
        
    VALIDATION CHECKLIST:
        ☐ Derivative at x=0 equals 1.0
        ☐ Derivative is always positive
        ☐ Derivative is always ≤ 1.0
        ☐ Derivative is symmetric: f'(x) = f'(-x)
        ☐ Derivative matches numerical approximation
        ☐ For |x| > 5, derivative ≈ 0
    
    ═══════════════════════════════════════════════════════════
    USAGE EXAMPLES IN CONTEXT
    ═══════════════════════════════════════════════════════════
    
    EXAMPLE 1: Single value computation
        DECLARE x as double = 0.5
        DECLARE grad as double
        
        grad = tanh_derivative(x)
        PRINT "Gradient at x=" + x + " is " + grad
        
    EXAMPLE 2: With pre-computed activation
        DECLARE activation as double = tanh_activation(0.5)
        DECLARE grad as double = 1.0 - (activation * activation)
        
    EXAMPLE 3: Batch gradient computation
        DECLARE inputs as array[100] of double
        DECLARE gradients as array[100] of double
        
        FOR i from 0 to 99:
            gradients[i] = tanh_derivative(inputs[i])
        END FOR
        
    EXAMPLE 4: Neural network backprop
        DECLARE layer_activation as array[neurons] of double
        DECLARE layer_gradient as array[neurons] of double
        DECLARE upstream_grad as array[neurons] of double
        
        FOR i from 0 to neurons - 1:
            local_grad = 1.0 - (layer_activation[i] * layer_activation[i])
            layer_gradient[i] = upstream_grad[i] * local_grad
        END FOR
    
    ═══════════════════════════════════════════════════════════
    RELATED FUNCTIONS AND CONNECTIONS
    ═══════════════════════════════════════════════════════════
    
    RELATIONSHIP TO SECH²:
        tanh'(x) = sech²(x) = 1/cosh²(x)
        Hyperbolic secant squared
        
    RELATIONSHIP TO SIGMOID:
        tanh'(x) = 4 × sigmoid(2x) × sigmoid(-2x)
        Alternative formula connecting the two
        
    SECOND DERIVATIVE:
        tanh''(x) = -2 × tanh(x) × tanh'(x)
        Used in second-order optimization methods
        
    INTEGRAL:
        ∫ tanh'(x) dx = tanh(x) + C
        By definition (fundamental theorem of calculus)
    
END FUNCTION

═══════════════════════════════════════════════════════════
STRUCTURE: Matrix
═══════════════════════════════════════════════════════════

PURPOSE:
    Represent a 2D array of floating-point numbers
    Core data structure for neural network computations
    Enables efficient storage and manipulation of numerical data
    
STRUCTURE DEFINITION:
    
    STRUCTURE Matrix
        FIELD: number_of_rows
            TYPE: Integer (typically int or size_t)
            PURPOSE: Store the number of rows in the matrix
            RANGE: 1 to maximum integer value
            TYPICAL VALUES: 1-10,000 for neural networks
            CONSTRAINTS: Must be positive (> 0)
            
        FIELD: number_of_columns
            TYPE: Integer (typically int or size_t)
            PURPOSE: Store the number of columns in the matrix
            RANGE: 1 to maximum integer value
            TYPICAL VALUES: 1-10,000 for neural networks
            CONSTRAINTS: Must be positive (> 0)
            
        FIELD: data
            TYPE: Pointer to pointer to double (double**)
            PURPOSE: Store the actual matrix elements
            STRUCTURE: Array of row pointers, each pointing to a row array
            MEMORY LAYOUT:
                data[0] → [element_0_0, element_0_1, ..., element_0_(cols-1)]
                data[1] → [element_1_0, element_1_1, ..., element_1_(cols-1)]
                ...
                data[rows-1] → [element_(rows-1)_0, ..., element_(rows-1)_(cols-1)]
                
            ACCESS PATTERN:
                Element at row i, column j: data[i][j]
                This gives natural mathematical notation: M[i,j]
    END STRUCTURE

MEMORY LAYOUT VISUALIZATION:
    
    For a 3×4 matrix:
    
    Matrix structure (on stack or heap):
        ┌─────────────────────┐
        │ number_of_rows: 3   │
        │ number_of_columns: 4│
        │ data: ──────────────┼──→ Row pointers array (heap)
        └─────────────────────┘       ┌────┐
                                      │ ───┼──→ Row 0: [1.0, 2.0, 3.0, 4.0]
                                      │ ───┼──→ Row 1: [5.0, 6.0, 7.0, 8.0]
                                      │ ───┼──→ Row 2: [9.0, 10.0, 11.0, 12.0]
                                      └────┘

DESIGN CONSIDERATIONS:
    
    WHY POINTER-TO-POINTER:
        ADVANTAGES:
            + Natural 2D array syntax: data[i][j]
            + Each row is contiguous in memory
            + Flexible (rows can be different sizes, though we don't use this)
            + Easy to pass rows to functions
            
        DISADVANTAGES:
            - Not fully contiguous (rows are separate allocations)
            - Cache-unfriendly for large matrices
            - More complex memory management
            - Fragmentation risk
            
    ALTERNATIVE DESIGNS:
        
        OPTION A: Single contiguous allocation
            TYPE: double* data
            ACCESS: data[i * number_of_columns + j]
            PROS: Better cache locality, single allocation
            CONS: Less intuitive syntax
            
        OPTION B: Array of pointers to single allocation
            Allocate one block for all elements
            Point each row into that block
            PROS: Contiguous data, natural syntax
            CONS: More setup complexity

USAGE SCENARIOS:
    
    NEURAL NETWORK WEIGHTS:
        Rows = number of neurons in current layer
        Columns = number of neurons in previous layer
        Example: 128×784 for MNIST input layer
        
    ACTIVATION MATRICES:
        Rows = batch size
        Columns = number of features/neurons
        Example: 32×100 for batch of 32, layer of 100 neurons
        
    GRADIENT MATRICES:
        Same dimensions as weight matrices
        Store gradients for backpropagation
        
    INPUT DATA:
        Rows = number of samples
        Columns = number of features
        Example: 1000×50 for 1000 samples, 50 features

TYPICAL SIZES:
    Small: 10×10 (100 elements, ~800 bytes)
    Medium: 100×100 (10,000 elements, ~80 KB)
    Large: 1000×1000 (1,000,000 elements, ~8 MB)
    Very large: 10000×10000 (100M elements, ~800 MB)


═══════════════════════════════════════════════════════════
FUNCTION: create_matrix
═══════════════════════════════════════════════════════════

PURPOSE:
    Allocate and initialize a new matrix structure
    Set all elements to zero by default
    Prepare matrix for subsequent operations
    Ensure proper memory management
    
INPUT PARAMETERS:
    rows: Integer specifying number of rows
          Must be positive (> 0)
          Typical range: 1 to 10,000
          
    cols: Integer specifying number of columns
          Must be positive (> 0)
          Typical range: 1 to 10,000
          
RETURN VALUE:
    Pointer to newly created Matrix structure
    NULL if allocation fails
    Matrix with all elements initialized to 0.0
    
VARIABLE DECLARATIONS:
    DECLARE new_matrix as pointer to Matrix
    DECLARE i as integer (loop counter for rows)
    DECLARE j as integer (loop counter for columns)
    
═══════════════════════════════════════════════════════════

STEP 1: Input Validation (Optional but Recommended)
    PURPOSE: Ensure valid parameters before allocation
    
    IF rows <= 0 OR cols <= 0
        PRINT "Error: Matrix dimensions must be positive"
        PRINT "Requested: " + rows + " rows, " + cols + " columns"
        RETURN NULL
    END IF
    
    IF rows > 100000 OR cols > 100000
        PRINT "Warning: Very large matrix requested"
        PRINT "This may consume significant memory"
        COMMENT: rows × cols × 8 bytes + overhead
    END IF
    
    OVERFLOW CHECK:
        IF rows × cols would overflow integer
            PRINT "Error: Matrix dimensions too large"
            RETURN NULL
        END IF

STEP 2: Allocate memory for matrix structure
    PURPOSE: Create the Matrix structure itself
    
    CALL malloc or equivalent memory allocator
        REQUEST: sizeof(Matrix) bytes
        TYPICAL SIZE: 16-24 bytes (2 ints + 1 pointer)
        
    STORE returned pointer in new_matrix
    
    ERROR HANDLING:
        IF new_matrix is NULL
            PRINT "Error: Failed to allocate matrix structure"
            PRINT "Requested size: " + sizeof(Matrix) + " bytes"
            RETURN NULL
        END IF
        
    WHAT WE HAVE NOW:
        A Matrix structure on the heap
        Fields contain garbage values
        data pointer is uninitialized (dangerous!)

STEP 3: Set dimensions
    PURPOSE: Store matrix size in the structure
    
    SET new_matrix->number_of_rows to rows
    SET new_matrix->number_of_columns to cols
    
    ALTERNATIVE NOTATION:
        (*new_matrix).number_of_rows = rows
        (*new_matrix).number_of_columns = cols
        
    WHY THIS STEP IS IMPORTANT:
        Functions operating on matrix need to know its size
        Prevents array bounds violations
        Enables dynamic operations (don't need to pass size separately)
        
    WHAT WE HAVE NOW:
        Matrix structure with valid dimensions
        data pointer still uninitialized

STEP 4: Allocate memory for row pointers array
    PURPOSE: Create array to hold pointers to each row
    
    CALL malloc or equivalent
        REQUEST: rows × sizeof(double*) bytes
        EXAMPLE: 10 rows × 8 bytes = 80 bytes
        
    STORE returned pointer in new_matrix->data
    
    ERROR HANDLING:
        IF new_matrix->data is NULL
            PRINT "Error: Failed to allocate row pointer array"
            PRINT "Requested: " + rows + " pointers"
            
            CLEANUP:
                FREE new_matrix (the structure itself)
                    COMMENT: Prevent memory leak
                    
            RETURN NULL
        END IF
        
    WHAT WE HAVE NOW:
        Matrix structure with dimensions
        Array of row pointers allocated
        Each row pointer contains garbage (uninitialized)
        Actual row data not yet allocated

STEP 5: Allocate memory for each row
    PURPOSE: Allocate space for actual matrix elements
    
    FOR row_index from 0 to rows - 1
        
        CALL malloc or equivalent
            REQUEST: cols × sizeof(double) bytes
            EXAMPLE: 20 columns × 8 bytes = 160 bytes
            
        STORE returned pointer in new_matrix->data[row_index]
        
        ERROR HANDLING:
            IF new_matrix->data[row_index] is NULL
                PRINT "Error: Failed to allocate row " + row_index
                PRINT "Requested: " + cols + " doubles"
                
                CLEANUP PROCEDURE:
                    COMMENT: Free all previously allocated rows
                    FOR cleanup_index from 0 to row_index - 1
                        IF new_matrix->data[cleanup_index] is not NULL
                            FREE new_matrix->data[cleanup_index]
                        END IF
                    END FOR
                    
                    COMMENT: Free row pointer array
                    FREE new_matrix->data
                    
                    COMMENT: Free matrix structure
                    FREE new_matrix
                    
                RETURN NULL
            END IF
            
    END FOR
    
    WHAT WE HAVE NOW:
        Complete matrix structure
        All memory allocated
        Elements contain garbage values (not yet initialized)
        
    MEMORY LAYOUT AT THIS POINT:
        Matrix structure: allocated ✓
        Row pointers array: allocated ✓
        Row 0 data: allocated ✓
        Row 1 data: allocated ✓
        ...
        Row (rows-1) data: allocated ✓
        
    TOTAL MEMORY USED:
        Matrix structure: ~20 bytes
        Row pointers: rows × 8 bytes
        Element data: rows × cols × 8 bytes
        TOTAL ≈ 20 + 8×rows + 8×rows×cols bytes

STEP 6: Initialize all values to zero
    PURPOSE: Ensure predictable, safe initial state
    
    FOR row_index from 0 to rows - 1
        FOR col_index from 0 to cols - 1
            
            SET new_matrix->data[row_index][col_index] to 0.0
            
            ALTERNATIVE METHODS:
                METHOD A: Use memset (faster but less portable)
                    memset(new_matrix->data[row_index], 0, 
                           cols × sizeof(double))
                    NOTE: Works because 0.0 is all-bits-zero in IEEE 754
                    
                METHOD B: Use calloc instead of malloc (in Step 5)
                    calloc automatically initializes to zero
                    Slightly slower but cleaner
                    
        END FOR
    END FOR
    
    WHY INITIALIZE TO ZERO:
        - Prevents undefined behavior
        - Makes debugging easier (predictable initial state)
        - Mathematical convenience (additive identity)
        - Many algorithms assume zero initialization
        
    WHAT WE HAVE NOW:
        Fully allocated matrix
        All elements set to 0.0
        Ready for use

STEP 7: Return the matrix pointer
    
    RETURN new_matrix
    
    WHAT CALLER RECEIVES:
        Pointer to Matrix structure
        Structure contains:
            - number_of_rows field (set correctly)
            - number_of_columns field (set correctly)
            - data field (pointing to allocated and initialized data)
            
    CALLER RESPONSIBILITIES:
        - Use the matrix for computations
        - Eventually call free_matrix() to deallocate
        - Not modify the structure fields directly (use provided functions)
        
═══════════════════════════════════════════════════════════
ENHANCED ERROR CHECKING VERSION
═══════════════════════════════════════════════════════════

FUNCTION create_matrix_safe(rows, cols)
    PURPOSE: More robust version with comprehensive error checking
    
    STEP 0: Validate inputs
        IF rows <= 0
            LOG_ERROR("Invalid rows: " + rows)
            RETURN NULL
        END IF
        
        IF cols <= 0
            LOG_ERROR("Invalid cols: " + cols)
            RETURN NULL
        END IF
        
        IF rows > MAX_REASONABLE_SIZE OR cols > MAX_REASONABLE_SIZE
            LOG_WARNING("Very large matrix requested")
        END IF
        
        CALCULATE total_elements = rows × cols
        
IF total_elements > MAX_ELEMENTS
            LOG_ERROR("Matrix too large: " + total_elements + " elements")
            RETURN NULL
        END IF
    
    THEN proceed with Steps 1-7 as described above
    
    FINALLY:
        LOG_INFO("Matrix created: " + rows + "×" + cols)
        RETURN new_matrix
    END
    
END FUNCTION

═══════════════════════════════════════════════════════════
MEMORY USAGE CALCULATION
═══════════════════════════════════════════════════════════

FOR A MATRIX OF DIMENSIONS rows × cols:

    Structure overhead:
        Matrix structure: sizeof(Matrix) ≈ 16-24 bytes
        
    Row pointer array:
        rows × sizeof(double*) = rows × 8 bytes (on 64-bit systems)
        
    Actual data:
        rows × cols × sizeof(double) = rows × cols × 8 bytes
        
    TOTAL:
        ≈ 24 + (8 × rows) + (8 × rows × cols) bytes
        ≈ 8 × rows × (1 + cols) + 24 bytes
        
    EXAMPLES:
        10×10 matrix: 8×10×11 + 24 = 904 bytes ≈ 0.9 KB
        100×100: 8×100×101 + 24 = 80,824 bytes ≈ 79 KB
        1000×1000: 8×1000×1001 + 24 ≈ 8 MB
        
═══════════════════════════════════════════════════════════
COMMON USAGE PATTERNS
═══════════════════════════════════════════════════════════

PATTERN 1: Create and initialize
    Matrix *m = create_matrix(10, 20);
    IF m is NULL:
        Handle error
    END IF
    // All elements are 0.0 by default
    
PATTERN 2: Create, check, use, destroy
    Matrix *weights = create_matrix(128, 784);
    IF weights is not NULL:
        // Use weights for computations
        ...
        free_matrix(weights);
    END IF
    
PATTERN 3: Multiple matrix creation with cleanup
    Matrix *A = create_matrix(100, 50);
    Matrix *B = create_matrix(50, 30);
    
    IF A is NULL OR B is NULL:
        IF A is not NULL: free_matrix(A)
        IF B is not NULL: free_matrix(B)
        RETURN error
    END IF
    
    // Use A and B
    ...
    
    free_matrix(A);
    free_matrix(B);

═══════════════════════════════════════════════════════════
OPTIMIZATION CONSIDERATIONS
═══════════════════════════════════════════════════════════

MEMORY ALIGNMENT:
    For better cache performance:
        Use aligned allocation (aligned_alloc)
        Align to 64-byte boundaries (cache line size)
        Can improve performance by 10-30%
        
CONTIGUOUS ALLOCATION ALTERNATIVE:
    Instead of allocating each row separately:
        Allocate all data in one block
        Set up row pointers into this block
        Better cache locality
        Faster allocation/deallocation
        
    MODIFIED STEP 4-5:
        Allocate single block: rows × cols × sizeof(double)
        Allocate row pointer array
        FOR each row:
            Set row pointer to: data_block + (row × cols)
        END FOR

MEMORY POOLING:
    For frequent matrix creation/destruction:
        Pre-allocate pool of matrices
        Reuse instead of malloc/free
        Reduces allocation overhead
        Reduces memory fragmentation
```

## Expanded Apply Function Elementwise
```
═══════════════════════════════════════════════════════════
FUNCTION: apply_function_elementwise
═══════════════════════════════════════════════════════════

PURPOSE:
    Apply any mathematical function to every element in a matrix
    Create a new matrix with transformed values
    Generic operation that enables sigmoid, tanh, ReLU, etc.
    Core building block for neural network forward/backward passes
    
FUNCTION SIGNATURE:
    apply_function_elementwise(input_matrix, function_to_apply)
    
INPUT PARAMETERS:
    
    input_matrix: 
        TYPE: Pointer to Matrix structure
        DESCRIPTION: Source matrix containing values to transform
        REQUIREMENTS: Must not be NULL, must be properly initialized
        
    function_to_apply:
        TYPE: Function pointer
        SIGNATURE: double (*function)(double)
        DESCRIPTION: Function that takes one double and returns one double
        EXAMPLES: sigmoid, tanh, exp, log, sqrt, custom functions
        REQUIREMENTS: Must handle all possible double values gracefully
        
RETURN VALUE:
    Pointer to newly created Matrix with transformed values
    Same dimensions as input_matrix
    NULL if allocation fails or input is invalid
    
VARIABLE DECLARATIONS:
    DECLARE rows as integer
    DECLARE cols as integer
    DECLARE output_matrix as pointer to Matrix
    DECLARE row_index as integer (loop counter)
    DECLARE col_index as integer (loop counter)
    DECLARE current_value as double
    DECLARE transformed_value as double
    
═══════════════════════════════════════════════════════════
DETAILED IMPLEMENTATION
═══════════════════════════════════════════════════════════

INPUT VALIDATION:
    PURPOSE: Ensure safe operation before processing
    
    IF input_matrix is NULL
        PRINT "Error: input_matrix is NULL"
        PRINT "Cannot apply function to non-existent matrix"
        RETURN NULL
    END IF
    
    ADDITIONAL CHECKS (Recommended):
        IF input_matrix->data is NULL
            PRINT "Error: Matrix data array is NULL"
            RETURN NULL
        END IF
        
        IF input_matrix->number_of_rows <= 0
            PRINT "Error: Invalid row count: " + input_matrix->number_of_rows
            RETURN NULL
        END IF
        
        IF input_matrix->number_of_columns <= 0
            PRINT "Error: Invalid column count: " + input_matrix->number_of_columns
            RETURN NULL
        END IF
        
        IF function_to_apply is NULL
            PRINT "Error: No function provided"
            PRINT "function_to_apply pointer is NULL"
            RETURN NULL
        END IF
        
    WHY VALIDATION IS CRITICAL:
        - Prevents segmentation faults
        - Provides clear error messages
        - Enables easier debugging
        - Fails safely rather than corrupting memory

STEP 1: Get dimensions from input matrix
    PURPOSE: Store dimensions for creating output and iteration
    
    GET input_matrix->number_of_rows
    STORE in rows variable
    
    GET input_matrix->number_of_columns
    STORE in cols variable
    
    RATIONALE:
        - Avoids repeated pointer dereferences
        - Slightly more efficient (compiler may optimize anyway)
        - More readable code
        - Local variables easier to work with
        
    ALTERNATIVE:
        Can use input_matrix->number_of_rows directly in loops
        But storing in local variables is cleaner
        
    WHAT WE HAVE NOW:
        rows = number of rows in input
        cols = number of columns in input
        Ready to create output matrix of same size

STEP 2: Create output matrix
    PURPOSE: Allocate space for transformed values
    
    CALL create_matrix(rows, cols)
    STORE returned pointer in output_matrix
    
    ERROR HANDLING:
        IF output_matrix is NULL
            PRINT "Error: Failed to create output matrix"
            PRINT "Requested dimensions: " + rows + "×" + cols
            PRINT "Possible causes:"
            PRINT "  - Insufficient memory"
            PRINT "  - Invalid dimensions"
            PRINT "  - System resource limits"
            RETURN NULL
        END IF
        
    MEMORY STATE:
        output_matrix now points to allocated matrix
        All elements initialized to 0.0 (from create_matrix)
        Same dimensions as input_matrix
        Ready to receive transformed values
        
    IMPORTANT NOTE:
        Caller is responsible for freeing output_matrix
        Must call free_matrix(output_matrix) when done
        Failure to do so causes memory leak

STEP 3: Process each element
    PURPOSE: Apply transformation function to every matrix element
    
    OUTER LOOP - Iterate through rows:
        FOR row_index from 0 to rows - 1
            
            INNER LOOP - Iterate through columns:
                FOR col_index from 0 to cols - 1
                    
                    SUBSTEP 3a: Get current input value
                        GET input_matrix->data[row_index][col_index]
                        STORE in current_value
                        
                        WHY STORE IN VARIABLE:
                            - More readable
                            - Can add debugging/logging
                            - Avoids repeated array access
                            - Makes next step clearer
                            
                        ALTERNATIVE (Direct):
                            transformed_value = function_to_apply(
                                input_matrix->data[row_index][col_index]
                            );
                            
                    SUBSTEP 3b: Apply transformation function
                        CALL function_to_apply(current_value)
                        STORE result in transformed_value
                        
                        WHAT HAPPENS HERE:
                            Function pointer is dereferenced
                            Function is called with current_value
                            Function returns transformed result
                            Result stored in transformed_value
                            
                        EXAMPLES:
                            If function_to_apply = sigmoid:
                                current_value = 0.5
                                transformed_value = sigmoid(0.5) ≈ 0.622
                                
                            If function_to_apply = tanh:
                                current_value = -1.0
                                transformed_value = tanh(-1.0) ≈ -0.762
                                
                        ERROR HANDLING CONSIDERATIONS:
                            Some functions may return NaN for certain inputs
                            Consider checking: if isnan(transformed_value)
                            Some functions may return infinity
                            Consider checking: if isinf(transformed_value)
                            
                    SUBSTEP 3c: Store transformed value in output
                        SET output_matrix->data[row_index][col_index] 
                            to transformed_value
                            
                        WHAT WE'RE DOING:
                            Writing result to corresponding position
                            Preserves matrix structure
                            Element (i,j) in input → Element (i,j) in output
                            
                        ELEMENT-WISE OPERATION:
                            This is the defining characteristic
                            Each output element depends only on corresponding input
                            No interaction between different elements
                            Perfectly parallelizable
                            
                END FOR (columns)
                
            END FOR (rows)
            
    ITERATION ORDER CONSIDERATIONS:
        
        WHY ROW-MAJOR ORDER:
            Rows are stored contiguously
            Better cache locality
            Fewer cache misses
            Typical speedup: 2-5x over column-major
            
        CACHE BEHAVIOR:
            When accessing data[i][j], CPU loads cache line
            Cache line contains data[i][j], data[i][j+1], etc.
            Next iteration (j+1) is already in cache
            Very efficient
            
        ALTERNATIVE (Column-major, slower):
            FOR col_index from 0 to cols - 1
                FOR row_index from 0 to rows - 1
                    // Process data[row_index][col_index]
                END FOR
            END FOR
            
            This jumps between rows (separate cache lines)
            Cache thrashing, slower performance
            
    WHAT WE HAVE AFTER STEP 3:
        output_matrix fully populated
        Every element transformed
        Ready to return to caller

STEP 4: Return the output matrix
    
    RETURN output_matrix
    
    WHAT CALLER RECEIVES:
        Pointer to Matrix structure
        Matrix contains transformed values
        Same dimensions as input
        Caller must free when done
        
    CALLER RESPONSIBILITIES:
        Use the output matrix
        Call free_matrix(output_matrix) when done
        Do not modify original input_matrix (unchanged)
        
═══════════════════════════════════════════════════════════
FUNCTION POINTER MECHANICS
═══════════════════════════════════════════════════════════

WHAT IS A FUNCTION POINTER:
    Variable that stores the address of a function
    Can be called like a regular function
    Enables runtime selection of functions
    Basis for callbacks and higher-order functions
    
DECLARATION SYNTAX:
    double (*function_ptr)(double);
    
    BREAKDOWN:
        double: return type
        (*function_ptr): pointer to function
        (double): parameter type
        
ASSIGNMENT EXAMPLES:
    function_ptr = sigmoid;
    function_ptr = tanh_activation;
    function_ptr = &exp;  // & is optional
    
CALLING THROUGH POINTER:
    result = function_ptr(5.0);
    result = (*function_ptr)(5.0);  // Equivalent, explicit dereference
    
USAGE IN THIS FUNCTION:
    apply_function_elementwise(matrix, sigmoid);
    apply_function_elementwise(matrix, tanh_activation);
    apply_function_elementwise(matrix, custom_function);

═══════════════════════════════════════════════════════════
PERFORMANCE CHARACTERISTICS
═══════════════════════════════════════════════════════════

TIME COMPLEXITY:
    O(rows × cols) - must visit every element
    Linear in the number of elements
    Optimal (can't do better than visiting each element)
    
SPACE COMPLEXITY:
    O(rows × cols) - creates new matrix
    Additional space equals input size
    Does not modify input (could save space if allowed)
    
TYPICAL EXECUTION TIME:
    Small matrix (10×10 = 100 elements):
        ~1-5 microseconds
        
    Medium matrix (100×100 = 10,000 elements):
        ~100-500 microseconds
        
    Large matrix (1000×1000 = 1,000,000 elements):
        ~10-50 milliseconds
        
    Times vary based on:
        - Function complexity (exp vs simple arithmetic)
        - CPU speed and architecture
        - Memory access patterns
        - Cache efficiency
        
BOTTLENECKS:
    1. Function call overhead (pointer indirection)
    2. Memory allocation for output matrix
    3. Cache misses for large matrices
    4. Complexity of transformation function
    
OPTIMIZATION OPPORTUNITIES:
    1. Inline the function (remove pointer indirection)
    2. Vectorization (SIMD instructions)
    3. Parallelization (multiple threads/GPU)
    4. In-place operation (modify input, avoid allocation)

═══════════════════════════════════════════════════════════
OPTIMIZATION TECHNIQUES
═══════════════════════════════════════════════════════════

TECHNIQUE 1: In-place operation (when safe)
    
    FUNCTION apply_function_inplace(matrix, function)
        FOR each element in matrix:
            matrix->data[i][j] = function(matrix->data[i][j])
        END FOR
    END FUNCTION
    
    ADVANTAGES:
        - No memory allocation
        - Faster (no copying)
        - Lower memory usage
        
    DISADVANTAGES:
        - Destroys original data
        - Can't undo operation
        - Not always safe (depends on use case)
        
TECHNIQUE 2: Vectorization (SIMD)
    
    Use CPU vector instructions (SSE, AVX)
    Process 2-8 elements simultaneously
    Requires alignment and special intrinsics
    
    PSEUDO-SIMD CODE:
        FOR i from 0 to rows-1:
            FOR j from 0 to cols-1 STEP 4:
                LOAD 4 values into vector register
                APPLY function to all 4 (vectorized)
                STORE 4 results back
            END FOR
            HANDLE remaining elements (if cols not divisible by 4)
        END FOR
        
    SPEEDUP: 2-4x for simple functions
    
TECHNIQUE 3: Multi-threading
    
    Divide rows among threads
    Each thread processes subset of rows
    No synchronization needed (independent operations)
    
    PSEUDO-THREADED CODE:
        DETERMINE number_of_threads (e.g., 4)
        CALCULATE rows_per_thread = rows / number_of_threads
        
        FOR thread_id from 0 to number_of_threads-1:
            START_THREAD:
                start_row = thread_id × rows_per_thread
                end_row = start_row + rows_per_thread
                
                FOR row from start_row to end_row-1:
                    FOR col from 0 to cols-1:
                        Apply function
                    END FOR
                END FOR
            END_THREAD
        END FOR
        
        WAIT for all threads to complete
        
    SPEEDUP: Near-linear with thread count (e.g., 3.8x with 4 threads)
    
TECHNIQUE 4: GPU acceleration
    
    Transfer matrix to GPU memory
    Launch kernel with one thread per element
    Each thread applies function independently
    Transfer result back to CPU
    
    SPEEDUP: 10-100x for large matrices
    OVERHEAD: Data transfer time significant for small matrices
    BEST FOR: Large matrices (> 1000×1000)

═══════════════════════════════════════════════════════════
ERROR HANDLING AND EDGE CASES
═══════════════════════════════════════════════════════════

EDGE CASE 1: Empty matrix (0 rows or 0 cols)
    HANDLING:
        Should be caught in input validation
        If not, loops won't execute (safe but returns empty matrix)
        Better to explicitly reject
        
EDGE CASE 2: Very large matrix
    HANDLING:
        Output allocation may fail
        Return NULL and error message
        Caller should check return value
        
EDGE CASE 3: Function returns NaN
    DETECTION:
        After applying function: IF isnan(transformed_value)
        
    OPTIONS:
        A) Replace with 0.0 or other sentinel
        B) Print warning and continue
        C) Abort operation and return NULL
        D) Store NaN and let caller handle
        
    RECOMMENDATION: Option D (preserve NaN, caller decides)
    
EDGE CASE 4: Function returns infinity
    Similar handling to NaN
    May indicate numerical overflow
    Consider clamping to max/min finite value
    
EDGE CASE 5: Allocation fails mid-process
    If output matrix creation fails:
        Clean up and return NULL
        No other cleanup needed (didn't allocate partially)
        
EDGE CASE 6: 1×1 matrix
    Valid edge case
    Single element processed
    Loops execute once
    No special handling needed

═══════════════════════════════════════════════════════════
USAGE EXAMPLES
═══════════════════════════════════════════════════════════

EXAMPLE 1: Basic usage with sigmoid
    Matrix *input = create_matrix(3, 3);
    // ... fill input with values ...
    
    Matrix *output = apply_function_elementwise(input, sigmoid);
    
    IF output is not NULL:
        // Use output
        print_matrix(output);
        
        // Clean up
        free_matrix(output);
    END IF
    
EXAMPLE 2: Chaining operations
    Matrix *layer1 = apply_function_elementwise(weights, sigmoid);
    Matrix *layer2 = apply_function_elementwise(layer1, tanh_activation);
    
    // Clean up intermediate result
    free_matrix(layer1);
    
    // Use layer2
    ...
    free_matrix(layer2);
    
EXAMPLE 3: Custom function
    DEFINE custom_square(x):
        RETURN x * x
    END DEFINE
    
    Matrix *squared = apply_function_elementwise(matrix, custom_square);
    
EXAMPLE 4: With error checking
    Matrix *result = apply_function_elementwise(data, exp);
    
    IF result is NULL:
        PRINT "Operation failed"
        // Handle error appropriately
        RETURN error_code
    END IF
    
    // Check for numerical issues
    FOR each element in result:
        IF isnan(element) OR isinf(element):
            PRINT "Warning: numerical instability detected"
            BREAK
        END IF
    END FOR
    
EXAMPLE 5: Batch processing multiple matrices
    Matrix *matrices[10];
    Matrix *results[10];
    
    FOR i from 0 to 9:
        results[i] = apply_function_elementwise(matrices[i], sigmoid);
        IF results[i] is NULL:
            // Clean up previously allocated results
            FOR j from 0 to i-1:
                free_matrix(results[j]);
            END FOR
            RETURN error
        END IF
    END FOR
    
    // Use all results...
    
    // Clean up all
    FOR i from 0 to 9:
        free_matrix(results[i]);
    END FOR

═══════════════════════════════════════════════════════════
TESTING AND VERIFICATION
═══════════════════════════════════════════════════════════

TEST 1: Correct dimensions
    input = create_matrix(5, 7)
    output = apply_function_elementwise(input, sigmoid)
    VERIFY: output->rows = 5
    VERIFY: output->cols = 7
    
TEST 2: Function applied correctly
    input = create_matrix(2, 2)
    input->data[0][0] = 0.0
    input->data[0][1] = 1.0
    input->data[1][0] = -1.0
    input->data[1][1] = 2.0
    
    output = apply_function_elementwise(input, sigmoid)
    
    VERIFY: output->data[0][0] ≈ sigmoid(0.0) = 0.5
    VERIFY: output->data[0][1] ≈ sigmoid(1.0) ≈ 0.731
    VERIFY: output->data[1][0] ≈ sigmoid(-1.0) ≈ 0.269
    VERIFY: output->data[1][1] ≈ sigmoid(2.0) ≈ 0.881
    
TEST 3: Original unchanged
    input = create_matrix(3, 3)
    // ... fill with known values ...
    
    Matrix *copy = dupe_matrix(input)
    output = apply_function_elementwise(input, some_function)
    
    VERIFY: input matches copy (element by element)
    
TEST 4: NULL handling
    output = apply_function_elementwise(NULL, sigmoid)
    VERIFY: output is NULL
    
    output = apply_function_elementwise(input, NULL)
    VERIFY: output is NULL
    
TEST 5: Large matrix
    large = create_matrix(1000, 1000)
    init_random(large, -10.0, 10.0)
    
    output = apply_function_elementwise(large, tanh_activation)
    VERIFY: output is not NULL
    VERIFY: All elements in range (-1, 1)
    
TEST 6: Identity function
    DEFINE identity(x):
        RETURN x
    END DEFINE
    
    output = apply_function_elementwise(input, identity)
    VERIFY: output equals input (element by element)

═══════════════════════════════════════════════════════════
MEMORY NOTES
═══════════════════════════════════════════════════════════

MEMORY LEAK PREVENTION:
    CRITICAL: Caller MUST free returned matrix
    Use tools: valgrind, AddressSanitizer
    Pattern: Always pair create with free
    
ALLOCATION FAILURE:
    If create_matrix fails inside function
    Function returns NULL immediately
    No cleanup needed (nothing allocated yet)
    Caller checks for NULL return
    
MEMORY USAGE PATTERN:
    Input matrix: Already allocated (not our responsibility)
    Output matrix: We allocate, caller frees
    No temporary allocations
    Stack usage: Minimal (~50-100 bytes for locals)
    
TYPICAL MEMORY FOOTPRINT:
    10×10 matrix: ~1 KB output allocation
    100×100 matrix: ~80 KB output allocation
    1000×1000 matrix: ~8 MB output allocation
    
MEMORY BANDWIDTH:
    Reading input: rows × cols × 8 bytes
    Writing output: rows × cols × 8 bytes
    Total: 2 × rows × cols × 8 bytes
    
    Example (1000×1000):
        Read: 8 MB
        Write: 8 MB
        Total: 16 MB transferred
        
    Modern CPU bandwidth: ~20-50 GB/s
    Time for memory transfer: ~0.3-0.8 ms
    Usually not the bottleneck

END FUNCTION

═══════════════════════════════════════════════════════════
FUNCTION: apply_sigmoid_to_matrix
═══════════════════════════════════════════════════════════

PURPOSE:
    Convenience wrapper for applying sigmoid to entire matrix
    Simplifies common operation in neural networks
    Cleaner syntax than using apply_function_elementwise directly
    
FUNCTION SIGNATURE:
    apply_sigmoid_to_matrix(input_matrix)
    
INPUT PARAMETERS:
    input_matrix: Pointer to Matrix structure
                  Matrix containing values to transform with sigmoid
                  
RETURN VALUE:
    Pointer to new Matrix with sigmoid applied to all elements
    NULL if operation fails
    
═══════════════════════════════════════════════════════════

STEP 1: Call general elementwise function
    PURPOSE: Delegate to the generic function
    
    CALL apply_function_elementwise(input_matrix, sigmoid)
        PARAMETER 1: input_matrix (pass through from caller)
        PARAMETER 2: sigmoid (function pointer to sigmoid function)
        
    STORE returned pointer in sigmoid_matrix
    
    WHAT HAPPENS:
        apply_function_elementwise receives:
            - The input matrix
            - Pointer to sigmoid function
            
        It then:
            - Creates output matrix
            - Applies sigmoid to each element
            - Returns result
            
    ERROR PROPAGATION:
        If apply_function_elementwise returns NULL:
            sigmoid_matrix will be NULL
            We pass this NULL back to caller
            Caller is responsible for checking
            
STEP 2: Return the result
    RETURN sigmoid_matrix
    
    ALTERNATIVES CONSIDERED:
        Could add error checking here
        Could print diagnostic messages
        But keeping it simple is better
        Error handling in apply_function_elementwise is sufficient
        
═══════════════════════════════════════════════════════════
WHY THIS WRAPPER IS USEFUL
═══════════════════════════════════════════════════════════

READABILITY:
    Compare:
        Matrix *output = apply_sigmoid_to_matrix(hidden);
        
    vs:
        Matrix *output = apply_function_elementwise(hidden, sigmoid);
        
    First version is more self-documenting
    Intent is immediately clear
    
CONSISTENCY:
    Matches naming convention of matrix operations
    add_matrix(), multiply_matrix(), apply_sigmoid_to_matrix()
    Easier to remember and discover
    
TYPE SAFETY:
    Ensures sigmoid function is used
    Can't accidentally pass wrong function
    Less error-prone
    
FUTURE FLEXIBILITY:
    If sigmoid implementation changes
    Only need to update in one place
    Could add sigmoid-specific optimizations
    Could add parameter for sigmoid variant
    
USAGE IN NEURAL NETWORKS:
    # Neural network forward pass
    Matrix *hidden = multiply_matrix(weights1, input);
    Matrix *hidden_activated = apply_sigmoid_to_matrix(hidden);
    Matrix *output = multiply_matrix(weights2, hidden_activated);
    Matrix *output_activated = apply_sigmoid_to_matrix(output);
    
    Clear data flow
    Each step is explicit
    Easy to understand

═══════════════════════════════════════════════════════════
PERFORMANCE NOTES
═══════════════════════════════════════════════════════════

OVERHEAD:
    Function call overhead: Negligible (~few CPU cycles)
    No performance penalty vs direct call
    Compiler may inline (zero overhead)
    
OPTIMIZATION:
    If compiler supports it:
        Mark as inline function
        Zero runtime cost
        Same as calling apply_function_elementwise directly
        
    In C header file:
        static inline Matrix* apply_sigmoid_to_matrix(Matrix *m) {
            return apply_function_elementwise(m, sigmoid);
        }

END FUNCTION


═══════════════════════════════════════════════════════════
FUNCTION: apply_tanh_to_matrix
═══════════════════════════════════════════════════════════

PURPOSE:
    Convenience wrapper for applying tanh to entire matrix
    Common activation function in recurrent neural networks
    Zero-centered alternative to sigmoid
    
IMPLEMENTATION:
    Identical structure to apply_sigmoid_to_matrix
    Simply passes tanh_activation instead of sigmoid
    
STEP 1: Call general elementwise function
    CALL apply_function_elementwise(input_matrix, tanh_activation)
    STORE result in tanh_matrix
    
STEP 2: Return result
    RETURN tanh_matrix
    
USAGE EXAMPLES:
    # RNN/LSTM computations
    Matrix *cell_state = apply_tanh_to_matrix(candidate);
    Matrix *hidden_state = apply_tanh_to_matrix(updated_cell);
    
    # Classic neural network
    Matrix *layer1 = apply_tanh_to_matrix(z1);
    Matrix *layer2 = apply_tanh_to_matrix(z2);

END FUNCTION


═══════════════════════════════════════════════════════════
FUNCTION: apply_sigmoid_derivative_to_matrix
═══════════════════════════════════════════════════════════

PURPOSE:
    Apply sigmoid derivative to entire matrix
    Critical for backpropagation in neural networks
    Used to compute gradients during training
    
IMPLEMENTATION:
    Same pattern as activation functions
    Applies sigmoid_derivative to each element
    
STEP 1: Call general elementwise function
    CALL apply_function_elementwise(input_matrix, sigmoid_derivative)
    STORE result in derivative_matrix
    
    NOTE ON INPUT:
        input_matrix typically contains pre-activation values (z)
        OR could contain sigmoid activations (a)
        Depends on how sigmoid_derivative is implemented
        
        If sigmoid_derivative(x) computes from raw input x:
            Pass the z values (weighted sums)
            
        If you have optimization that takes sigmoid(x):
            Pass the activations (a values)
            Use formula: a * (1 - a)
            
STEP 2: Return result
    RETURN derivative_matrix
    
USAGE IN BACKPROPAGATION:
    # Forward pass (save activations)
    z1 = multiply_matrix(W1, input)
    a1 = apply_sigmoid_to_matrix(z1)
    
    # Backward pass
    da1 = // ... gradient from next layer
    dz1_local = apply_sigmoid_derivative_to_matrix(z1)
    dz1 = elementwise_multiply(da1, dz1_local)
    
    # Continue backprop...
    
OPTIMIZATION OPPORTUNITY:
    If activations are available:
        It's faster to compute: a * (1-a)
        Than to recompute sigmoid then derivative
        Consider alternative function:
            apply_sigmoid_derivative_from_activation(activation_matrix)

END FUNCTION


═══════════════════════════════════════════════════════════
FUNCTION: apply_tanh_derivative_to_matrix
═══════════════════════════════════════════════════════════

PURPOSE:
    Apply tanh derivative to entire matrix
    Used in backpropagation through tanh layers
    Formula: 1 - tanh²(x)
    
IMPLEMENTATION:
    Same pattern as other derivative functions
    
STEP 1: Call general elementwise function
    CALL apply_function_elementwise(input_matrix, tanh_derivative)
    STORE result in derivative_matrix
    
STEP 2: Return result
    RETURN derivative_matrix
    
USAGE IN RNN BACKPROPAGATION:
    # Forward pass
    c_tilde = apply_tanh_to_matrix(candidate)
    
    # Backward pass
    dc_tilde = // ... gradient from loss
    dc_candidate = apply_tanh_derivative_to_matrix(candidate)
    dcandidate = elementwise_multiply(dc_tilde, dc_candidate)
    
    # Continue backprop through previous layers...
    
OPTIMIZATION OPPORTUNITY:
    Similar to sigmoid, if tanh activations available:
        Faster to compute: 1 - a²
        Where a = tanh(x)
        Avoids recomputing tanh
        
    Consider implementing:
        apply_tanh_derivative_from_activation(activation_matrix)
            FOR each element a in activation_matrix:
                derivative = 1 - a²
            END FOR

END FUNCTION
```

## Expanded Free Matrix Function
```
═══════════════════════════════════════════════════════════
FUNCTION: free_matrix
═══════════════════════════════════════════════════════════

PURPOSE:
    Properly deallocate all memory associated with a matrix
    Prevent memory leaks
    Clean up in reverse order of allocation
    Essential for long-running programs
    
FUNCTION SIGNATURE:
    free_matrix(matrix)
    
INPUT PARAMETERS:
    matrix: Pointer to Matrix structure to be deallocated
            Can be NULL (function handles gracefully)
            
RETURN VALUE:
    None (void function)
    After call, matrix pointer is invalid (dangling)
    
CRITICAL IMPORTANCE:
    Every matrix created with create_matrix MUST be freed
    Failing to free causes memory leaks
    Memory leaks accumulate over time
    Eventually exhaust system memory
    Program may crash or slow down dramatically
    
═══════════════════════════════════════════════════════════
DETAILED IMPLEMENTATION
═══════════════════════════════════════════════════════════

VALIDATION:
    PURPOSE: Handle NULL pointer gracefully
    
    IF matrix is NULL
        RETURN immediately
        COMMENT: Nothing to free
    END IF
    
    WHY THIS IS IMPORTANT:
        Calling free(NULL) is safe in C
        But dereferencing NULL causes segfault
        Must check before accessing matrix->data
        Defensive programming practice
        Allows calling free_matrix unconditionally
        
    USAGE PATTERN:
        Matrix *m = create_matrix(10, 10);
        IF some_error:
            free_matrix(m);  // Safe even if m is NULL
            RETURN
        END IF

STEP 1: Free each row
    PURPOSE: Deallocate memory for each row's data
    
    FOR row_index from 0 to matrix->number_of_rows - 1
        
        NESTED VALIDATION:
            IF matrix->data[row_index] is not NULL
                
                FREE memory at matrix->data[row_index]
                
                OPTIONAL: Set to NULL after freeing
                    matrix->data[row_index] = NULL
                    COMMENT: Prevents double-free bugs
                    
            END IF
            
        WHY CHECK FOR NULL:
            In case of partial allocation failure
            Some rows might not have been allocated
            Calling free(NULL) is safe, but check is clearer
            Defensive programming
            
    END FOR
    
    WHAT THIS DOES:
        Frees the actual data arrays
        Each row was allocated separately in create_matrix
        Must free each one individually
        After this step, all element storage is freed
        
    ORDER MATTERS:
        Must free row data BEFORE freeing row pointer array
        Reverse order of allocation
        If we freed data array first, we'd lose row pointers
        Would create memory leak (orphaned row data)
        
    MEMORY STATE AFTER STEP 1:
        Matrix structure: still allocated
        Row pointer array: still allocated
        Row data: all freed ✓
        
    ALTERNATIVE (More robust):
        FOR row_index from 0 to matrix->number_of_rows - 1
            IF matrix->data is not NULL
                IF matrix->data[row_index] is not NULL
                    FREE matrix->data[row_index]
                    matrix->data[row_index] = NULL
                END IF
            END IF
        END FOR

STEP 2: Free row pointer array
    PURPOSE: Deallocate the array of row pointers
    
    IF matrix->data is not NULL
        
        FREE memory at matrix->data
        
        OPTIONAL: Set to NULL
            matrix->data = NULL
            COMMENT: Prevents use-after-free bugs
            
    END IF
    
    WHY CHECK FOR NULL:
        If create_matrix failed partway through
        data might never have been allocated
        Always check before freeing
        
    WHAT THIS DOES:
        Frees the array that held row pointers
        This array was allocated in create_matrix Step 4
        After this, no row pointer storage remains
        
    MEMORY STATE AFTER STEP 2:
        Matrix structure: still allocated
        Row pointer array: freed ✓
        Row data: freed ✓

STEP 3: Free matrix structure itself
    PURPOSE: Deallocate the Matrix structure
    
    FREE memory at matrix
    
    COMMENT: Do not set matrix = NULL here
    REASON: 'matrix' is a local parameter copy
            Setting it to NULL doesn't affect caller's pointer
            Caller is responsible for managing their pointer
            
    WHAT THIS DOES:
        Frees the Matrix structure itself
        This was allocated in create_matrix Step 1
        After this, everything is freed
        
    MEMORY STATE AFTER STEP 3:
        Matrix structure: freed ✓
        Row pointer array: freed ✓
        Row data: freed ✓
        All memory returned to system

═══════════════════════════════════════════════════════════
CRITICAL CONCEPTS
═══════════════════════════════════════════════════════════

DEALLOCATION ORDER:
    MUST be reverse of allocation order:
        
    ALLOCATION ORDER (create_matrix):
        1. Matrix structure
        2. Row pointer array
        3. Each row's data
        
    DEALLOCATION ORDER (free_matrix):
        1. Each row's data
        2. Row pointer array
        3. Matrix structure
        
    WHY REVERSE ORDER:
        Need pointers to access what they point to
        Can't free container before contents
        Like unpacking nested boxes from inside out
        
DANGLING POINTER PROBLEM:
    After calling free_matrix(m):
        Pointer 'm' still contains the old address
        But memory at that address is freed
        Dereferencing 'm' is undefined behavior
        May crash, may return garbage, may "work" (dangerous)
        
    SOLUTION (Caller's responsibility):
        Matrix *m = create_matrix(10, 10);
        // ... use m ...
        free_matrix(m);
        m = NULL;  // Explicitly mark as invalid
        
        IF m is not NULL:
            use(m);  // Safe check
        END IF

DOUBLE-FREE BUG:
    Calling free on same memory twice
    Undefined behavior (usually crash)
    
    EXAMPLE OF BUG:
        Matrix *m = create_matrix(5, 5);
        free_matrix(m);
        free_matrix(m);  // ERROR! Double free!
        
    PREVENTION:
        Set to NULL after freeing:
            free_matrix(m);
            m = NULL;
            
        Check before freeing:
            IF m is not NULL:
                free_matrix(m);
                m = NULL;
            END IF

MEMORY LEAK:
    Allocated memory that is never freed
    Memory becomes inaccessible but not returned to system
    Accumulates over time
    
    EXAMPLE OF LEAK:
        Matrix *m = create_matrix(100, 100);
        // ... use m ...
        // Forgot to call free_matrix(m)!
        m = NULL;  // Memory lost forever (this program session)
        
    DETECTION TOOLS:
        valgrind (Linux/Mac):
            valgrind --leak-check=full ./program
            
        AddressSanitizer (Clang/GCC):
            Compile with -fsanitize=address
            
        Visual Studio (Windows):
            Built-in memory leak detector

═══════════════════════════════════════════════════════════
USAGE PATTERNS
═══════════════════════════════════════════════════════════

PATTERN 1: Basic usage
    Matrix *m = create_matrix(10, 10);
    IF m is not NULL:
        // Use matrix
        ...
        free_matrix(m);
        m = NULL;
    END IF

PATTERN 2: Error handling with cleanup
    Matrix *A = create_matrix(100, 50);
    Matrix *B = create_matrix(50, 30);
    Matrix *C = NULL;
    
    IF A is NULL OR B is NULL:
        GOTO cleanup;
    END IF
    
    C = multiply_matrix(A, B);
    IF C is NULL:
        GOTO cleanup;
    END IF
    
    // Use matrices...
    
    cleanup:
        free_matrix(A);
        free_matrix(B);
        free_matrix(C);  // Safe even if NULL

PATTERN 3: Function that creates and returns matrix
    FUNCTION compute_something(input):
        result = create_matrix(10, 10);
        IF result is NULL:
            RETURN NULL
        END IF
        
        // Fill result with computed values
        ...
        
        RETURN result
        COMMENT: Caller must free result
    END FUNCTION
    
    // Caller code:
    Matrix *output = compute_something(data);
    IF output is not NULL:
        use(output);
        free_matrix(output);
    END IF

PATTERN 4: Array of matrices
    Matrix *layers[5];
    
    // Create multiple matrices
    FOR i from 0 to 4:
        layers[i] = create_matrix(100, 100);
        IF layers[i] is NULL:
            // Clean up previously created matrices
            FOR j from 0 to i-1:
                free_matrix(layers[j]);
            END FOR
            RETURN error;
        END IF
    END FOR
    
    // Use matrices...
    
    // Clean up all
    FOR i from 0 to 4:
        free_matrix(layers[i]);
    END FOR

PATTERN 5: Conditional freeing
    Matrix *temp = NULL;
    
    IF need_temporary:
        temp = create_matrix(50, 50);
        // Use temp...
    END IF
    
    // Later, safe to free regardless
    free_matrix(temp);  // Safe even if temp is NULL

═══════════════════════════════════════════════════════════
COMMON MISTAKES AND BUGS
═══════════════════════════════════════════════════════════

MISTAKE 1: Forgetting to free
    Matrix *m = create_matrix(1000, 1000);
    // Use m
    // Program ends without free_matrix(m)
    
    CONSEQUENCE: Memory leak
    IMPACT: 8 MB leaked per call
    FIX: Always pair create with free

MISTAKE 2: Using after free
    Matrix *m = create_matrix(10, 10);
    free_matrix(m);
    print_matrix(m);  // ERROR! Undefined behavior
    
    CONSEQUENCE: Crash or garbage data
    FIX: Set to NULL after free, check before use

MISTAKE 3: Double free
    Matrix *m = create_matrix(10, 10);
    free_matrix(m);
    free_matrix(m);  // ERROR!
    
    CONSEQUENCE: Crash or corruption
    FIX: Set to NULL after first free

MISTAKE 4: Freeing stack variable
    Matrix m;  // Allocated on stack
    m.rows = 10;
    // ...
    free_matrix(&m);  // ERROR! Can't free stack memory
    
    CONSEQUENCE: Crash
    FIX: Only free heap-allocated matrices

MISTAKE 5: Partial cleanup
    Matrix *m = create_matrix(10, 10);
    free(m->data[0]);  // Freeing just one row
    free(m);           // ERROR! Incomplete cleanup
    
    CONSEQUENCE: Memory leak (other rows) or crash
    FIX: Use free_matrix() which handles everything

MISTAKE 6: Wrong order
    Matrix *m = create_matrix(10, 10);
    free(m);           // Free structure first
    free(m->data);     // ERROR! m is already freed
    
    CONSEQUENCE: Undefined behavior
    FIX: Follow correct deallocation order

═══════════════════════════════════════════════════════════
DEBUGGING TIPS
═══════════════════════════════════════════════════════════

SYMPTOM: Segmentation fault in free_matrix
    POSSIBLE CAUSES:
        1. Matrix pointer is NULL
        2. Matrix pointer is uninitialized
        3. Matrix was already freed (double free)
        4. Matrix structure is corrupted
        
    DEBUGGING STEPS:
        1. Check if pointer is NULL before function
        2. Print pointer value: printf("%p\n", (void*)matrix)
        3. Verify matrix was created successfully
        4. Use debugger to inspect matrix structure
        5. Enable AddressSanitizer: -fsanitize=address

SYMPTOM: Memory leak reported by valgrind
    EXAMPLE OUTPUT:
        100,000 bytes in 1 blocks are definitely lost
        
    DEBUGGING STEPS:
        1. Run: valgrind --leak-check=full --show-leak-kinds=all ./program
        2. Find allocation stack trace
        3. Search code for corresponding create_matrix call
        4. Verify free_matrix is called
        5. Check all return paths (errors too)

SYMPTOM: Heap corruption detected
    POSSIBLE CAUSES:
        1. Writing past array bounds before free
        2. Double free
        3. Freeing wrong pointer
        
    DEBUGGING STEPS:
        1. Enable guard pages: MALLOC_CHECK_=3 ./program
        2. Use AddressSanitizer
        3. Review all array accesses
        4. Check loop bounds

═══════════════════════════════════════════════════════════
TESTING FREE_MATRIX
═══════════════════════════════════════════════════════════

TEST 1: Normal usage
    Matrix *m = create_matrix(5, 5);
    ASSERT m is not NULL
    free_matrix(m);
    // No crash = success

TEST 2: NULL input
    free_matrix(NULL);
    // Should not crash

TEST 3: Multiple frees with NULL reset
    Matrix *m = create_matrix(5, 5);
    free_matrix(m);
    m = NULL;
    free_matrix(m);  // Should be safe

TEST 4: Large matrix
    Matrix *large = create_matrix(1000, 1000);
    free_matrix(large);
    // Verify with valgrind: no leaks

TEST 5: Memory leak test
    FOR i from 1 to 1000:
        Matrix *m = create_matrix(100, 100);
        free_matrix(m);
    END FOR
    // Monitor memory usage: should stay constant

TEST 6: Stress test
    FOR i from 1 to 10000:
        Matrix *m = create_matrix(10, 10);
        // Use m briefly
        free_matrix(m);
    END FOR
    // Should complete without errors

═══════════════════════════════════════════════════════════
PERFORMANCE CONSIDERATIONS
═══════════════════════════════════════════════════════════

DEALLOCATION SPEED:
    Small matrix (10×10): ~1-2 microseconds
    Medium matrix (100×100): ~10-20 microseconds
    Large matrix (1000×1000): ~100-200 microseconds
    
    Times include:
        - free() system calls
        - Memory manager bookkeeping
        - Possible memory defragmentation
        
OPTIMIZATION:
    Freeing is typically fast
    Not usually a bottleneck
    If freeing many matrices:
        Consider memory pooling
        Reuse matrices instead of free/create
        
MEMORY FRAGMENTATION:
    Repeated alloc/free can fragment heap
    Over time, malloc may become slower
    Solutions:
        - Use memory pools
        - Allocate fewer, larger blocks
        - Restart program periodically (long-running)

═══════════════════════════════════════════════════════════
ALTERNATIVES AND VARIATIONS
═══════════════════════════════════════════════════════════

VARIATION 1: Return status
    FUNCTION free_matrix_checked(matrix):
        IF matrix is NULL:
            RETURN ERROR_NULL_POINTER
        END IF
        
        // Free as normal
        ...
        
        RETURN SUCCESS
    END FUNCTION

VARIATION 2: Reference counting
    Add reference count to Matrix structure
    Only free when count reaches zero
    Enables shared ownership
    More complex but powerful

VARIATION 3: Memory pool version
    FUNCTION free_matrix_pool(matrix, pool):
        // Return matrix to pool instead of freeing
        pool_return(matrix)
    END FUNCTION
    
    Faster for repeated alloc/free cycles

VARIATION 4: Debug version
    FUNCTION free_matrix_debug(matrix):
        IF matrix is NULL:
            PRINT "Warning: Attempted to free NULL matrix"
            RETURN
        END IF
        
        PRINT "Freeing matrix: " + matrix->rows + "×" + matrix->cols
        PRINT "Address: " + pointer_to_string(matrix)
        
        // Fill with sentinel value before freeing
        FOR each element:
            element = NaN or 0xDEADBEEF
        END FOR
        
        // Free as normal
        ...
        
        PRINT "Matrix freed successfully"
    END FUNCTION

END FUNCTION

═══════════════════════════════════════════════════════════
FUNCTION: test_sigmoid_basic
═══════════════════════════════════════════════════════════

PURPOSE:
    Verify sigmoid function works correctly
    Test known mathematical properties
    Ensure implementation matches specification
    Catch regression errors
    Validate edge cases
    
STRATEGY:
    Test specific input values with known outputs
    Test mathematical properties (symmetry)
    Test boundary behavior
    Verify function shape characteristics
    
VARIABLE DECLARATIONS:
    DECLARE tolerance as double = 0.0001
        PURPOSE: Acceptable error margin for floating-point comparison
        RATIONALE: Exact equality rarely works with floating-point
        VALUE: 0.0001 = 1/10000th precision
        
    DECLARE test_passed_count as integer = 0
        PURPOSE: Count successful tests
        USED FOR: Summary statistics
        
    DECLARE test_failed_count as integer = 0
        PURPOSE: Count failed tests
        USED FOR: Summary statistics and return code
        
    DECLARE result as double
        PURPOSE: Store function output for checking
        
    DECLARE difference as double
        PURPOSE: Store absolute difference for comparison
        
═══════════════════════════════════════════════════════════
DETAILED TEST CASES
═══════════════════════════════════════════════════════════

TEST 1: Sigmoid of zero equals 0.5
    PURPOSE: Verify midpoint property
    MATHEMATICAL FACT: σ(0) = 1/(1+e^0) = 1/2 = 0.5
    
    CALL sigmoid(0.0)
    STORE result
    
    CALCULATE absolute difference from 0.5:
        difference = abs(result - 0.5)
        ALTERNATIVE: difference = fabs(result - 0.5)
        
    IF difference is less than tolerance:
        PRINT "✓ Test 1 PASSED: sigmoid(0) = " + result
        PRINT "  Expected: 0.5, Got: " + result
        PRINT "  Error: " + difference
        INCREMENT test_passed_count
    ELSE:
        PRINT "✗ Test 1 FAILED: sigmoid(0) = " + result
        PRINT "  Expected: 0.5, Got: " + result
        PRINT "  Error: " + difference + " exceeds tolerance " + tolerance
        INCREMENT test_failed_count
    END IF
    
    WHY THIS TEST IS IMPORTANT:
        - Midpoint is critical for sigmoid
        - Common initialization point for weights
        - If this fails, fundamental issue exists
        - Easy to verify by hand calculation
        
TEST 2: Sigmoid of large positive number approaches 1
    PURPOSE: Verify saturation behavior at positive infinity
    MATHEMATICAL FACT: lim(x→∞) σ(x) = 1
    
    CALL sigmoid(10.0)
    STORE result
    
    THRESHOLD CHECK:
        IF result is greater than 0.99:
            PRINT "✓ Test 2 PASSED: sigmoid(10) = " + result
            PRINT "  Result > 0.99 (approaches 1) ✓"
            INCREMENT test_passed_count
        ELSE:
            PRINT "✗ Test 2 FAILED: sigmoid(10) = " + result
            PRINT "  Expected > 0.99, got " + result
            INCREMENT test_failed_count
        END IF
        
    ADDITIONAL CHECKS (Optional):
        IF result > 1.0:
            PRINT "  WARNING: Result exceeds 1.0!"
            PRINT "  Sigmoid must be in range (0,1)"
        END IF
        
        IF result < 0.0:
            PRINT "  ERROR: Result is negative!"
            PRINT "  Sigmoid must be positive"
        END IF
    
    WHY THIS TEST:
        - Verifies correct asymptotic behavior
        - Tests exponential calculation for large negative exponents
        - Ensures no overflow issues
        - Common in trained neural networks (strong activations)
        
    NUMERICAL NOTE:
        sigmoid(10) = 0.9999546021...
        Very close to 1 but not exactly 1
        Using > 0.99 allows some tolerance
        Could use tighter bound: > 0.999
        
TEST 3: Sigmoid of large negative number approaches 0
    PURPOSE: Verify saturation behavior at negative infinity
    MATHEMATICAL FACT: lim(x→-∞) σ(x) = 0
    
    CALL sigmoid(-10.0)
    STORE result
    
    IF result is less than 0.01:
        PRINT "✓ Test 3 PASSED: sigmoid(-10) = " + result
        PRINT "  Result < 0.01 (approaches 0) ✓"
        INCREMENT test_passed_count
    ELSE:
        PRINT "✗ Test 3 FAILED: sigmoid(-10) = " + result
        PRINT "  Expected < 0.01, got " + result
        INCREMENT test_failed_count
    END IF
    
    ADDITIONAL VALIDATION:
        IF result < 0.0:
            PRINT "  ERROR: Negative output!"
        END IF
        
        VERIFY SYMMETRY WITH TEST 2:
            result_10 = sigmoid(10)
            result_neg10 = sigmoid(-10)
            sum = result_10 + result_neg10
            IF abs(sum - 1.0) < tolerance:
                PRINT "  Symmetry verified: σ(10) + σ(-10) = " + sum + " ≈ 1"
            END IF
    
    WHY THIS TEST:
        - Tests other saturation direction
        - Verifies exp calculation for large positive exponents
        - Ensures no underflow issues
        - Common for strongly negative inputs
        
TEST 4: Sigmoid symmetry property
    PURPOSE: Verify mathematical property σ(x) + σ(-x) = 1
    MATHEMATICAL PROOF:
        σ(x) = 1/(1+e^(-x))
        σ(-x) = 1/(1+e^x) = e^(-x)/(e^(-x)+1) = (1+e^(-x)-1)/(1+e^(-x))
        σ(x) + σ(-x) = 1/(1+e^(-x)) + [1 - 1/(1+e^(-x))] = 1 ✓
        
    CALL sigmoid(2.0)
    STORE as result_pos
    
    CALL sigmoid(-2.0)
    STORE as result_neg
    
    CALCULATE sum:
        sum = result_pos + result_neg
        
    CALCULATE absolute difference from 1.0:
        difference = abs(sum - 1.0)
        
    IF difference is less than tolerance:
        PRINT "✓ Test 4 PASSED: sigmoid symmetry holds"
        PRINT "  σ(2) = " + result_pos
        PRINT "  σ(-2) = " + result_neg
        PRINT "  Sum = " + sum + " ≈ 1.0"
        PRINT "  Error: " + difference
        INCREMENT test_passed_count
    ELSE:
        PRINT "✗ Test 4 FAILED: symmetry broken"
        PRINT "  σ(2) + σ(-2) = " + sum
        PRINT "  Expected: 1.0"
        PRINT "  Error: " + difference
        INCREMENT test_failed_count
    END IF
    
    WHY THIS TEST:
        - Tests fundamental mathematical property
        - Catches sign errors in implementation
        - Verifies consistency across positive/negative inputs
        - Important for theoretical guarantees

OPTIONAL ADDITIONAL TESTS:

TEST 5: Specific known value
    EXPECTED: sigmoid(1) ≈ 0.7310585786
    CALL sigmoid(1.0)
    VERIFY result within tolerance of expected

TEST 6: Monotonicity
    FOR x from -5.0 to 5.0 in steps of 0.5:
        current = sigmoid(x)
        next = sigmoid(x + 0.5)
        VERIFY: next > current (strictly increasing)
    END FOR

TEST 7: Range verification
    FOR x from -20.0 to 20.0 in steps of 1.0:
        result = sigmoid(x)
        VERIFY: 0 < result < 1
    END FOR

TEST 8: Numerical stability
    CALL sigmoid(100.0)
    VERIFY: not NaN, not Inf
    CALL sigmoid(-100.0)
    VERIFY: not NaN, not Inf

═══════════════════════════════════════════════════════════

PRINT SUMMARY:
    PRINT blank line
    PRINT "========================================="
    PRINT "SIGMOID TEST SUMMARY"
    PRINT "========================================="
    PRINT "Tests passed: " + test_passed_count
    PRINT "Tests failed: " + test_failed_count
    PRINT "Total tests: " + (test_passed_count + test_failed_count)
    
    IF test_failed_count is 0:
        PRINT "Status: ALL TESTS PASSED ✓"
        PRINT "========================================="
    ELSE:
        PRINT "Status: SOME TESTS FAILED ✗"
        PRINT "========================================="
        PRINT "Please review failed tests above"
    END IF
    
    RETURN test_failed_count
        COMMENT: Return 0 for success, >0 for failure count
        USAGE: Can be used as exit code

END FUNCTION


═══════════════════════════════════════════════════════════
FUNCTION: test_tanh_basic
═══════════════════════════════════════════════════════════

PURPOSE:
    Verify tanh function works correctly
    Test mathematical properties specific to tanh
    Validate anti-symmetry and zero-centering
    Ensure proper range (-1, 1)
    
SIMILAR STRUCTURE TO test_sigmoid_basic but adapted for tanh properties

VARIABLE DECLARATIONS:
    DECLARE tolerance as double = 0.0001
    DECLARE test_passed_count as integer = 0
    DECLARE test_failed_count as integer = 0
    
═══════════════════════════════════════════════════════════

TEST 1: Tanh of zero equals 0
    PURPOSE: Verify zero-centered property
    MATHEMATICAL FACT: tanh(0) = 0 exactly
    
    CALL tanh_activation(0.0)
    STORE result
    
    CALCULATE absolute value of result:
        difference = abs(result)
        
    IF difference is less than tolerance:
        PRINT "✓ Test 1 PASSED: tanh(0) = " + result + " ≈ 0"
        INCREMENT test_passed_count
    ELSE:
        PRINT "✗ Test 1 FAILED: tanh(0) = " + result
        PRINT "  Expected: 0, Error: " + difference
        INCREMENT test_failed_count
    END IF
    
    WHY CRITICAL:
        - Zero-centering is key tanh advantage
        - Different from sigmoid (which gives 0.5 at x=0)
        - Important for gradient flow
        - If this fails, implementation is fundamentally wrong

TEST 2: Tanh of 1 equals approximately 0.7616
    PURPOSE: Verify specific calculated value
    MATHEMATICAL FACT: tanh(1) = (e - 1/e)/(e + 1/e) ≈ 0.7615941559...
    
    CALL tanh_activation(1.0)
    STORE result
    
    EXPECTED_VALUE = 0.7616
        COMMENT: Rounded to 4 decimal places
        
    CALCULATE difference:
        difference = abs(result - EXPECTED_VALUE)
        
    IF difference is less than 0.001:
        COMMENT: Using larger tolerance due to rounding
        PRINT "✓ Test 2 PASSED: tanh(1) = " + result + " ≈ 0.7616"
        INCREMENT test_passed_count
    ELSE:
        PRINT "✗ Test 2 FAILED: tanh(1) = " + result
        PRINT "  Expected: ~0.7616"
        INCREMENT test_failed_count
    END IF
    
    PRECISION NOTE:
        Could use exact value: 0.7615941559557649
        But 0.001 tolerance with 0.7616 is sufficient
        Tests correct magnitude and computation

TEST 3: Tanh antisymmetry property
    PURPOSE: Verify tanh(-x) = -tanh(x)
    MATHEMATICAL PROOF:
        tanh(-x) = (e^(-x) - e^x)/(e^(-x) + e^x)
                 = -(e^x - e^(-x))/(e^x + e^(-x))
                 = -tanh(x) ✓
        
    CALL tanh_activation(2.0)
    STORE as result_pos
    
    CALL tanh_activation(-2.0)
    STORE as result_neg
    
    CALCULATE sum:
        sum = result_pos + result_neg
        EXPECTED: Should be ≈ 0 (they cancel)
        
    CALCULATE absolute value:
        difference = abs(sum)
        
    IF difference is less than tolerance:
        PRINT "✓ Test 3 PASSED: tanh antisymmetry holds"
        PRINT "  tanh(2) = " + result_pos
        PRINT "  tanh(-2) = " + result_neg
        PRINT "  Sum = " + sum + " ≈ 0"
        INCREMENT test_passed_count
    ELSE:
        PRINT "✗ Test 3 FAILED: antisymmetry broken"
        PRINT "  tanh(2) + tanh(-2) = " + sum
        PRINT "  Expected: ~0"
        INCREMENT test_failed_count
    END IF
    
    ALTERNATIVE CHECK:
        ratio = result_neg / result_pos
        IF abs(ratio - (-1.0)) < tolerance:
            PRINT "  Alternative verification: ratio = " + ratio + " ≈ -1"
        END IF
    
    WHY THIS TEST:
        - Fundamental property of tanh (odd function)
        - Different from sigmoid (which has different symmetry)
        - Tests sign handling
        - Important for understanding function behavior

TEST 4: Tanh bounds approaching 1
    PURPOSE: Verify saturation at large positive input
    MATHEMATICAL FACT: lim(x→∞) tanh(x) = 1
    
    CALL tanh_activation(100.0)
    STORE result
    
    IF result is between 0.99 and 1.01:
        COMMENT: Allowing small numerical error
        PRINT "✓ Test 4 PASSED: tanh(100) = " + result + " approaches 1"
        INCREMENT test_passed_count
    ELSE:
        PRINT "✗ Test 4 FAILED: tanh(100) = " + result
        PRINT "  Expected: ~1.0"
        INCREMENT test_failed_count
    END IF
    
    ADDITIONAL CHECK:
        CALL tanh_activation(-100.0)
        STORE as result_neg
        
        IF abs(result_neg - (-1.0)) < 0.01:
            PRINT "  Negative saturation verified: tanh(-100) ≈ -1"
        END IF
    
    WHY THIS TEST:
        - Verifies correct saturation behavior
        - Tests numerical stability for extreme values
        - Ensures no overflow/underflow issues
        - Common in practice (saturated activations)

OPTIONAL ADDITIONAL TESTS:

TEST 5: Range verification
    FOR x from -10.0 to 10.0 in steps of 0.5:
        result = tanh_activation(x)
        VERIFY: -1 < result < 1
        VERIFY: not NaN, not Inf
    END FOR
    
TEST 6: Monotonicity
    FOR x from -5.0 to 5.0 in steps of 0.5:
        current = tanh_activation(x)
        next = tanh_activation(x + 0.5)
        VERIFY: next > current (strictly increasing)
    END FOR

TEST 7: Derivative at zero
    result = tanh_derivative(0.0)
    VERIFY: result ≈ 1.0
    COMMENT: Maximum derivative occurs at x=0

TEST 8: Comparison with sigmoid
    FOR x from -3.0 to 3.0:
        tanh_val = tanh_activation(x)
        sigmoid_val = sigmoid(x)
        
        # Verify relationship: tanh(x) = 2*sigmoid(2x) - 1
        expected = 2.0 * sigmoid(2.0 * x) - 1.0
        VERIFY: abs(tanh_val - expected) < tolerance
    END FOR

═══════════════════════════════════════════════════════════

PRINT SUMMARY:
    PRINT blank line
    PRINT "========================================="
    PRINT "TANH TEST SUMMARY"
    PRINT "========================================="
    PRINT "Tests passed: " + test_passed_count
    PRINT "Tests failed: " + test_failed_count
    PRINT "Total tests: " + (test_passed_count + test_failed_count)
    
    IF test_failed_count is 0:
        PRINT "Status: ALL TESTS PASSED ✓"
    ELSE:
        PRINT "Status: SOME TESTS FAILED ✗"
    END IF
    PRINT "========================================="
    
    RETURN test_failed_count

END FUNCTION


═══════════════════════════════════════════════════════════
FUNCTION: test_derivative_numerically
═══════════════════════════════════════════════════════════

PURPOSE:
    Verify analytical derivative matches numerical approximation
    Use finite difference method to estimate derivative
    Compare with closed-form derivative function
    Catch implementation errors in derivative formulas
    Validate across multiple input values
    
FUNCTION SIGNATURE:
    test_derivative_numerically(test_function, derivative_function, x_value)
    
INPUT PARAMETERS:
    test_function: Function pointer to the activation function
                   e.g., sigmoid, tanh_activation
                   
    derivative_function: Function pointer to the derivative
                        e.g., sigmoid_derivative, tanh_derivative
                        
    x_value: The point at which to test the derivative
            Can be any real number
            
RETURN VALUE:
    true if test passes (derivatives match)
    false if test fails (derivatives don't match)
    
VARIABLE DECLARATIONS:
    DECLARE step_size as double = 0.00001
        PURPOSE: Small h for numerical differentiation
        ALSO CALLED: epsilon, delta, dx
        RATIONALE: Must be small but not too small
        
    DECLARE tolerance as double = 0.001
        PURPOSE: Acceptable error between numerical and analytical
        RATIONALE: Numerical methods have inherent error
        
    DECLARE f_plus as double
        PURPOSE: Store f(x + h)
        
    DECLARE f_minus as double
        PURPOSE: Store f(x - h)
        
    DECLARE numerical_derivative as double
        PURPOSE: Estimated derivative using finite difference
        
    DECLARE analytical_derivative as double
        PURPOSE: Exact derivative from formula
        
    DECLARE error as double
        PURPOSE: Difference between numerical and analytical
        
═══════════════════════════════════════════════════════════
DETAILED IMPLEMENTATION
═══════════════════════════════════════════════════════════

STEP 1: Set up parameters
    PURPOSE: Define precision and tolerance for comparison
    
    SET step_size to 0.00001
        COMMENT: h in the finite difference formula
        
    WHY THIS VALUE:
        Too large (e.g., 0.1):
            - Approximation is poor
            - First-order error dominates
            - May fail tests
            
        Too small (e.g., 1e-15):
            - Floating-point precision limits
            - Catastrophic cancellation
            - Subtraction of nearly equal numbers
            - May get less accurate result!
            
        Sweet spot (1e-5 to 1e-7):
            - Balance between approximation error and roundoff error
            - Works well for most smooth functions
            - 0.00001 = 1e-5 is typically good
            
    SET tolerance to 0.001
        COMMENT: Maximum acceptable difference
        
    WHY THIS VALUE:
        Numerical differentiation is approximate
        Central difference has O(h²) error
        With h = 1e-5, error ≈ 1e-10 theoretically
        But floating-point adds error
        0.001 is conservative but reliable
        Could tighten to 0.0001 if needed

STEP 2: Calculate numerical derivative using central difference
    PURPOSE: Approximate f'(x) using finite differences
    FORMULA: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    
    SUBSTEP 2a: Evaluate function at x + h
        CALL test_function(x_value + step_size)
        STORE result as f_plus
        
        WHAT THIS IS:
            Function value slightly to the right of x
            Numerator part 1
            
    SUBSTEP 2b: Evaluate function at x - h
        CALL test_function(x_value - step_size)
        STORE result as f_minus
        
        WHAT THIS IS:
            Function value slightly to the left of x
            Numerator part 2
            
    SUBSTEP 2c: Calculate difference
        CALCULATE difference = f_plus - f_minus
        
        GEOMETRIC INTERPRETATION:
            Vertical distance between two points
            On opposite sides of x
            
    SUBSTEP 2d: Calculate divisor
        CALCULATE divisor = 2.0 * step_size
        
        WHY 2 TIMES STEP_SIZE:
            Distance between x-h and x+h is 2h
            Not just h
            Common mistake to use just h
            
    SUBSTEP 2e: Complete the division
        DIVIDE difference by divisor
        STORE as numerical_derivative
        
        FINAL FORMULA:
            numerical_derivative = (f_plus - f_minus) / (2 * step_size)
            
        MATHEMATICAL JUSTIFICATION:
            Taylor expansion:
                f(x+h) = f(x) + f'(x)h + f''(x)h²/2 + O(h³)
                f(x-h) = f(x) - f'(x)h + f''(x)h²/2 + O(h³)
                
            Subtract:
                f(x+h) - f(x-h) = 2f'(x)h + O(h³)
                
            Divide by 2h:
                [f(x+h) - f(x-h)]/(2h) = f'(x) + O(h²)
                
            Error is O(h²), much better than forward/backward difference O(h)

WHY CENTRAL DIFFERENCE VS ALTERNATIVES:

    FORWARD DIFFERENCE (less accurate):
        f'(x) ≈ [f(x+h) - f(x)] / h
        Error: O(h)
        Simpler but less precise
        
    BACKWARD DIFFERENCE (less accurate):
        f'(x) ≈ [f(x) - f(x-h)] / h
        Error: O(h)
        
    CENTRAL DIFFERENCE (most accurate):
        f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
        Error: O(h²)
        Recommended for testing
        
    HIGHER-ORDER METHODS (overkill for testing):
        Can achieve O(h⁴), O(h⁶), etc.
        Require more function evaluations
        Not necessary for validation

STEP 3: Calculate analytical derivative
    PURPOSE: Get the "correct" value from closed-form formula
    
    CALL derivative_function(x_value)
    STORE as analytical_derivative
    
    WHAT THIS IS:
        Result from the derivative formula we're testing
        e.g., sigmoid_derivative(x) returns σ(x)·(1-σ(x))
        e.g., tanh_derivative(x) returns 1 - tanh²(x)
        
    THIS IS WHAT WE'RE VALIDATING:
        Does our analytical formula match numerical approximation?
        If yes: formula is correctly implemented
        If no: bug in derivative implementation

STEP 4: Compare results
    PURPOSE: Determine if derivatives match within tolerance
    
    CALCULATE absolute difference:
        error = abs(analytical_derivative - numerical_derivative)
        
    ALTERNATIVE:
        error = fabs(analytical_derivative - numerical_derivative)
        
    WHY ABSOLUTE DIFFERENCE:
        Sign doesn't matter (both could be off by same amount)
        Magnitude of error is what we care about
        Easier to interpret than relative error
        
    IF error is less than tolerance:
        TEST PASSED
        
        PRINT "✓ Derivative correct at x = " + x_value
        PRINT "  Numerical: " + numerical_derivative
        PRINT "  Analytical: " + analytical_derivative
        PRINT "  Error: " + error
        PRINT "  (Error < " + tolerance + " ✓)"
        
        RETURN true
        
    ELSE:
        TEST FAILED
        
        PRINT "✗ Derivative mismatch at x = " + x_value
        PRINT "  Numerical: " + numerical_derivative
        PRINT "  Analytical: " + analytical_derivative
        PRINT "  Error: " + error
        PRINT "  (Error > " + tolerance + " ✗)"
        
        ADDITIONAL DIAGNOSTICS:
            PRINT "  Relative error: " + (error / analytical_derivative * 100) + "%"
            PRINT "  Please check derivative implementation"
            
        RETURN false
    END IF

═══════════════════════════════════════════════════════════
EXPLANATION OF METHOD (FOR DOCUMENTATION)
═══════════════════════════════════════════════════════════

PRINT EXPLANATION:
    PRINT "─────────────────────────────────────────"
    PRINT "NUMERICAL DIFFERENTIATION METHOD"
    PRINT "─────────────────────────────────────────"
    PRINT "The numerical derivative approximates f'(x) using:"
    PRINT "  f'(x) ≈ [f(x+h) - f(x-h)] / (2h)"
    PRINT ""
    PRINT "This is called the central difference method"
    PRINT ""
    PRINT "Advantages:"
    PRINT "  • More accurate than forward/backward difference"
    PRINT "  • Error is O(h²) instead of O(h)"
    PRINT "  • Symmetric around the point"
    PRINT ""
    PRINT "Why it works:"
    PRINT "  • Uses function values on both sides of x"
    PRINT "  • Cancels even-order error terms"
    PRINT "  • Second-order accurate approximation"
    PRINT "─────────────────────────────────────────"

GEOMETRIC INTERPRETATION:
    The numerical derivative is the slope of the secant line
    connecting points (x-h, f(x-h)) and (x+h, f(x+h))
    
    As h → 0, this secant line approaches the tangent line
    The slope of the tangent line is f'(x)
    
    DIAGRAM (conceptual):
        f(x+h) •
                 \
                  \  secant line (slope ≈ f'(x))
                   \
                    • x  ← tangent line (slope = f'(x))
                   /
                  /
                 /
        f(x-h) •

═══════════════════════════════════════════════════════════
ERROR ANALYSIS
═══════════════════════════════════════════════════════════

SOURCES OF ERROR:

    1. TRUNCATION ERROR (from finite h):
        Magnitude: O(h²) for central difference
        With h = 1e-5: error ≈ 1e-10
        Decreases as h decreases (to a point)
        
    2. ROUNDOFF ERROR (from floating-point):
        Magnitude: ≈ ε / h, where ε = machine epsilon
        For double precision: ε ≈ 2.2e-16
        With h = 1e-5: error ≈ 2.2e-11
        Increases as h decreases!
        
    3. FUNCTION EVALUATION ERROR:
        Error in computing f(x)
        Propagates through calculation
        Usually negligible for well-behaved functions
        
TOTAL ERROR:
    Total ≈ C₁h² + C₂ε/h
    
    Minimized when: h ≈ (ε)^(1/3)
    For double precision: optimal h ≈ 6e-6
    Our choice of 1e-5 is close to optimal

WHAT CAN GO WRONG:

    ISSUE 1: h too large
        Symptom: Consistently high errors
        Solution: Decrease h
        
    ISSUE 2: h too small
        Symptom: Erratic errors, sometimes very large
        Cause: Catastrophic cancellation
        Solution: Increase h
        
    ISSUE 3: Function discontinuity
        Symptom: Huge errors at certain points
        Cause: Function not differentiable there
        Solution: Expected behavior, not a bug
        
    ISSUE 4: Numerical instability in f
        Symptom: NaN or Inf results
        Cause: Overflow/underflow in function
        Solution: Check function implementation

═══════════════════════════════════════════════════════════
TESTING STRATEGY
═══════════════════════════════════════════════════════════

GOOD TEST POINTS:

    For sigmoid:
        x = 0:     Maximum derivative (0.25)
        x = ±1:    Moderate derivative (~0.20)
        x = ±2:    Smaller derivative (~0.10)
        x = ±5:    Very small derivative (~0.007)
        
    For tanh:
        x = 0:     Maximum derivative (1.0)
        x = ±1:    Moderate derivative (~0.42)
        x = ±2:    Smaller derivative (~0.07)
        x = ±3:    Very small derivative (~0.01)
        
    For ReLU (if implemented):
        x = -1:    Zero derivative
        x = 0:     Not differentiable (test separately)
        x = 1:     Derivative = 1
        
WHY MULTIPLE TEST POINTS:
    Different regions may have different behavior
    Some implementations work at x=0 but fail elsewhere
    Edge cases reveal bugs
    Comprehensive coverage increases confidence

═══════════════════════════════════════════════════════════
USAGE EXAMPLES
═══════════════════════════════════════════════════════════

EXAMPLE 1: Test sigmoid derivative at x=0
    bool passed = test_derivative_numerically(
        sigmoid,
        sigmoid_derivative,
        0.0
    );
    
    IF not passed:
        PRINT "Sigmoid derivative failed at x=0"
    END IF

EXAMPLE 2: Test across range
    FOR x from -5.0 to 5.0 step 0.5:
        bool result = test_derivative_numerically(
            tanh_activation,
            tanh_derivative,
            x
        );
        IF not result:
            PRINT "Failed at x=" + x
            BREAK
        END IF
    END FOR

EXAMPLE 3: Random testing
    FOR i from 1 to 100:
        random_x = random_double(-10.0, 10.0)
        test_derivative_numerically(
            sigmoid,
            sigmoid_derivative,
            random_x
        );
    END FOR

═══════════════════════════════════════════════════════════
ADVANCED VARIATIONS
═══════════════════════════════════════════════════════════

VARIATION 1: Adaptive step size
    Try multiple step sizes
    Pick one with smallest error
    More robust but slower
    
VARIATION 2: Richardson extrapolation
    Combine multiple approximations with different h
    Achieves higher-order accuracy
    More complex but very accurate
    
VARIATION 3: Complex step method
    Use complex arithmetic: f'(x) ≈ Im[f(x + ih)] / h
    No cancellation error
    Requires complex-valued function

END FUNCTION


═══════════════════════════════════════════════════════════
FUNCTION: test_all_derivatives
═══════════════════════════════════════════════════════════

PURPOSE:
    Comprehensive testing of all derivative functions
    Test across multiple input values
    Verify derivatives at critical points
    Generate statistical summary of results
    
VARIABLE DECLARATIONS:
    DECLARE test_points as array of double
        VALUES: [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
        PURPOSE: Representative points across input range
        
    DECLARE passed_count as integer = 0
        PURPOSE: Count total passed tests
        
    DECLARE total_tests as integer = 0
        PURPOSE: Count total tests run
        
═══════════════════════════════════════════════════════════

INITIALIZE test_points array:
    test_points[0] = -5.0   // Large negative (near saturation)
    test_points[1] = -2.0   // Moderate negative
    test_points[2] = -1.0   // Small negative
    test_points[3] = 0.0    // Origin (critical point)
    test_points[4] = 1.0    // Small positive
    test_points[5] = 2.0    // Moderate positive
    test_points[6] = 5.0    // Large positive (near saturation)
    
WHY THESE POINTS:
    Cover full range of behaviors
    Include critical point (x=0)
    Include saturation regions (|x| > 3)
    Include symmetric pairs (±1, ±2, ±5)
    Reasonable for typical neural network activations

PRINT header:
    PRINT "========================================="
    PRINT "COMPREHENSIVE DERIVATIVE TESTING"
    PRINT "========================================="
    PRINT blank line

PART 1: Test Sigmoid Derivatives:
    
    PRINT "─────────────────────────────────────────"
    PRINT "Testing Sigmoid Derivatives:"
    PRINT "─────────────────────────────────────────"
    
    FOR each x_value in test_points:
        INCREMENT total_tests
        
        CALL test_derivative_numerically(
            sigmoid,
            sigmoid_derivative,
            x_value
        )
        STORE result as test_passed
        
        IF test_passed is true:
            INCREMENT passed_count
        ELSE:
            PRINT "  ⚠ Consider reviewing sigmoid_derivative implementation"
        END IF
        
        PRINT blank line
    END FOR

PART 2: Test Tanh Derivatives:
    
    PRINT "─────────────────────────────────────────"
    PRINT "Testing Tanh Derivatives:"
    PRINT "─────────────────────────────────────────"
    
    FOR each x_value in test_points:
        INCREMENT total_tests
        
        CALL test_derivative_numerically(
            tanh_activation,
            tanh_derivative,
            x_value
        )
        STORE result as test_passed
        
        IF test_passed is true:
            INCREMENT passed_count
        ELSE:
            PRINT "  ⚠ Consider reviewing tanh_derivative implementation"
        END IF
        
        PRINT blank line
    END FOR

PRINT final summary:
    PRINT "========================================="
    PRINT "DERIVATIVE TESTING SUMMARY"
    PRINT "========================================="
    PRINT "Total derivative tests run: " + total_tests
    PRINT "Tests passed: " + passed_count
    PRINT "Tests failed: " + (total_tests - passed_count)
    
    CALCULATE pass_rate:
        pass_rate = (passed_count / total_tests) * 100.0
        
    PRINT "Pass rate: " + pass_rate + "%"
    PRINT blank line
    
    IF passed_count == total_tests:
        PRINT "✓✓✓ ALL DERIVATIVE TESTS PASSED ✓✓✓"
        PRINT "Derivative implementations are correct!"
    ELSE:
        failed = total_tests - passed_count
        PRINT "✗✗✗ " + failed + " TEST(S) FAILED ✗✗✗"
        PRINT "Please review derivative implementations"
        PRINT "Check the failed test output above for details"
    END IF
    
    PRINT "========================================="
    
    RETURN (total_tests - passed_count)
        COMMENT: Return number of failures

END FUNCTION

═══════════════════════════════════════════════════════════
FUNCTION: test_matrix_operations
═══════════════════════════════════════════════════════════

PURPOSE:
    Verify that matrix operations work correctly
    Test activation functions applied to matrices
    Validate matrix element-wise transformations
    Ensure matrix structure is preserved
    Check that all derivative values are positive
    
STRATEGY:
    Create test matrix with known values
    Apply various activation functions
    Verify spot checks against individual function calls
    Validate properties (e.g., derivatives > 0)
    Clean up all allocated memory
    
VARIABLE DECLARATIONS:
    DECLARE test_matrix as pointer to Matrix
    DECLARE sigmoid_result as pointer to Matrix
    DECLARE tanh_result as pointer to Matrix
    DECLARE sigmoid_deriv_result as pointer to Matrix
    DECLARE tanh_deriv_result as pointer to Matrix
    DECLARE expected_value as double
    DECLARE actual_value as double
    DECLARE difference as double
    DECLARE tolerance as double = 0.0001
    DECLARE all_positive as boolean
    DECLARE test_passed as boolean
    
═══════════════════════════════════════════════════════════
DETAILED IMPLEMENTATION
═══════════════════════════════════════════════════════════

PRINT header:
    PRINT "========================================="
    PRINT "MATRIX OPERATIONS TESTING"
    PRINT "========================================="
    PRINT blank line

STEP 1: Create test matrix
    PURPOSE: Allocate matrix with specific dimensions for testing
    
    PRINT "Step 1: Creating test matrix (3×3)..."
    
    CALL create_matrix(3, 3)
    STORE pointer in test_matrix
    
    ERROR HANDLING:
        IF test_matrix is NULL:
            PRINT "✗ ERROR: Failed to create test matrix"
            PRINT "Cannot proceed with matrix tests"
            RETURN -1
        END IF
        
    PRINT "✓ Test matrix created successfully"
    PRINT blank line
    
    WHAT WE HAVE:
        3×3 matrix allocated
        All elements initialized to 0.0
        Ready to be filled with test values

STEP 2: Fill with test values
    PURPOSE: Populate matrix with diverse values for comprehensive testing
    
    PRINT "Step 2: Populating matrix with test values..."
    
    ROW 0 (Negative values):
        SET test_matrix->data[0][0] to -2.0
            COMMENT: Large negative value
        SET test_matrix->data[0][1] to -1.0
            COMMENT: Moderate negative value
        SET test_matrix->data[0][2] to 0.0
            COMMENT: Zero (critical point)
            
    ROW 1 (Positive values):
        SET test_matrix->data[1][0] to 1.0
            COMMENT: Small positive value
        SET test_matrix->data[1][1] to 2.0
            COMMENT: Moderate positive value
        SET test_matrix->data[1][2] to 3.0
            COMMENT: Large positive value
            
    ROW 2 (Mixed values):
        SET test_matrix->data[2][0] to -0.5
            COMMENT: Small negative
        SET test_matrix->data[2][1] to 0.5
            COMMENT: Small positive
        SET test_matrix->data[2][2] to 1.5
            COMMENT: Moderate positive
            
    PRINT "Test matrix contents:"
    CALL print_matrix(test_matrix, "Test Matrix")
    
    MATRIX VISUALIZATION:
        [-2.0  -1.0   0.0]
        [ 1.0   2.0   3.0]
        [-0.5   0.5   1.5]
        
    WHY THESE VALUES:
        Cover negative, zero, and positive ranges
        Include values near saturation (±2, ±3)
        Include zero (maximum derivative point)
        Include symmetric pairs for property testing
        Diverse enough to catch edge cases

STEP 3: Test sigmoid on matrix
    PURPOSE: Verify apply_sigmoid_to_matrix works correctly
    
    PRINT "─────────────────────────────────────────"
    PRINT "Step 3: Testing sigmoid matrix operation..."
    PRINT "─────────────────────────────────────────"
    
    CALL apply_sigmoid_to_matrix(test_matrix)
    STORE returned pointer in sigmoid_result
    
    ERROR HANDLING:
        IF sigmoid_result is NULL:
            PRINT "✗ ERROR: Sigmoid matrix operation failed"
            PRINT "Function returned NULL"
            free_matrix(test_matrix)
            RETURN -1
        END IF
        
    PRINT "✓ Sigmoid matrix created successfully"
    PRINT blank line
    
    PRINT "Sigmoid result matrix:"
    CALL print_matrix(sigmoid_result, "Sigmoid Result")
    
    SPOT CHECK 1: Center element (originally 0.0)
        PURPOSE: Verify sigmoid(0) = 0.5
        
        PRINT "Spot check: Center element (originally 0.0)"
        
        GET sigmoid_result->data[0][2]
        STORE as actual_value
        
        CALL sigmoid(0.0)
        STORE as expected_value
        
        CALCULATE difference = abs(actual_value - expected_value)
        
        IF difference < tolerance:
            PRINT "✓ Sigmoid matrix center element correct"
            PRINT "  Expected: " + expected_value
            PRINT "  Actual:   " + actual_value
            PRINT "  Error:    " + difference
        ELSE:
            PRINT "✗ Sigmoid matrix operation failed"
            PRINT "  Expected: " + expected_value
            PRINT "  Actual:   " + actual_value
            PRINT "  Error:    " + difference + " (exceeds tolerance)"
        END IF
        PRINT blank line
        
    SPOT CHECK 2: Another element for verification
        GET sigmoid_result->data[1][0]  // Originally 1.0
        STORE as actual_value
        
        expected_value = sigmoid(1.0)
        
        CALCULATE difference = abs(actual_value - expected_value)
        
        IF difference < tolerance:
            PRINT "✓ Additional spot check passed (element [1][0])"
        ELSE:
            PRINT "✗ Additional spot check failed (element [1][0])"
        END IF
        
    PROPERTY VERIFICATION:
        PRINT "Verifying all sigmoid values in range (0, 1)..."
        
        SET all_in_range to true
        FOR i from 0 to 2:
            FOR j from 0 to 2:
                value = sigmoid_result->data[i][j]
                IF value <= 0.0 OR value >= 1.0:
                    PRINT "✗ Value out of range at [" + i + "][" + j + "]: " + value
                    SET all_in_range to false
                END IF
            END FOR
        END FOR
        
        IF all_in_range:
            PRINT "✓ All sigmoid values in valid range"
        END IF

STEP 4: Test tanh on matrix
    PURPOSE: Verify apply_tanh_to_matrix works correctly
    
    PRINT "─────────────────────────────────────────"
    PRINT "Step 4: Testing tanh matrix operation..."
    PRINT "─────────────────────────────────────────"
    
    CALL apply_tanh_to_matrix(test_matrix)
    STORE returned pointer in tanh_result
    
    ERROR HANDLING:
        IF tanh_result is NULL:
            PRINT "✗ ERROR: Tanh matrix operation failed"
            free_matrix(test_matrix)
            free_matrix(sigmoid_result)
            RETURN -1
        END IF
        
    PRINT "✓ Tanh matrix created successfully"
    PRINT blank line
    
    PRINT "Tanh result matrix:"
    CALL print_matrix(tanh_result, "Tanh Result")
    
    SPOT CHECK: Center element (originally 0.0)
        PURPOSE: Verify tanh(0) = 0
        
        PRINT "Spot check: Center element (originally 0.0)"
        
        GET tanh_result->data[0][2]
        STORE as actual_value
        
        CALL tanh_activation(0.0)
        STORE as expected_value
        
        CALCULATE difference = abs(actual_value - expected_value)
        
        IF difference < tolerance:
            PRINT "✓ Tanh matrix center element correct"
            PRINT "  Expected: " + expected_value + " (should be ~0)"
            PRINT "  Actual:   " + actual_value
            PRINT "  Error:    " + difference
        ELSE:
            PRINT "✗ Tanh matrix operation failed"
            PRINT "  Expected: " + expected_value
            PRINT "  Actual:   " + actual_value
            PRINT "  Error:    " + difference
        END IF
        PRINT blank line
        
    PROPERTY VERIFICATION:
        PRINT "Verifying all tanh values in range (-1, 1)..."
        
        SET all_in_range to true
        FOR i from 0 to 2:
            FOR j from 0 to 2:
                value = tanh_result->data[i][j]
                IF value <= -1.0 OR value >= 1.0:
                    PRINT "✗ Value out of range at [" + i + "][" + j + "]: " + value
                    SET all_in_range to false
                END IF
            END FOR
        END FOR
        
        IF all_in_range:
            PRINT "✓ All tanh values in valid range"
        END IF
        
    ANTISYMMETRY CHECK:
        PRINT "Checking antisymmetry property..."
        
        // Element [0][0] = tanh(-2.0)
        // Element [1][1] = tanh(2.0)
        // Should be negatives of each other
        
        value1 = tanh_result->data[0][0]  // tanh(-2.0)
        value2 = tanh_result->data[1][1]  // tanh(2.0)
        sum = value1 + value2
        
        IF abs(sum) < tolerance:
            PRINT "✓ Antisymmetry verified: tanh(-2) + tanh(2) ≈ 0"
        ELSE:
            PRINT "⚠ Antisymmetry check: sum = " + sum
        END IF

STEP 5: Test derivatives on matrix
    PURPOSE: Verify derivative functions work on matrices
    
    PRINT "─────────────────────────────────────────"
    PRINT "Step 5: Testing derivative matrix operations..."
    PRINT "─────────────────────────────────────────"
    
    SUBSTEP 5a: Apply sigmoid derivative
        CALL apply_sigmoid_derivative_to_matrix(test_matrix)
        STORE in sigmoid_deriv_result
        
        IF sigmoid_deriv_result is NULL:
            PRINT "✗ ERROR: Sigmoid derivative matrix operation failed"
            GOTO cleanup
        END IF
        
        PRINT "✓ Sigmoid derivative matrix created"
        PRINT blank line
        
    SUBSTEP 5b: Apply tanh derivative
        CALL apply_tanh_derivative_to_matrix(test_matrix)
        STORE in tanh_deriv_result
        
        IF tanh_deriv_result is NULL:
            PRINT "✗ ERROR: Tanh derivative matrix operation failed"
            GOTO cleanup
        END IF
        
        PRINT "✓ Tanh derivative matrix created"
        PRINT blank line
        
    PRINT "Sigmoid derivative matrix:"
    CALL print_matrix(sigmoid_deriv_result, "Sigmoid Derivative")
    
    PRINT "Tanh derivative matrix:"
    CALL print_matrix(tanh_deriv_result, "Tanh Derivative")
    
    PROPERTY CHECK: All derivatives should be positive
        PURPOSE: Verify mathematical property (derivatives > 0)
        
        PRINT "Verifying that all derivative values are positive..."
        PRINT "(Derivatives of sigmoid and tanh are always positive)"
        PRINT blank line
        
        SET all_positive to true
        
        PRINT "Checking sigmoid derivatives..."
        FOR row from 0 to 2:
            FOR col from 0 to 2:
                value = sigmoid_deriv_result->data[row][col]
                
                IF value <= 0.0:
                    PRINT "✗ Non-positive sigmoid derivative at [" + row + "][" + col + "]: " + value
                    SET all_positive to false
                ELSE:
                    PRINT "  [" + row + "][" + col + "] = " + value + " ✓"
                END IF
            END FOR
        END FOR
        PRINT blank line
        
        PRINT "Checking tanh derivatives..."
        FOR row from 0 to 2:
            FOR col from 0 to 2:
                value = tanh_deriv_result->data[row][col]
                
                IF value <= 0.0:
                    PRINT "✗ Non-positive tanh derivative at [" + row + "][" + col + "]: " + value
                    SET all_positive to false
                ELSE:
                    PRINT "  [" + row + "][" + col + "] = " + value + " ✓"
                END IF
            END FOR
        END FOR
        PRINT blank line
        
        IF all_positive is true:
            PRINT "✓✓✓ ALL DERIVATIVE VALUES ARE POSITIVE ✓✓✓"
            PRINT "This is the expected mathematical property"
        ELSE:
            PRINT "✗✗✗ SOME DERIVATIVE VALUES ARE NON-POSITIVE ✗✗✗"
            PRINT "This indicates a bug in the derivative implementation"
        END IF
        
    RANGE CHECKS:
        PRINT blank line
        PRINT "Checking derivative value ranges..."
        
        // Sigmoid derivative should be in (0, 0.25]
        PRINT "Sigmoid derivatives (should be in range (0, 0.25]):"
        FOR row from 0 to 2:
            FOR col from 0 to 2:
                value = sigmoid_deriv_result->data[row][col]
                IF value > 0.0 AND value <= 0.25:
                    PRINT "  [" + row + "][" + col + "] = " + value + " ✓"
                ELSE:
                    PRINT "  [" + row + "][" + col + "] = " + value + " ⚠ (out of range)"
                END IF
            END FOR
        END FOR
        
        PRINT blank line
        
        // Tanh derivative should be in (0, 1.0]
        PRINT "Tanh derivatives (should be in range (0, 1.0]):"
        FOR row from 0 to 2:
            FOR col from 0 to 2:
                value = tanh_deriv_result->data[row][col]
                IF value > 0.0 AND value <= 1.0:
                    PRINT "  [" + row + "][" + col + "] = " + value + " ✓"
                ELSE:
                    PRINT "  [" + row + "][" + col + "] = " + value + " ⚠ (out of range)"
                END IF
            END FOR
        END FOR
        
    SPOT CHECK: Derivative at x=0
        PRINT blank line
        PRINT "Special check: Derivatives at x=0..."
        
        // Element [0][2] corresponds to original value 0.0
        sigmoid_deriv_at_zero = sigmoid_deriv_result->data[0][2]
        tanh_deriv_at_zero = tanh_deriv_result->data[0][2]
        
        PRINT "Sigmoid derivative at x=0: " + sigmoid_deriv_at_zero
        PRINT "  Expected: 0.25 (maximum for sigmoid)"
        IF abs(sigmoid_deriv_at_zero - 0.25) < tolerance:
            PRINT "  ✓ Correct!"
        ELSE:
            PRINT "  ✗ Incorrect"
        END IF
        
        PRINT blank line
        PRINT "Tanh derivative at x=0: " + tanh_deriv_at_zero
        PRINT "  Expected: 1.0 (maximum for tanh)"
        IF abs(tanh_deriv_at_zero - 1.0) < tolerance:
            PRINT "  ✓ Correct!"
        ELSE:
            PRINT "  ✗ Incorrect"
        END IF

STEP 6: Clean up memory
    PURPOSE: Free all allocated matrices to prevent memory leaks
    
    PRINT blank line
    PRINT "─────────────────────────────────────────"
    PRINT "Step 6: Cleaning up allocated memory..."
    PRINT "─────────────────────────────────────────"
    
    cleanup:
        PRINT "Freeing test_matrix..."
        free_matrix(test_matrix)
        PRINT "  ✓ Freed"
        
        PRINT "Freeing sigmoid_result..."
        free_matrix(sigmoid_result)
        PRINT "  ✓ Freed"
        
        PRINT "Freeing tanh_result..."
        free_matrix(tanh_result)
        PRINT "  ✓ Freed"
        
        PRINT "Freeing sigmoid_deriv_result..."
        free_matrix(sigmoid_deriv_result)
        PRINT "  ✓ Freed"
        
        PRINT "Freeing tanh_deriv_result..."
        free_matrix(tanh_deriv_result)
        PRINT "  ✓ Freed"
        
    PRINT blank line
    PRINT "✓ All memory cleaned up successfully"
    
    MEMORY LEAK CHECK:
        PRINT blank line
        PRINT "Memory leak check:"
        PRINT "  Run with valgrind to verify: valgrind --leak-check=full ./program"
        PRINT "  Should show: All heap blocks were freed -- no leaks are possible"
    
PRINT final status:
    PRINT blank line
    PRINT "========================================="
    PRINT "MATRIX OPERATION TESTS COMPLETE"
    PRINT "========================================="
    
    IF all_positive AND all tests passed:
        PRINT "Status: ✓ ALL TESTS PASSED"
        RETURN 0
    ELSE:
        PRINT "Status: ✗ SOME TESTS FAILED"
        PRINT "Review output above for details"
        RETURN 1
    END IF

END FUNCTION


═══════════════════════════════════════════════════════════
FUNCTION: main_test_runner
═══════════════════════════════════════════════════════════

PURPOSE:
    Orchestrate execution of all test suites
    Run tests in logical order
    Collect and report overall results
    Provide clear summary of test outcomes
    Entry point for comprehensive testing
    
RETURN VALUE:
    0 if all tests pass
    Non-zero if any tests fail (number of failures)
    
VARIABLE DECLARATIONS:
    DECLARE total_failures as integer = 0
    DECLARE part_failures as integer
    DECLARE start_time as time_t (optional, for timing)
    DECLARE end_time as time_t (optional, for timing)
    
═══════════════════════════════════════════════════════════

PRINT test suite header:
    PRINT "========================================="
    PRINT "╔═══════════════════════════════════════╗"
    PRINT "║  ACTIVATION FUNCTIONS TEST SUITE      ║"
    PRINT "║  Comprehensive Validation             ║"
    PRINT "╚═══════════════════════════════════════╝"
    PRINT "========================================="
    PRINT blank line
    
    PRINT "This test suite will verify:"
    PRINT "  • Sigmoid and tanh activation functions"
    PRINT "  • Derivative computations"
    PRINT "  • Matrix operations"
    PRINT "  • Numerical accuracy"
    PRINT "  • Memory management"
    PRINT blank line
    
    OPTIONAL: Record start time
        start_time = current_time()
        PRINT "Test suite started at: " + format_time(start_time)
        PRINT blank line

PART 1: Basic Function Tests
    PRINT "========================================="
    PRINT "PART 1: BASIC FUNCTION TESTS"
    PRINT "========================================="
    PRINT "Testing individual activation functions"
    PRINT "with known input/output pairs"
    PRINT "-----------------------------------------"
    PRINT blank line
    
    PRINT "Testing Sigmoid Function..."
    PRINT "-----------------------------------------"
    part_failures = test_sigmoid_basic()
    total_failures = total_failures + part_failures
    
    IF part_failures is 0:
        PRINT "✓ Sigmoid tests: PASSED"
    ELSE:
        PRINT "✗ Sigmoid tests: FAILED (" + part_failures + " failures)"
    END IF
    PRINT blank line
    
    PRINT "Testing Tanh Function..."
    PRINT "-----------------------------------------"
    part_failures = test_tanh_basic()
    total_failures = total_failures + part_failures
    
    IF part_failures is 0:
        PRINT "✓ Tanh tests: PASSED"
    ELSE:
        PRINT "✗ Tanh tests: FAILED (" + part_failures + " failures)"
    END IF
    PRINT blank line
    
    PRINT "Part 1 Summary:"
    IF total_failures is 0:
        PRINT "  ✓✓✓ All basic function tests passed!"
    ELSE:
        PRINT "  ✗✗✗ " + total_failures + " test(s) failed"
    END IF
    PRINT blank line

PART 2: Derivative Verification
    PRINT "========================================="
    PRINT "PART 2: DERIVATIVE VERIFICATION"
    PRINT "========================================="
    PRINT "Comparing analytical derivatives with"
    PRINT "numerical approximations"
    PRINT "-----------------------------------------"
    PRINT blank line
    
    part_failures = test_all_derivatives()
    total_failures = total_failures + part_failures
    
    IF part_failures is 0:
        PRINT "✓ All derivative tests: PASSED"
    ELSE:
        PRINT "✗ Derivative tests: FAILED (" + part_failures + " failures)"
    END IF
    PRINT blank line
    
    PRINT "Part 2 Summary:"
    IF part_failures is 0:
        PRINT "  ✓✓✓ All derivatives verified correct!"
    ELSE:
        PRINT "  ✗✗✗ " + part_failures + " derivative test(s) failed"
    END IF
    PRINT blank line

PART 3: Matrix Operations
    PRINT "========================================="
    PRINT "PART 3: MATRIX OPERATIONS"
    PRINT "========================================="
    PRINT "Testing activation functions applied"
    PRINT "to entire matrices"
    PRINT "-----------------------------------------"
    PRINT blank line
    
    part_failures = test_matrix_operations()
    total_failures = total_failures + part_failures
    
    IF part_failures is 0:
        PRINT "✓ Matrix operation tests: PASSED"
    ELSE:
        PRINT "✗ Matrix operation tests: FAILED"
    END IF
    PRINT blank line
    
    PRINT "Part 3 Summary:"
    IF part_failures is 0:
        PRINT "  ✓✓✓ All matrix operations work correctly!"
    ELSE:
        PRINT "  ✗✗✗ Matrix operation tests failed"
    END IF
    PRINT blank line

FINAL SUMMARY:
    OPTIONAL: Calculate elapsed time
        end_time = current_time()
        elapsed = end_time - start_time
        PRINT "Test suite completed at: " + format_time(end_time)
        PRINT "Total elapsed time: " + elapsed + " seconds"
        PRINT blank line
    
    PRINT "========================================="
    PRINT "╔═══════════════════════════════════════╗"
    PRINT "║      FINAL TEST SUITE SUMMARY         ║"
    PRINT "╚═══════════════════════════════════════╝"
    PRINT "========================================="
    PRINT blank line
    
    PRINT "Test Results:"
    PRINT "─────────────────────────────────────────"
    PRINT "Part 1 - Basic Functions:     " + status_for_part1
    PRINT "Part 2 - Derivatives:         " + status_for_part2
    PRINT "Part 3 - Matrix Operations:   " + status_for_part3
    PRINT "─────────────────────────────────────────"
    PRINT "Total failures: " + total_failures
    PRINT blank line
    
    IF total_failures is 0:
        PRINT "╔═══════════════════════════════════════╗"
        PRINT "║   ✓✓✓ ALL TESTS PASSED! ✓✓✓          ║"
        PRINT "║                                       ║"
        PRINT "║   Implementation is correct and       ║"
        PRINT "║   ready for use in neural networks    ║"
        PRINT "╚═══════════════════════════════════════╝"
        PRINT blank line
        PRINT "Next steps:"
        PRINT "  • Integrate into neural network code"
        PRINT "  • Run performance benchmarks"
        PRINT "  • Test on real datasets"
    ELSE:
        PRINT "╔═══════════════════════════════════════╗"
        PRINT "║   ✗✗✗ TESTS FAILED ✗✗✗                ║"
        PRINT "║                                       ║"
        PRINT "║   " + total_failures + " test(s) did not pass            ║"
        PRINT "║   Review output above for details     ║"
        PRINT "╚═══════════════════════════════════════╝"
        PRINT blank line
        PRINT "Debugging steps:"
        PRINT "  1. Review failed test output above"
        PRINT "  2. Check function implementations"
        PRINT "  3. Verify mathematical formulas"
        PRINT "  4. Run with debugger if needed"
        PRINT "  5. Check for numerical precision issues"
    END IF
    
    PRINT "========================================="
    
    RETURN total_failures

END FUNCTION


═══════════════════════════════════════════════════════════
HELPER FUNCTION: print_matrix
═══════════════════════════════════════════════════════════

PURPOSE:
    Display matrix contents in readable format
    Useful for debugging and verification
    Shows all elements with proper alignment
    Includes optional label
    
INPUT PARAMETERS:
    matrix: Pointer to Matrix to display
    label: String label to print before matrix (optional)
    
VARIABLE DECLARATIONS:
    DECLARE row as integer
    DECLARE col as integer
    
VALIDATION:
    IF matrix is NULL:
        PRINT "Error: Cannot print NULL matrix"
        RETURN
    END IF
    
    IF matrix->data is NULL:
        PRINT "Error: Matrix data is NULL"
        RETURN
    END IF

IMPLEMENTATION:
    IF label is provided and not empty:
        PRINT label + ":"
    END IF
    
    PRINT "Dimensions: " + matrix->number_of_rows + " × " + matrix->number_of_columns
    PRINT blank line
    
    FOR row from 0 to matrix->number_of_rows - 1:
        PRINT "  ["
        
        FOR col from 0 to matrix->number_of_columns - 1:
            PRINT matrix->data[row][col] with formatting
                FORMAT: Fixed width (e.g., 8 characters)
                FORMAT: 4 decimal places
                EXAMPLE: "  -2.0000"
                
            IF col is not the last column:
                PRINT ", "
            END IF
        END FOR
        
        PRINT "]"
    END FOR
    
    PRINT blank line

FORMATTING VARIATIONS:
    BASIC:
        printf("%8.4f", value)
        
    SCIENTIFIC:
        printf("%12.6e", value)
        
    COMPACT:
        printf("%.2f", value)

END FUNCTION


═══════════════════════════════════════════════════════════
HELPER FUNCTION: absolute_value
═══════════════════════════════════════════════════════════

PURPOSE:
    Calculate absolute value of a number
    Handle both positive and negative inputs
    Simple utility for error calculations
    
INPUT: number (double)
RETURN: Absolute value of number (double)

IMPLEMENTATION:
    IF number is less than 0:
        RETURN -number
            COMMENT: Negate to make positive
    ELSE:
        RETURN number
            COMMENT: Already positive or zero
    END IF
    
ALTERNATIVE (Using standard library):
    #include <math.h>
    RETURN fabs(number)
    
NOTES:
    Standard library version (fabs) is preferred
    Custom implementation useful for understanding
    Handles special cases (NaN, Inf) differently

END FUNCTION


═══════════════════════════════════════════════════════════
HELPER FUNCTION: compare_floats
═══════════════════════════════════════════════════════════

PURPOSE:
    Compare two floating-point numbers with tolerance
    Handle imprecision inherent in floating-point arithmetic
    Determine if two values are "close enough"
    
INPUT PARAMETERS:
    value1: First floating-point number
    value2: Second floating-point number
    tolerance: Maximum acceptable difference
    
RETURN VALUE:
    true if values are equal within tolerance
    false if values differ by more than tolerance
    
VARIABLE DECLARATIONS:
    DECLARE difference as double
    DECLARE abs_difference as double

IMPLEMENTATION:
    CALCULATE difference = value1 - value2
    
    CALL absolute_value(difference)
    STORE in abs_difference
    
    ALTERNATIVE:
        abs_difference = fabs(value1 - value2)
    
    IF abs_difference is less than or equal to tolerance:
        RETURN true
            COMMENT: Values are equal within tolerance
    ELSE:
        RETURN false
            COMMENT: Values are different
    END IF

WHY NOT EXACT EQUALITY:
    Floating-point arithmetic is imprecise
    0.1 + 0.2 may not exactly equal 0.3
    Rounding errors accumulate
    Must use tolerance-based comparison
    
TYPICAL TOLERANCE VALUES:
    Very strict: 1e-10
    Strict: 1e-6
    Normal: 1e-4 or 1e-3
    Loose: 1e-2
    
ALTERNATIVE APPROACH (Relative error):
    Calculate: |value1 - value2| / max(|value1|, |value2|)
    Compare relative error to tolerance
    Better for values of different magnitudes
    
EXAMPLE USAGE:
    IF compare_floats(result, expected, 0.0001):
        PRINT "Test passed"
    ELSE:
        PRINT "Test failed"
    END IF

END FUNCTION

*/

