package com.example.calc;

import org.springframework.stereotype.Service;

/**
 * Calculator service.
 *
 * BUG (issue #1): divide() does not guard against a null divisor.
 * When divisor is null, a NullPointerException is thrown rather than
 * an IllegalArgumentException with a clear message.
 *
 * Root cause: the null check is missing before the unboxing call
 * Integer.intValue() that happens implicitly in the division expression.
 */
@Service
public class CalculatorService {

    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }

    /**
     * Divides {@code dividend} by {@code divisor}.
     *
     * @throws IllegalArgumentException if divisor is zero
     * @throws IllegalArgumentException if divisor is null   ← MISSING: currently throws NPE
     */
    public int divide(Integer dividend, Integer divisor) {
        // BUG: missing null check on divisor — throws NullPointerException when divisor == null
        if (divisor == 0) {
            throw new IllegalArgumentException("Divisor must not be zero");
        }
        return dividend / divisor;
    }
}
