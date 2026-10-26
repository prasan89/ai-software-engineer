package com.example.calc;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class CalculatorServiceTest {

    @Autowired
    private CalculatorService service;

    @Test
    void add_returnsSum() {
        assertEquals(5, service.add(2, 3));
    }

    @Test
    void subtract_returnsDifference() {
        assertEquals(1, service.subtract(3, 2));
    }

    @Test
    void divide_returnsQuotient() {
        assertEquals(4, service.divide(8, 2));
    }

    @Test
    void divide_byZero_throwsIllegalArgument() {
        assertThrows(IllegalArgumentException.class,
                () -> service.divide(10, 0));
    }

    /**
     * FAILING TEST: documents the expected behaviour after the bug is fixed.
     * Currently throws NullPointerException instead of IllegalArgumentException.
     */
    @Test
    void divide_nullDivisor_throwsIllegalArgument() {
        assertThrows(IllegalArgumentException.class,
                () -> service.divide(10, null),
                "divide() must throw IllegalArgumentException for null divisor, not NPE");
    }
}
