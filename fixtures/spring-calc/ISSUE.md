# Issue #1 — CalculatorService.divide() throws NullPointerException for null divisor

**Repository:** fixtures/spring-calc  
**Component:** `com.example.calc.CalculatorService`  
**Method:** `divide(Integer dividend, Integer divisor)`  

## Description

When `divide()` is called with a `null` divisor, it throws `java.lang.NullPointerException`
instead of a descriptive `IllegalArgumentException`.

## Steps to reproduce

```java
CalculatorService service = new CalculatorService();
service.divide(10, null); // throws NPE, expected IllegalArgumentException
```

## Expected behaviour

`divide()` must throw `IllegalArgumentException("Divisor must not be null")` when the
divisor argument is `null`, consistent with how it handles a zero divisor.

## Actual behaviour

`java.lang.NullPointerException: Cannot invoke "java.lang.Integer.intValue()" because
"divisor" is null`  at `CalculatorService.divide(CalculatorService.java:34)`

## Acceptance criteria

- [ ] `divide(10, null)` throws `IllegalArgumentException` with a non-empty message
- [ ] `divide(10, 0)` still throws `IllegalArgumentException`
- [ ] `divide(8, 2)` still returns `4`
- [ ] Existing passing tests remain green
