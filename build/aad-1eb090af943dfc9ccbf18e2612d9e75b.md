# Automatic Differentiation

I've wanted to write something about this technique for a while (and by doing so, I also deepen my understanding of it) to automatically compute derivatives in code. In the financial world, derivatives manifest in many ways, such as sensitivities to market variables, so being able to compute them accurately and quickly is of great value for risk management.

## Differentiation Techniques

Over the years, several differentiation techniques have emerged. In this document, I will discuss three:

- Manual differentiation
- Numerical differentiation
- Automatic differentiation

Since we will be reviewing different techniques that aim for the same objective, it would be good to have an idea of their differences and the implications of using one over the other. For each technique, I will comment on the following points, which, in my experience, have been relevant when choosing a method:

| Integration | Performance and Accuracy | Compatibility |
| --- | --- | --- |
| Here, we analyze what it takes to integrate the technique into an existing codebase. We assess how *plug-and-play* it is when using it. | In practical terms, how fast the implementation runs and how good the computed derivative is. | Similar, but not identical to the *scope* of integration, this evaluates how well this tool works with other tools in our project. |

## Before We Begin: A Reference Function

For all the following examples, we will use the following function as a reference:

$f(x) = 2x^2+2\cos{x}$

Where we know that the analytical derivative is:

$f'(x) = 4x-2\sin{x}$

A reference implementation in Rust would be:

```rust
fn f(x: f64) -> f64 {
    let g = 2.0 * x * x;
    let z = 2.0 * x.cos();
    g + z
}
```

## Manual Differentiation

The first option is the most straightforward: manually differentiating our functions. If we consider the function from the beginning, it would suffice to implement a specialized function to compute the derivative.

```rust
fn der_f(x: f64) -> f64 {
    let der_g = 4.0 * x;
    let der_z = -2.0 * x.sin();
    der_g + der_z
}
```

There's no magic here—it simply requires the developer to manually differentiate every line involving a mathematical operation. Looking at it through the *scopes* I initially outlined, we could expect:

| Integration | Performance and Accuracy | Compatibility |
| --- | --- | --- |
| Very low integration capability. We cannot differentiate code without clear and exactly differentiable mathematical functions. This technique is only feasible as demonstrated in the previous case (1). | The performance, assuming a good implementation, should be the best possible since we allow the compiler to optimize our code. Even though we need to run an extra function to compute the derivative, any method below also has additional computational costs. | Since we are implementing a separate function, we should not encounter major compatibility issues with other libraries. |

(1) Some specialized compilers generate code that represents the derivatives of our functions—such as *clad* for the *clang* compiler—though I haven't had the opportunity to try it.

## Numerical Differentiation

This is the traditional derivative method commonly used in finance. It essentially involves shocking the inputs of our function and observing how the outputs change. More precisely, we evaluate the derivative using a finite difference approximation:

$f'(x) = \frac{f(x+h)-f(x)}{h}$

In terms of implementation, we would have something like:

```rust
fn der_f(x: f64, h: f64) -> f64 {
    (f(x + h) - f(x)) / h
}
```

In this technique, we leverage the limit as $h \rightarrow 0$. The precision we seek depends on the discretization method we use. Generally, we can state:

| Integration | Performance and Accuracy | Compatibility |
| --- | --- | --- |
| Easier integration. If we have a working implementation, we just adjust the inputs, rerun the function, and we're done. | The performance, relative to our initial implementation, literally doubles the computation time for each input we evaluate. In some applications, this is infeasible (very long computations). The accuracy depends on the choice of delta but is never exact. | There should be no major compatibility issues with other tools since this simply involves running our code multiple times. |

## Automatic Differentiation

Automatic differentiation (AD) is a technique widely used in machine learning (ML) libraries that allows us to compute derivatives as the program runs. Depending on the implementation, it can have low integration costs. Let's explore it in more detail.

### How Does AD Work in Practice?

Automatic differentiation is based on the **chain rule**, a fundamental calculus concept. If we recall, this rule states that we can reconstruct the derivative of a function as a composition of the derivatives of its sub-functions. More precisely, if we have a function $h(x) = f(g(x))$, its derivative is:

$h'(x) = f'(g(x))g'(x)
$

For example, the function $f(x) = 2x^2+2\cos{x}$, whose derivative is $f'(x) = 4x+2\sin{x}$, can be rewritten as:

$h(x) = x^2, \quad h'(x) = 2x
$

$u(x) = \cos{x}, \quad u'(x) = -\sin{x}
$

$g(x) = g(h(x)) = 2h(x) = 2x^2, \quad g'(x) = g'(h(x))h'(x) = 2 \cdot 2x = 4x
$

$z(x) = z(u(x)) = 2\cos{x}, \quad z'(x) = z'(u(x))u'(x) = -2\sin{x}
$

$f(x) = g(x) + z(x) = 2x^2+2\cos{x}, \quad f'(x) = g'(x)+z'(x) = 4x-2\sin{x}
$

This allows us to create a computation tree, where the lower nodes represent basic functions like $h(x)$ and $u(x)$, for which we have exact derivatives.

### Conclusions

We've explored various techniques for computing derivatives, highlighting the importance of choosing the right one for a given problem. In finance, calculating sensitivities to market and business variables is crucial. Without an efficient method, some solutions (e.g., XVA) become impractical.

```{seealso}
For further reading:
- **The Art of Differentiating Computer Programs / Uwe Naumann**
- **Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation / Andreas Griewank & Andrea Walther**
```
