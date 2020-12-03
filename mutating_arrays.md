
# Working around Mutating Arrays

We use Julia's Zygote package to perform automatic code differentiation. But it does not work if you are differentiating code that is ‘mutating arrays’ somewhere. Here we present how to go around this problem in a couple of ways. Using a sample problem.

## The Problem
We have a dynamical system that takes an input $x$, and has a state $a$. 
Here both $x$ and $a$ are $n$-vectors.
The update rule for $a$ at time $t$ is. 
$$
a_t = A a_{t-1} + x_t
$$

$A$ a $m$-diagonal matrix of size $n\times n$, constructed as:


```julia
function getA(m, n)
    A = zeros(Int, n, n)
    for d in 0:m-1
        for j in 1:n-d 
            A[j, j+d] = m-d
        end
    end
    return A
end;
```


```julia
getA(1, 4)
```


```julia
getA(2, 4)
```


```julia
getA(3, 4)
```

After $t$ time, the output is just $y = a_t[1]$.
That is the first element of the state vector. 


```julia
function y(x, m; showstate=false)
    n, T = size(x)
    A = getA(m, n)
    aaa_ = zeros(eltype(x), n, T)      # Store all the states a in a matrix 
    aaa_[:, 1] = x[:, 1]
    for t in 2:T
        aaa_[:, t] = A * aaa_[:, t-1] + x[:, t]
    end
    if showstate 
        print("a... = ")
        show(stdout, "text/plain", aaa_)
    end
    aaa_[1, T]
end;
```

We represent the input $\{x_1, x_2, \cdots, x_t\}$ as the $n\times t$ matrix $X$.
Now we want to find the gradient $\nabla_X y$, which is also an $n\times t$ matrix.


```julia
# Generate data
N, T = 9, 11
X = reshape((1:N*T).%T, N, T)
```


```julia
y(X, 2, showstate=true)
```

You can see here that the result is the first element of the final state. That is the top right element of the $a$'s matrix above. This value depends on the previous states, and through them on the input matrix $X$. Now we need to differentiate $y = a_t[1]$, with respect to (each element of) the matrix $X$.


```julia
using Zygote: gradient, @ignore
M = 2
dy(x) = gradient(X_ -> y(X_, M), x)[1]
dy(X)
```

This is leading to an error as expected as we are assigning to existing matrices. 

In function `y`, we are assigning to `aaa_[:, t] = ...`

In function `getA`, we are assigning to `A[j, j+d] = ...`

## Ignoring
Now we see that the recurrence realation matrix $A$ is a constant with respect to $y$. So we can just ask the `Zygote` to ignore the consturction of `A`. 

While we are at it, let us also optimize our function to not store intermediate states. And avoid assigning to  `aaa_[:, t] = ...`


```julia
function y1(x, m)
    n, T = size(x)
    A = @ignore getA(m, n)     # Using @ignore macro
    a = x[:, 1]
    for t in 2:T
        a = A*a       # We are not assigning to part of an array/matrix anymore 
        a += x[:, t]
    end
    a[1]
end
dy1(x) = gradient(X_ -> y1(X_, M), x)[1];
```


```julia
dy1(X)
```

Yay! It works!

## An Inefficient Solution
As you can see above we removed the line `aaa_[:, t] = ...` and replaced it with the lines

```
a = A*a     
a += x[:, t]
```

So that we are not mutating a part of an array/matrix anywhere. 

This works but is inefficient as for large sizes $n$, multipication by $A$ can be achieved in $O(n)$ time vs. $O(n^2)$ time as our current direct matrix multiplication does! So we make the function $y$ more efficient by avoiding the matrix $A$. Instead, we write code to update state without the multipication.


```julia
function updatestate(a, m)
    aa = zero(a)
    n = length(a)
    for i in 1:n
        for d in 0:min(m-1, n-i)
            aa[i] += (m-d) * a[i+d] 
        end
    end
    return aa
end

function y2(x::Matrix{F}, m::Integer) where F
    n, t = size(x)
    a = zeros(F, n)
    for i in 1:t
        a = updatestate(a, m)
        a += x[:, i]
    end
    a[1]
end
@show y1(X, M)
@show y2(X, M)    ;
```


```julia
using BenchmarkTools
Xr = rand(1000, 10)
@btime y1(Xr, M)
@btime y2(Xr, M);
```

Our new implementation works and (for $n=1000$) is nearly **150 times faster**! So the optimization is worth it. 

Now let us try to differentiate it. We can already guess that `updatestate` is going to error as it is mutating the vector `aa`!


```julia
dy2(x) = gradient(X_ -> y2(X_, M), x)[1];
dy2(X)
```

But we can not just `@ignore updatestate` because this operation does affect the final gradient as opposed to building the constant `A` in `y1`. 

## The WorkAround
We need to work around the 'mutating array' operation of `updatestate`. Since `Zygote` is unable to differentiate `updatestate` automatically, we do it ourself!
This is simple as the derivative of $b = Aa$ is just $A^T$. So the chain rule is $\bar{a} = A^T\bar{b}$. Where $\bar{b}$ is the derivative of the final answer (here $y$) with respect to $b$, and similarly for $a$.

We write the chain rule as a reverse rule using the `ChainRulesCore` package.

The `rrule` function takes the same arguments as the original function `updatestate`, i.e. `a`, `m`. It calculates the actual ‘forward’ value which is just `update(a, m)`, and along with it returns the ‘pullback’ function. 

For a given `a` and `m`, the pullback function takes the derivative w.r.to the new state $\bar{a}$ and returns the derivative w.r.to the old state $\bar{b}$, thus implementing the chainrule in reverse. 

(For some unknown reason, I am having to redefine `updatestate`. This is not needed if you are not working at a REPL.)


```julia
import ChainRulesCore: rrule, DoesNotExist, NO_FIELDS

function updatestate(a, m)    # Same definition as above.
    aa = zero(a)
    n = length(a)
    for i in 1:n
        for d in 0:min(m-1, n-i)
            aa[i] += (m-d) * a[i+d] 
        end
    end
    return aa
end

function rrule(::typeof(updatestate), a, m)
    function update_pullback(ā)
        b̄ = zero(ā)
        for i in 1:length(a)
            for d in 0:m-1
                if i-d > 0
                    b̄[i] += (m-d) * ā[i-d]
                end
            end
        end
        return NO_FIELDS, b̄, DoesNotExist()
    end
    return updatestate(a, m), update_pullback
end;
```

`updatestate` does not have any ‘fields/parameters’, hence `NO_FIELDS`. It also is not differentiable w.r.to `m`, hence the `DoesNotExist()`.

Now let us try to diffentiate our efficient implementation `y2`.


```julia
dy2m(x) = gradient(X_ -> y2(X_, M), x)[1];
dy2m(X)
```


```julia
@btime dy1(Xr)
@btime dy2m(Xr);
```

Yay! It works and our efficient method is again nearly **170 times faster** than the multiplication-by-a-matrix version, making it worthwhile to mutate arrays.
