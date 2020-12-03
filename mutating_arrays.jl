
function getA(m, n)
    A = zeros(Int, n, n)
    for d in 0:m-1
        for j in 1:n-d 
            A[j, j+d] = m-d
        end
    end
    return A
end;

getA(1, 4)

getA(2, 4)

getA(3, 4)

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

# Generate data
N, T = 9, 11
X = reshape((1:N*T).%T, N, T)

y(X, 2, showstate=true)

using Zygote: gradient, @ignore
M = 2
dy(x) = gradient(X_ -> y(X_, M), x)[1]
dy(X)

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

dy1(X)

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

using BenchmarkTools
Xr = rand(1000, 10)
@btime y1(Xr, M)
@btime y2(Xr, M);

dy2(x) = gradient(X_ -> y2(X_, M), x)[1];
dy2(X)

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

dy2m(x) = gradient(X_ -> y2(X_, M), x)[1];
dy2m(X)

@btime dy1(Xr)
@btime dy2m(Xr);
