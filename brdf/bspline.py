import drjit as dr
import mitsuba as mi

@dr.syntax
def B(x, k, i, t):
    # k = degree = 3 (unchanged)
    # x = value (unchanged)
    # i = for i in len(t) - k - 1
    # t = knot vector
    result = mi.Float(0.0)
    if k == 0:
        if dr.gather(dtype=type(t), source=t, index=i) <= x:
            if x < dr.gather(dtype=type(t), source=t, index=i+1):
                result = mi.Float(1.0)
            else:
                result = mi.Float(0.0)
        else:
            result = mi.Float(0.0)
                
    else:
        if dr.gather(dtype=type(t), source=t, index=i+k) == dr.gather(dtype=type(t), source=t, index=i):
            c1 = mi.Float(0.0)
        else:
            c1 = mi.Float((x - dr.gather(dtype=type(t), source=t, index=i))/(dr.gather(dtype=type(t), source=t, index=i+k) - dr.gather(dtype=type(t), source=t, index=i)) * B(x, k-1, i, t))
        if dr.gather(dtype=type(t), source=t, index=i+k+1) == dr.gather(dtype=type(t), source=t, index=i+1):
            c2 = mi.Float(0.0)
        else:
            c2 = mi.Float((dr.gather(dtype=type(t), source=t, index=i+k+1) - x)/(dr.gather(dtype=type(t), source=t, index=i+k+1)- dr.gather(dtype=type(t), source=t, index=i+1)) * B(x, k-1, i+1, t))
        result = c1 + c2
    return result

@dr.syntax
def bspline(x, t, c, k):  
    n = len(t) - k - 1
    assert (n >= k+1) and (len(c) >= n) 
    
    return sum(dr.gather(dtype=type(c), source=c, index=i) * B(x, k, i, t) for i in range(n))
