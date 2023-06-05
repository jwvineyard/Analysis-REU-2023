import numpy as np
from numpy.polynomial import polynomial as poly


# generate array of k polynomials of degree n
# (most definitely not optimized)
def gen_poly(n, k):
    re_coeff = np.random.uniform(-1,1,(k,n))
    im_coeff = np.random.uniform(-1,1,(k,n))*1j
    coeff =  re_coeff + im_coeff

    poly_arr = np.zeros(k)
    i = 0
    while(i<k):
        poly_arr[i] = poly.Polynomial(coeff[i])
        i += 1
    return poly_arr


# find samp norm of polynomial
# F is set of points
# p is polynomial
def norm_samp(p,F):
    sup = 0
    for f in F:
        abs = abs(p.__call__(f))
        if abs > sup: sup = abs
    return sup


# TODO: write method to find sup norm of polynomial on unit circle


# TODO: write function to calculate K(n,F)/K(n,N)
def K(n,F):
    #dies
    return None