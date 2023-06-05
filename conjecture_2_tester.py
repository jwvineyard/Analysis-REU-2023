import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as poly
from scipy.stats import norm


# generate array of k polynomials of degree n
# (most definitely not optimized)
def gen_polyspace(n, k):
    # Generate random real and imaginary coefficients in the box from -1 to 1 and sum them
    re_coeff = np.random.uniform(-1,1,(k,n))
    im_coeff = np.random.uniform(-1,1,(k,n))*1j
    coeff =  re_coeff + im_coeff

    # Turns coefficient arrays to polynomial array
    poly_arr = np.empty(k,dtype=poly.Polynomial)
    i = 0
    while(i<k):
        poly_arr[i] = poly.Polynomial(coeff[i])
        i += 1
    return poly_arr


# find samp norm of polynomial
# F is set of points
# p is polynomial
def norm_samp(p,F):
    # Evaluate the polynomial at the points in F
    values = poly.polyval(F, p.coef)

    # Compute the magnitudes of these values
    magnitudes = np.abs(values)

    # Return the maximum magnitude
    return np.max(magnitudes)


# p is polynomial, k is number of points to evaluate p at.
def norm_sup(p,k):
    theta = np.linspace(0, 2 * np.pi, k)
    z = np.exp(1j * theta)  # 1j represents the imaginary unit in Python.

    # Evaluate the polynomial at these points
    values = poly.polyval(z, p.coef)

    # Compute the magnitudes of these values
    magnitudes = np.abs(values)

    # Return the maximum magnitude
    return np.max(magnitudes)

def K(n, F, pspace_size = 1,sup_eval_size = 10):
    polyspace = gen_polyspace(n,pspace_size)

    # Find suprememum of sup/samp given polynomial space P
    sup = 0
    for p in polyspace:
        a = norm_sup(p,sup_eval_size)/norm_samp(p,F)
        if a > sup: sup = a
    return sup

def roots_of_unity(N):
    """
    Returns a list of n-th roots of unity.
    """
    roots = []
    for k in range(N):
        angle = 2 * math.pi * k / N
        real_part = np.cos(angle)
        imag_part = np.sin(angle)
        root = complex(real_part, imag_part)
        roots.append(root)
    return roots

def K_graph(n, F, pspace_size=1, sup_eval_size=10, num_bins=1000):
    data = []
    polyspace = gen_polyspace(n,pspace_size)

    for p in polyspace:
        a = norm_sup(p,sup_eval_size)/norm_samp(p,F)
        data.append(a)

    hist, bins = np.histogram(data, bins=num_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fit a Gaussian distribution to the histogram data
    (mu, sigma) = norm.fit(data)

    # Create the fitted Gaussian curve
    pdf = norm.pdf(bin_centers, mu, sigma)

    # Plot the histogram
    plt.hist(data, bins=num_bins, color='b', label='Histogram')

    # Plot the fitted Gaussian curve
    plt.plot(bin_centers, pdf, 'r-', label='Gaussian Fit')

    print(data)

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.legend()

    # Display the histogram
    plt.show()
    return bins 

K_graph(6, roots_of_unity(7), pspace_size=2000, num_bins=5000)
