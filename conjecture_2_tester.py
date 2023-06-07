import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as poly
from scipy.stats import norm


# generate array of k polynomials of degree n
# (most definitely not optimized)
def gen_polyspace(n, k, coeff_bound=1):
    # Generate random real and imaginary coefficients in the box from -1 to 1 and sum them
    #re_coeff = np.random.uniform(-coeff_bound,coeff_bound,(k,n+1))
    #im_coeff = np.random.uniform(-coeff_bound,coeff_bound,(k,n+1))*1j
    #coeff =  re_coeff + im_coeff
    coeff = np.random.rand(k,n+1) * np.exp(2*np.pi*1j*np.random.rand(k,n+1))
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

def K(n, F, pspace_size = 1,sup_eval_size = 1000, return_argmax = False):
    polyspace = gen_polyspace(n,pspace_size)

    # Find suprememum of sup/samp given polynomial space P
    sup = 0
    argmax = None
    for p in polyspace:
        a = norm_sup(p,sup_eval_size)/norm_samp(p,F)
        if a > sup:
            sup = a
            argmax = p
    if return_argmax: return sup, argmax
    else: return sup

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

def K_graph(n, F, pspace_size=1, sup_eval_size=10, num_bins=1000, coeff_bound=1):
    polyspace = []
    for j in range(1, n):
        # generate psapce_size polynomials of degree j<n
        polyspace.extend(gen_polyspace(j, pspace_size, coeff_bound=coeff_bound))

    # evaluate the sup norm over the sampling norm for all polynomials
    data = [norm_sup(p,sup_eval_size)/norm_samp(p,F) for p in polyspace]

    hist, bins = np.histogram(data, bins=num_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    

    # Plot the histogram
    plt.hist(data, bins=num_bins, color='b', label='Histogram')

    # # Fit a Gaussian distribution to the histogram data
    # (mu, sigma) = norm.fit(data)

    # # Create the fitted Gaussian curve
    # pdf = norm.pdf(bin_centers, mu, sigma)
    # plt.plot(bin_centers, pdf, 'r-', label='Gaussian Fit')

    print(data)

    # Add labels and title
    plt.xlabel(r'$\frac{||p||}{||p||_N}$')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.legend()

    # Display the histogram
    plt.show()
    return bins

# Plots complex polynomial on the unit circle, with independent variable being theta
def plot_polynomial(p, num_points=1000,sample_points=None):
    # Generate angles for points on the unit circle
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Generate points on the unit circle
    unit_circle_points = np.exp(1j * theta)  # Euler's formula

    # Calculate polynomial values
    polynomial_values = p(unit_circle_points)

    # Plot real and imaginary parts separately
    plt.figure(figsize=(12, 6))

    # Real part plot
    plt.subplot(1, 2, 1)
    plt.plot(theta, polynomial_values.real)
    plt.title('Real Part')
    plt.xlabel('Theta')
    plt.ylabel('Value')

    # Imaginary part plot
    plt.subplot(1, 2, 2)
    plt.plot(theta, polynomial_values.imag)
    plt.title('Imaginary Part')
    plt.xlabel('Theta')
    plt.ylabel('Value')

    if sample_points is not None:
        for sp in sample_points:
            plt.subplot(1, 2, 1)
            plt.axvline(x=sp, linestyle='dotted', color='r')  # Real part plot
            plt.subplot(1, 2, 2)
            plt.axvline(x=sp, linestyle='dotted', color='r')  # Imaginary part plot

    plt.tight_layout()
    plt.show()


def plot_polynomial_3d(p, num_points=1000):
    # Generate angles for points on the unit circle
    theta = np.linspace(0, 2*np.pi, num_points)

    # Generate points on the unit circle
    unit_circle_points = np.exp(1j*theta) # Euler's formula

    # Calculate polynomial values
    polynomial_values = p(unit_circle_points)

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot real and imaginary parts on the same graph
    ax.plot(theta, polynomial_values.real, polynomial_values.imag)
    ax.set_title('3D Plot of Complex Polynomial')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Real Part')
    ax.set_zlabel('Imaginary Part')

    plt.show()


def plot_polynomial_abs(p, num_points=1000, sample_points=None):
    # Generate angles for points on the unit circle
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Generate points on the unit circle
    unit_circle_points = np.exp(1j * theta)  # Euler's formula

    # Calculate polynomial values
    polynomial_values = p(unit_circle_points)

    # Calculate absolute value of polynomial values
    absolute_values = np.abs(polynomial_values)

    # Plot absolute value
    plt.figure(figsize=(8, 6))
    plt.plot(theta, absolute_values)
    plt.title('Absolute Value of Complex Polynomial')
    plt.xlabel('Theta')
    plt.ylabel('Absolute Value')

    # If sample_points are provided, plot vertical lines at these points
    if sample_points is not None:
        for sp in sample_points:
            plt.axvline(x=sp, linestyle='dotted', color='r')

    plt.show()
