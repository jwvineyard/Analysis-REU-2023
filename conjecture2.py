import numpy as np
import numpy.polynomial.polynomial as poly

def genPolys(n, k, bound = 1):
    coeff = np.random.uniform(-bound, bound, (k, n + 1)) + np.random.uniform(-bound, bound, (k, n + 1)) * 1j

    return map(lambda i : poly.Polynomial(coeff[i]), range(k))

def samplingNorm(p, F):
    return np.max(np.abs(poly.polyval(F, p.coef)))

def supNorm(p, k):
    mesh = np.exp(np.linspace(0, 2 * np.pi, k) * 1j)

    return np.max(np.abs(poly.polyval(mesh, p.coef)))

def K(n, F, sampleSize = 1, supMeshSize = 10):
    polySamples = genPolys(n, sampleSize)

    return np.max(list(map(lambda p : supNorm(p, supMeshSize) / samplingNorm(p, F), polySamples)))

F = [1, 1j, -1, -1j]

for n in range(len(F)):
    print(K(n, F, 10000, 1000))
