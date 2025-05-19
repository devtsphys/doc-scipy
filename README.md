# SciPy Reference Guide

## Overview
SciPy (Scientific Python) is a Python library used for scientific and technical computing. It builds on NumPy and provides additional functionality for optimization, integration, interpolation, eigenvalue problems, algebraic equations, differential equations, statistics, and many other classes of problems.

## Core Components

### Core Modules

| Module | Description |
|--------|-------------|
| `scipy.cluster` | Clustering algorithms |
| `scipy.constants` | Physical and mathematical constants |
| `scipy.fft` | Fast Fourier Transform routines |
| `scipy.integrate` | Integration and ordinary differential equation solvers |
| `scipy.interpolate` | Interpolation and smoothing splines |
| `scipy.io` | Data input and output |
| `scipy.linalg` | Linear algebra operations |
| `scipy.ndimage` | N-dimensional image processing |
| `scipy.optimize` | Optimization and root-finding algorithms |
| `scipy.signal` | Signal processing |
| `scipy.sparse` | Sparse matrices and algorithms |
| `scipy.spatial` | Spatial data structures and algorithms |
| `scipy.special` | Special functions |
| `scipy.stats` | Statistical distributions and functions |

## Detailed Function Reference

### scipy.optimize

#### Root Finding

| Function | Description | Example |
|----------|-------------|---------|
| `root` | General root finding | `from scipy.optimize import root`<br>`result = root(lambda x: x**3 - 1, 0.5)`<br>`print(result.x)  # [1.0]` |
| `fsolve` | Find roots of a function | `from scipy.optimize import fsolve`<br>`result = fsolve(lambda x: x**2 - 4, 1)`<br>`print(result)  # [2.]` |
| `newton` | Newton-Raphson method | `from scipy.optimize import newton`<br>`result = newton(lambda x: x**2 - 2, 1)`<br>`print(result)  # 1.4142135623730951` |

#### Minimization

| Function | Description | Example |
|----------|-------------|---------|
| `minimize` | Unified interface for minimization | `from scipy.optimize import minimize`<br>`result = minimize(lambda x: x**2 + 2*x + 2, 0)`<br>`print(result.x)  # [-1.]` |
| `minimize_scalar` | Scalar function minimization | `from scipy.optimize import minimize_scalar`<br>`result = minimize_scalar(lambda x: x**2 + 2*x + 2)`<br>`print(result.x)  # -1.0` |
| `differential_evolution` | Global optimization | `from scipy.optimize import differential_evolution`<br>`result = differential_evolution(lambda x: x[0]**2, [(-10, 10)])`<br>`print(result.x)  # [~0]` |

#### Curve Fitting

| Function | Description | Example |
|----------|-------------|---------|
| `curve_fit` | Non-linear least squares | `from scipy.optimize import curve_fit`<br>`def f(x, a, b): return a * x + b`<br>`params, _ = curve_fit(f, [0, 1, 2], [1, 3, 5])`<br>`print(params)  # [2. 1.]` |

### scipy.integrate

| Function | Description | Example |
|----------|-------------|---------|
| `quad` | Single integration | `from scipy.integrate import quad`<br>`result, error = quad(lambda x: x**2, 0, 1)`<br>`print(result)  # 0.33333333333333337` |
| `dblquad` | Double integration | `from scipy.integrate import dblquad`<br>`result, error = dblquad(lambda y, x: x*y, 0, 1, lambda x: 0, lambda x: 1)`<br>`print(result)  # 0.25` |
| `odeint` | ODE solver (legacy) | `from scipy.integrate import odeint`<br>`def model(y, t): return -0.5 * y`<br>`t = [0, 1, 2]`<br>`result = odeint(model, 1, t)`<br>`print(result)  # [[1.], [0.60653066], [0.36787944]]` |
| `solve_ivp` | Modern ODE solver | `from scipy.integrate import solve_ivp`<br>`def model(t, y): return [-0.5 * y[0]]`<br>`result = solve_ivp(model, [0, 2], [1])`<br>`print(result.y)  # [[1. ... 0.36787944]]` |

### scipy.interpolate

| Function | Description | Example |
|----------|-------------|---------|
| `interp1d` | 1D interpolation | `from scipy.interpolate import interp1d`<br>`x = [0, 1, 2]`<br>`y = [0, 1, 4]`<br>`f = interp1d(x, y)`<br>`print(f(1.5))  # 2.5` |
| `CubicSpline` | Cubic spline | `from scipy.interpolate import CubicSpline`<br>`x = [0, 1, 2]`<br>`y = [0, 1, 4]`<br>`cs = CubicSpline(x, y)`<br>`print(cs(1.5))  # 2.125` |
| `griddata` | Interpolate unstructured data | `from scipy.interpolate import griddata`<br>`import numpy as np`<br>`points = np.array([[0, 0], [1, 0], [0, 1]])`<br>`values = np.array([1, 2, 3])`<br>`result = griddata(points, values, (0.5, 0.5), method='linear')`<br>`print(result)  # 2.0` |

### scipy.linalg

| Function | Description | Example |
|----------|-------------|---------|
| `inv` | Matrix inverse | `from scipy.linalg import inv`<br>`import numpy as np`<br>`A = np.array([[1, 2], [3, 4]])`<br>`A_inv = inv(A)`<br>`print(A_inv)  # [[-2.   1. ] [ 1.5 -0.5]]` |
| `solve` | Solve linear system | `from scipy.linalg import solve`<br>`A = np.array([[1, 2], [3, 4]])`<br>`b = np.array([5, 6])`<br>`x = solve(A, b)`<br>`print(x)  # [-4.   4.5]` |
| `eig` | Eigenvalues/eigenvectors | `from scipy.linalg import eig`<br>`A = np.array([[1, 2], [3, 4]])`<br>`eigenvals, eigenvecs = eig(A)`<br>`print(eigenvals)  # [-0.37228132  5.37228132]` |
| `svd` | Singular value decomposition | `from scipy.linalg import svd`<br>`A = np.array([[1, 2], [3, 4]])`<br>`U, s, Vh = svd(A)`<br>`print(s)  # [5.4649857  0.36596619]` |
| `cholesky` | Cholesky decomposition | `from scipy.linalg import cholesky`<br>`A = np.array([[4, 2], [2, 5]])`<br>`L = cholesky(A, lower=True)`<br>`print(L)  # [[2. 0.] [1. 2.]]` |

### scipy.stats

#### Distributions

| Distribution | Description | Example |
|--------------|-------------|---------|
| `norm` | Normal distribution | `from scipy.stats import norm`<br>`print(norm.pdf(0))  # 0.3989422804014327`<br>`print(norm.cdf(1.96))  # 0.9750021048517795` |
| `uniform` | Uniform distribution | `from scipy.stats import uniform`<br>`print(uniform.rvs(size=5))  # [Array of 5 random samples]` |
| `t` | Student's t-distribution | `from scipy.stats import t`<br>`print(t.ppf(0.975, df=10))  # 2.2281388519649385` |

#### Statistical Tests

| Function | Description | Example |
|----------|-------------|---------|
| `ttest_ind` | t-test for independent samples | `from scipy.stats import ttest_ind`<br>`a = [1, 2, 3, 4, 5]`<br>`b = [2, 3, 4, 5, 6]`<br>`t_stat, p_val = ttest_ind(a, b)`<br>`print(p_val)  # 0.1372785765621116` |
| `pearsonr` | Pearson correlation | `from scipy.stats import pearsonr`<br>`x = [1, 2, 3, 4, 5]`<br>`y = [2, 3, 5, 8, 9]`<br>`r, p = pearsonr(x, y)`<br>`print(r)  # 0.9684134781954834` |
| `linregress` | Linear regression | `from scipy.stats import linregress`<br>`x = [1, 2, 3, 4, 5]`<br>`y = [2, 4, 5, 4, 6]`<br>`result = linregress(x, y)`<br>`print(result.slope)  # 0.8` |

### scipy.signal

| Function | Description | Example |
|----------|-------------|---------|
| `convolve` | Convolution | `from scipy.signal import convolve`<br>`x = [1, 2, 3]`<br>`y = [0, 1, 0.5]`<br>`result = convolve(x, y, mode='full')`<br>`print(result)  # [0.  1.  2.5 3.5 1.5]` |
| `filtfilt` | Zero-phase filtering | `from scipy.signal import filtfilt, butter`<br>`b, a = butter(4, 0.2)`<br>`x = np.random.randn(100)`<br>`y = filtfilt(b, a, x)`<br> |
| `butter` | Butterworth filter design | `from scipy.signal import butter`<br>`b, a = butter(4, 0.2)`<br>`print(b[:3])  # [0.0048, 0.0193, 0.0290]` |
| `spectrogram` | Compute spectrogram | `from scipy.signal import spectrogram`<br>`fs = 10e3`<br>`x = np.random.randn(8000)`<br>`f, t, Sxx = spectrogram(x, fs)`<br> |

### scipy.fft

| Function | Description | Example |
|----------|-------------|---------|
| `fft` | Fast Fourier Transform | `from scipy.fft import fft`<br>`x = np.array([1, 2, 1, 0])`<br>`y = fft(x)`<br>`print(abs(y))  # [4. 1.41421356 0. 1.41421356]` |
| `ifft` | Inverse FFT | `from scipy.fft import ifft`<br>`y = np.array([4, 1+1j, 0, 1-1j])`<br>`x = ifft(y)`<br>`print(x)  # [1.+0.j 2.+0.j 1.+0.j 0.+0.j]` |
| `fft2` | 2D FFT | `from scipy.fft import fft2`<br>`x = np.ones((2, 2))`<br>`y = fft2(x)`<br>`print(y)  # [[4.+0.j 0.+0.j] [0.+0.j 0.+0.j]]` |

### scipy.sparse

| Function/Class | Description | Example |
|----------------|-------------|---------|
| `csr_matrix` | Compressed Sparse Row matrix | `from scipy.sparse import csr_matrix`<br>`data = np.array([1, 2, 3])`<br>`row = np.array([0, 0, 1])`<br>`col = np.array([0, 2, 1])`<br>`mat = csr_matrix((data, (row, col)), shape=(2, 3))`<br>`print(mat.toarray())  # [[1 0 2] [0 3 0]]` |
| `linalg.spsolve` | Solve sparse system | `from scipy.sparse import csr_matrix`<br>`from scipy.sparse.linalg import spsolve`<br>`A = csr_matrix([[3, 2], [1, 1]])`<br>`b = np.array([5, 2])`<br>`x = spsolve(A, b)`<br>`print(x)  # [1. 1.]` |

### scipy.spatial

| Function | Description | Example |
|----------|-------------|---------|
| `distance.pdist` | Pairwise distances | `from scipy.spatial import distance`<br>`x = np.array([[0, 0], [0, 1], [1, 0]])`<br>`d = distance.pdist(x, 'euclidean')`<br>`print(d)  # [1. 1. 1.41421356]` |
| `KDTree` | KD-Tree for quick nearest-neighbor lookup | `from scipy.spatial import KDTree`<br>`x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])`<br>`tree = KDTree(x)`<br>`dist, idx = tree.query(np.array([0.5, 0.5]), k=2)`<br>`print(idx)  # [0 2]` |
| `Voronoi` | Voronoi diagrams | `from scipy.spatial import Voronoi`<br>`points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])`<br>`vor = Voronoi(points)`<br>`print(vor.vertices)  # [[0.5 0.5]]` |

### scipy.ndimage

| Function | Description | Example |
|----------|-------------|---------|
| `gaussian_filter` | Gaussian smoothing | `from scipy.ndimage import gaussian_filter`<br>`x = np.random.randn(10, 10)`<br>`y = gaussian_filter(x, sigma=1)`<br> |
| `rotate` | Rotate array | `from scipy.ndimage import rotate`<br>`x = np.eye(4)`<br>`y = rotate(x, 45)`<br> |
| `label` | Label features in image | `from scipy.ndimage import label`<br>`x = np.array([[0, 1, 1], [0, 0, 1], [1, 1, 0]])`<br>`labeled, num_features = label(x)`<br>`print(num_features)  # 2` |

### scipy.constants

| Constant | Value | Example |
|----------|-------|---------|
| `pi` | 3.141592653589793 | `from scipy import constants`<br>`print(constants.pi)  # 3.141592653589793` |
| `c` | Speed of light (m/s) | `print(constants.c)  # 299792458.0` |
| `G` | Gravitational constant | `print(constants.G)  # 6.67430e-11` |
| `h` | Planck constant | `print(constants.h)  # 6.62607015e-34` |

## Advanced Techniques

### Optimization Problems

```python
# Global optimization
from scipy.optimize import differential_evolution

# Define objective function
def objective(x):
    return x[0]**2 + x[1]**2

# Define bounds
bounds = [(-5, 5), (-5, 5)]

# Solve the minimization problem
result = differential_evolution(objective, bounds)
print(result.x)  # Should be close to [0, 0]
```

### ODE Solving

```python
# Solving a system of ODEs
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Define the ODE system (Lotka-Volterra predator-prey model)
def lotka_volterra(t, z, a, b, c, d):
    x, y = z
    dx_dt = a*x - b*x*y
    dy_dt = -c*y + d*x*y
    return [dx_dt, dy_dt]

# Parameters
a, b, c, d = 1.0, 0.1, 1.5, 0.075
z0 = [10, 5]  # Initial population
t_span = (0, 40)  # Time span
t_eval = np.linspace(0, 40, 1000)  # Points to evaluate

# Solve
sol = solve_ivp(
    lambda t, z: lotka_volterra(t, z, a, b, c, d),
    t_span, z0, method='RK45', t_eval=t_eval
)

# Extract results
t = sol.t
x, y = sol.y
```

### FFT for Signal Processing

```python
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Create a signal with two sine waves
sample_rate = 1000  # Hz
duration = 1  # second
t = np.linspace(0, duration, sample_rate, endpoint=False)
freq1, freq2 = 50, 120  # Hz
signal = np.sin(2*np.pi*freq1*t) + 0.5*np.sin(2*np.pi*freq2*t)

# Add some noise
signal += 0.2 * np.random.randn(len(t))

# Compute FFT
yf = fft(signal)
xf = fftfreq(sample_rate, 1/sample_rate)[:sample_rate//2]
yplot = 2.0/sample_rate * np.abs(yf[:sample_rate//2])

# Plot the spectrum
plt.figure(figsize=(10, 6))
plt.plot(xf, yplot)
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 200)
```

### Working with Sparse Matrices

```python
from scipy.sparse import csr_matrix, diags
import numpy as np

# Create a sparse matrix from diagonals
n = 1000
diagonals = [np.ones(n), -2*np.ones(n), np.ones(n)]
offsets = [-1, 0, 1]
A = diags(diagonals, offsets, shape=(n, n), format='csr')

# Create a random sparse matrix
density = 0.01  # 1% of elements are non-zero
B = np.random.random((n, n))
B[B > density] = 0
B_sparse = csr_matrix(B)

# Sparse matrix operations
C = A @ B_sparse  # Matrix multiplication
```

### Spatial Data Processing

```python
from scipy.spatial import Delaunay, ConvexHull
import numpy as np
import matplotlib.pyplot as plt

# Generate random points
points = np.random.rand(30, 2)

# Compute Delaunay triangulation
tri = Delaunay(points)

# Compute the convex hull
hull = ConvexHull(points)

# Plot
plt.figure(figsize=(10, 5))
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r-', lw=2)
```

## Tips and Best Practices

1. **Import Conventions**: Use named imports for clarity:
   ```python
   from scipy import optimize, stats, signal
   ```

2. **Computational Efficiency**:
   - Use sparse matrices for large, sparse problems
   - Vectorize operations instead of loops
   - Use appropriate methods for large-scale problems (e.g., `scipy.sparse.linalg` vs `scipy.linalg`)

3. **Error Handling**:
   - Check return values of optimization functions like `result.success`
   - Use `try/except` blocks for numerical routines that might fail

4. **Integration with NumPy and Matplotlib**:
   - SciPy works seamlessly with NumPy arrays
   - Results are easily visualized with Matplotlib

5. **Version Compatibility**:
   - SciPy functions and behavior can change between versions
   - Check documentation for your specific version