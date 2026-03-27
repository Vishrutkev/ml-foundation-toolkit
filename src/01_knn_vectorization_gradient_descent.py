"""
# Machine Learning Foundations

This notebook explores classical machine learning methods and numerical computing techniques with a focus on nearest neighbors, vectorization, and optimization.

## Topics covered
- k-nearest neighbors for classification and regression
- feature scaling and distance metrics
- vectorized Euclidean distance computation
- gradient descent for solving linear systems with JAX autograd
"""


"""
## k-Nearest Neighbors

### Iris classification across different k values Show the classification accuracies for different $k$ values.
"""


# k-NN classification on Iris dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris(as_frame=True)

# Use all four features as input
X = iris.data[["sepal length (cm)", "sepal width (cm)",
               "petal length (cm)", "petal width (cm)"]]
y = iris.target

# Split data into training and test sets (preserve class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0
)

# Train and evaluate k-NN for different values of k
for k in range(1, 20):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"kNN accuracy (k={k}) is {accuracy:.2f}%")


"""
### California housing regression with KNN
"""


# k-NN regression on California Housing dataset

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Load California Housing dataset
housing = fetch_california_housing()

# Use all features and target values
X = housing.data
y = housing.target

# Split data into training and test sets (no stratify since y is continuous)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

# Train k-NN regressor with k = 10
regressor = KNeighborsRegressor(n_neighbors=10)
regressor.fit(X_train, y_train)

# Predict on test set
y_pred = regressor.predict(X_test)

# Compute and print Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(mse)


"""
### Effect of feature scaling and distance metrics
"""


# k-NN regression with feature scaling and different distance metrics

from sklearn.preprocessing import MinMaxScaler

# Scale features to [0, 1] range
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# k-NN regressor using Euclidean distance
regressor_euclidean = KNeighborsRegressor(n_neighbors=10, metric="euclidean")
regressor_euclidean.fit(X_train_scaled, y_train)
y_pred_euclidean = regressor_euclidean.predict(X_test_scaled)

mse_euclidean = mean_squared_error(y_test, y_pred_euclidean)
print(mse_euclidean)

# k-NN regressor using Manhattan distance
regressor_manhattan = KNeighborsRegressor(n_neighbors=10, metric="manhattan")
regressor_manhattan.fit(X_train_scaled, y_train)
y_pred_manhattan = regressor_manhattan.predict(X_test_scaled)

mse_manhattan = mean_squared_error(y_test, y_pred_manhattan)
print(mse_manhattan)


"""
## Vectorized Euclidean Distance Matrix

### Naive pairwise distance computation

Given $x_i \in \mathrm{R}^{n}$ for $i=1,...,m$ and $y_j \in \mathrm{R}^{n}$ for $j=1,...,k$, where:
- $\mathbf{X} = [x_1,x_2,....,x_m]$ so $\mathbf{X} \in \mathrm{R}^{n \times m}$

- $\mathbf{Y} = [y_1,y_2,....,y_k]$ so $\mathbf{Y} \in \mathrm{R}^{n \times k}$

we define $\mathbf{D} \in \mathrm{R}^{m,k}$ with each element defined as:

$d_{i,j}=\|x_i-y_j\|^2$

Complete the following implementation to complete the matrix $\mathbf{D}$
"""


import numpy as np
import time

# Define input matrices X and Y
X = np.array([[1, 2, 3, 5],
              [5, 9, 8, 6]])
Y = np.array([[2, 0, 1],
              [0, 4, 1]])

# Start timing
start_time = time.time()

# Initialize distance matrix D (m x k)
D = np.empty((X.shape[1], Y.shape[1]))

# Compute Euclidean distance using nested loops
for i in range(X.shape[1]):
    xi = X[:, i]            # i-th column of X
    for j in range(Y.shape[1]):
        yj = Y[:, j]        # j-th column of Y
        D[i, j] = np.sqrt(np.sum((xi - yj) ** 2))

# End timing
print("--- %s seconds ---" % (time.time() - start_time))
print("D = \n", D)


"""
### Deriving the vectorized distance formula
1. Show that $d_{i,j}=\sum_{l=1}^n x_{i,l}^2 + \sum_{l=1}^n y_{j,l}^2-2x_i^Ty_j$

2. Verify that the summation $\mathbf{x}_i^T  \mathbf{y}_j$ can be vectorized as the following:
$$
\sum x_i^Ty_j = \mathbf{X}^T\mathbf{Y},
$$

2. Show that each element in the row of the matrix $diag(\mathbf{X}^T\mathbf{X}) \mathbf{1}_{m,k}$ can be expressed as $\sum_{l=1}^n x_{i,l}^2$

where:
- $diag$ is a diagonal matrix (all elements are zeros except those in the diagonal)
- $\mathbf{1}_{m,k}$ is a matrix of ones of dimension $m \times k$

4. Similarly, show that each element in the columns of the matrix $\mathbf{1}_{m,k} diag(\mathbf{Y}^T\mathbf{Y})^T $ can be expressed as $\sum_{l=1}^n y_{j,l}^2$
5. Show that $\mathbf{D}=diag(\mathbf{X}^T\mathbf{X}) \mathbf{1}_{m,k}+\mathbf{1}_{m,k} diag(\mathbf{Y}^T\mathbf{Y})-2\mathbf{X}^T\mathbf{Y}$
"""


"""
*## Mathematical derivation*

1. **Show that $d_{i,j}=\sum_{l=1}^n x_{i,l}^2 + \sum_{l=1}^n y_{j,l}^2 - 2x_i^Ty_j$**

We start with the squared Euclidean distance between $x_i$ and $y_j$, for all $i,j$:

$\mathbf{d}_{i,j} = || \mathbf{x}_i - \mathbf{y}_j ||_2^2$

I know that for any vector $v$, $||v||_2^2 = v^Tv$.  
So we can write:

$d_{i,j} = (\mathbf{x}_i - \mathbf{y}_j)^T(\mathbf{x}_i - \mathbf{y}_j)$

Now expand it (similar to $(a-b)^2$):

$(\mathbf{x}_i - \mathbf{y}_j)^T(\mathbf{x}_i - \mathbf{y}_j)
= \mathbf{x}_i^T\mathbf{x}_i - \mathbf{x}_i^T\mathbf{y}_j
- \mathbf{y}_j^T\mathbf{x}_i + \mathbf{y}_j^T\mathbf{y}_j$

Since $\mathbf{x}_i^T\mathbf{y}_j$ is a scalar,
$\mathbf{x}_i^T\mathbf{y}_j = \mathbf{y}_j^T\mathbf{x}_i$. Therefore:

$d_{i,j} = \mathbf{x}_i^T\mathbf{x}_i + \mathbf{y}_j^T\mathbf{y}_j - 2\mathbf{x}_i^T\mathbf{y}_j$

Also,

$\mathbf{x}_i^T\mathbf{x}_i = \sum_{l=1}^n x_{i,l}^2$  
$\mathbf{y}_j^T\mathbf{y}_j = \sum_{l=1}^n y_{j,l}^2$

So we get:

$\boxed{
d_{i,j} = \sum_{l=1}^n x_{i,l}^2
+ \sum_{l=1}^n y_{j,l}^2
- 2\mathbf{x}_i^T\mathbf{y}_j
}$

---

2. **Verify that the summation $\mathbf{x}_i^T\mathbf{y}_j$ can be vectorized as
$\sum \mathbf{x}_i^T\mathbf{y}_j = \mathbf{X}^T\mathbf{Y}$**

Given:

$\mathbf{X} = [\mathbf{x}_1,\dots,\mathbf{x}_m] \in \mathbb{R}^{n\times m}$  
$\mathbf{Y} = [\mathbf{y}_1,\dots,\mathbf{y}_k] \in \mathbb{R}^{n\times k}$

Then:

$\mathbf{X}^T\mathbf{Y} \in \mathbb{R}^{m\times k}$

The $(i,j)$ entry of $\mathbf{X}^T\mathbf{Y}$ is:

$(\mathbf{X}^T\mathbf{Y})_{i,j} = \mathbf{x}_i^T\mathbf{y}_j$

So $\mathbf{X}^T\mathbf{Y}$ contains all dot products
$\mathbf{x}_i^T\mathbf{y}_j$ at once.

---

3. **Show that each element in the row of
$diag(\mathbf{X}^T\mathbf{X})\mathbf{1}_{m,k}$
can be expressed as $\sum_{l=1}^n x_{i,l}^2$**

First,

$\mathbf{X}^T\mathbf{X} \in \mathbb{R}^{m\times m}$

The diagonal entry is:

$(\mathbf{X}^T\mathbf{X})_{i,i} = \mathbf{x}_i^T\mathbf{x}_i
= \sum_{l=1}^n x_{i,l}^2$

So $diag(\mathbf{X}^T\mathbf{X})$ stores these values on its diagonal.

When multiplying:

$diag(\mathbf{X}^T\mathbf{X})\mathbf{1}_{m,k}$

each row $i$ is filled with $\mathbf{x}_i^T\mathbf{x}_i$, meaning every entry
in row $i$ equals $\sum_{l=1}^n x_{i,l}^2$.

---

4. **Similarly, show that each element in the columns of
$\mathbf{1}_{m,k}diag(\mathbf{Y}^T\mathbf{Y})^T$
can be expressed as $\sum_{l=1}^n y_{j,l}^2$**

Similarly,

$(\mathbf{Y}^T\mathbf{Y})_{j,j} = \mathbf{y}_j^T\mathbf{y}_j
= \sum_{l=1}^n y_{j,l}^2$

So $diag(\mathbf{Y}^T\mathbf{Y})$ stores these values.

Multiplying:

$\mathbf{1}_{m,k}diag(\mathbf{Y}^T\mathbf{Y})^T$

repeats $\mathbf{y}_j^T\mathbf{y}_j$ down column $j$, so each entry
in column $j$ equals $\sum_{l=1}^n y_{j,l}^2$.

---

5. **Show that
$\mathbf{D} = diag(\mathbf{X}^T\mathbf{X})\mathbf{1}_{m,k}
+ \mathbf{1}_{m,k}diag(\mathbf{Y}^T\mathbf{Y})
- 2\mathbf{X}^T\mathbf{Y}$**

From (1),

$d_{i,j} = \mathbf{x}_i^T\mathbf{x}_i
+ \mathbf{y}_j^T\mathbf{y}_j
- 2\mathbf{x}_i^T\mathbf{y}_j$

Writing each term in matrix form:
- $\mathbf{x}_i^T\mathbf{x}_i$ becomes $diag(\mathbf{X}^T\mathbf{X})\mathbf{1}_{m,k}$
- $\mathbf{y}_j^T\mathbf{y}_j$ becomes $\mathbf{1}_{m,k}diag(\mathbf{Y}^T\mathbf{Y})$
- $\mathbf{x}_i^T\mathbf{y}_j$ becomes $\mathbf{X}^T\mathbf{Y}$

So the full distance matrix is:

$\boxed{
\mathbf{D}
=
diag(\mathbf{X}^T\mathbf{X})\mathbf{1}_{m,k}
+
\mathbf{1}_{m,k}diag(\mathbf{Y}^T\mathbf{Y})
- 2\mathbf{X}^T\mathbf{Y}
}$
"""


"""
### Vectorized implementation

Use the result $\mathbf{D}=diag(\mathbf{X}^T\mathbf{X}) \mathbf{1}_{m,k}+\mathbf{1}_{m,k} diag(\mathbf{Y}^T\mathbf{Y})-2\mathbf{X}^T\mathbf{Y}$ to write a vectorized program for better implementation efficiency for the matrix $\mathbf{D}$.

Compare your code with the one in  in terms of running speed.
"""


import numpy as np
import time

# Input matrices (columns are data points)
X = np.array([[1,2,3,5],
              [5,9,8,6]])
Y = np.array([[2,0,1],
              [0,4,1]])

# Dimensions
n, m = X.shape
n, k = Y.shape

# Start timing
start_time = time.time()

# Vectorized computation of squared Euclidean distance matrix
D = (np.diag(np.sum(X**2, axis=0)) @ np.ones((m, k))
     + np.ones((m, k)) @ np.diag(np.sum(Y**2, axis=0))
     - 2 * (X.T @ Y))

# End timing
print("--- %s seconds ---" % (time.time() - start_time))
print(D)


"""
### Broadcasting-based optimization

Using the results from .2 and 2.2.3, optimize further in terms of running time the vectorized implementation of $\mathbf{D}$.

*Hint:* check Figure 4 in numpy documentation about [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html#broadcasting-figure-4).
"""


import numpy as np
import time

# Input matrices (columns are data points)
X = np.array([[1, 2, 3, 5],
              [5, 9, 8, 6]])
Y = np.array([[2, 0, 1],
              [0, 4, 1]])

# Dimensions
n, m = X.shape
n, k = Y.shape

# Start timing
start_time = time.time()

# Fully vectorized computation using broadcasting
D = (np.sum(X**2, axis=0)[:, None]
     + np.sum(Y**2, axis=0)[None, :]
     - 2 * (X.T @ Y))

# End timing
print("--- %s seconds ---" % (time.time() - start_time))
print(D)


"""
## Gradient Descent for Linear Systems

### Derivation of gradient descent update rule
Given the matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ and the output vector $\mathbf{y} \in \mathbb{R}^m$, in order to solve the following linear equation:
$$
\mathbf{A} \mathbf{x} = \mathbf{y}
$$
for the input vector $\mathbf{x} \in \mathbb{R}^n$, we may seek to solve the following optimization problem:
$$
\min_{\mathbf{x}}  \big\Vert \mathbf{A} \mathbf{x} - \mathbf{y} \big\Vert^2
$$

Solving this optimization problem is equivallent to solving the original linear equation. Derive a gradient decent algorithm to solve this optimization optimization.
"""


"""
*## Gradient descent derivation*

** –  : Gradient Descent for**  $\min_{\mathbf{x}} ||\mathbf{A}\mathbf{x}-\mathbf{y}||_2^2$

We want to solve the linear system $\mathbf{A}\mathbf{x}=\mathbf{y}$.  
Instead of solving it directly, we can minimize the squared error:

$\min_{\mathbf{x}} \; ||\mathbf{A}\mathbf{x}-\mathbf{y}||_2^2$

---

### **Step 1: Define the objective function**
Let the cost (loss) function be:

$J(\mathbf{x}) = ||\mathbf{A}\mathbf{x}-\mathbf{y}||_2^2$

Using the identity $||\mathbf{v}||_2^2 = \mathbf{v}^T\mathbf{v}$:

$J(\mathbf{x}) = (\mathbf{A}\mathbf{x}-\mathbf{y})^T(\mathbf{A}\mathbf{x}-\mathbf{y})$

---

### **Step 2: Expand the expression**
Expand like $(u-v)^T(u-v)=u^Tu-u^Tv-v^Tu+v^Tv$, with $u=\mathbf{A}\mathbf{x}$ and $v=\mathbf{y}$:

$J(\mathbf{x}) = (\mathbf{A}\mathbf{x})^T(\mathbf{A}\mathbf{x}) - (\mathbf{A}\mathbf{x})^T\mathbf{y} - \mathbf{y}^T(\mathbf{A}\mathbf{x}) + \mathbf{y}^T\mathbf{y}$

Now simplify each part:

- $(\mathbf{A}\mathbf{x})^T(\mathbf{A}\mathbf{x}) = \mathbf{x}^T\mathbf{A}^T\mathbf{A}\mathbf{x}$
- $(\mathbf{A}\mathbf{x})^T\mathbf{y} = \mathbf{x}^T\mathbf{A}^T\mathbf{y}$
- $\mathbf{y}^T(\mathbf{A}\mathbf{x}) = \mathbf{y}^T\mathbf{A}\mathbf{x}$

Also, $\mathbf{x}^T\mathbf{A}^T\mathbf{y}$ and $\mathbf{y}^T\mathbf{A}\mathbf{x}$ are scalars and equal, so they combine into a factor 2:

So we get:

$J(\mathbf{x}) = \mathbf{x}^T\mathbf{A}^T\mathbf{A}\mathbf{x} - 2\mathbf{y}^T\mathbf{A}\mathbf{x} + \mathbf{y}^T\mathbf{y}$

---

### **Step 3: Compute the gradient**
We differentiate w.r.t. $\mathbf{x}$ term-by-term.

Known matrix derivative facts:
- $\nabla_{\mathbf{x}}(\mathbf{x}^T\mathbf{B}\mathbf{x}) = (\mathbf{B}+\mathbf{B}^T)\mathbf{x}$
- Here $\mathbf{B}=\mathbf{A}^T\mathbf{A}$ is symmetric, so $\mathbf{B}+\mathbf{B}^T=2\mathbf{A}^T\mathbf{A}$
- $\nabla_{\mathbf{x}}(\mathbf{c}^T\mathbf{x}) = \mathbf{c}$
- $\nabla_{\mathbf{x}}(\mathbf{y}^T\mathbf{y})=0$ since it does not depend on $\mathbf{x}$

So:

$\nabla J(\mathbf{x}) = 2\mathbf{A}^T\mathbf{A}\mathbf{x} - 2\mathbf{A}^T\mathbf{y}$

Factor out $2\mathbf{A}^T$:

$\boxed{\nabla J(\mathbf{x}) = 2\mathbf{A}^T(\mathbf{A}\mathbf{x}-\mathbf{y})}$

---

### **Step 4: Gradient Descent update rule**
Gradient descent updates $\mathbf{x}$ by moving in the negative gradient direction:

$\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \alpha \nabla J(\mathbf{x}^{(t)})$

Substitute the gradient:

$\boxed{
\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - 2\alpha \mathbf{A}^T(\mathbf{A}\mathbf{x}^{(t)}-\mathbf{y})
}$

Sometimes we absorb the 2 into the learning rate, letting $\eta = 2\alpha$:

$\boxed{
\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \mathbf{A}^T(\mathbf{A}\mathbf{x}^{(t)}-\mathbf{y})
}$

---

### **Final Gradient Descent Algorithm (steps)**
1. Initialize $\mathbf{x}^{(0)}$ (often zeros).
2. For $t=0,1,2,\dots$ until convergence:
   - compute error: $\mathbf{e}^{(t)}=\mathbf{A}\mathbf{x}^{(t)}-\mathbf{y}$
   - update: $\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \mathbf{A}^T\mathbf{e}^{(t)}$
3. Stop when $||\mathbf{e}^{(t)}||$ or $||\nabla J(\mathbf{x}^{(t)})||$ becomes small.

This procedure minimizes $||\mathbf{A}\mathbf{x}-\mathbf{y}||_2^2$, which gives the solution to $\mathbf{A}\mathbf{x}=\mathbf{y}$ (or the least-squares solution if there is no exact solution).
"""


"""
### JAX autograd implementation  Use `jax.numpy` and its auto-grad function to write a gradient descent code to solve the above optimization problem for the following two cases. For each case,  print the found solution $\mathbf{x}^*$.
"""


import jax.numpy as jnp

A1 = jnp.array([[ 0.59752613, 0.6905346, -0.50891773, 0.65463308, 0.26701531, -0.56915274, 0.11808333, 0.46735838],
 [-0.89526606, 0.79715922, 0.57452342, -0.00629485, 0.44091118, -0.90772543, 0.34577912, 0.67014199],
 [ 0.93670858, 0.72254387, 0.99337376, 0.21567713, 0.05358001, 0.71163904, -0.8129734, -0.77292358],
 [-0.40845634, 0.74702203, -0.36284625, 0.92230974, 0.98837581, 0.69059759, -0.33978374, 0.59868784],
 [ 0.09642706, -0.43787392, -0.67600912, -0.45671585, 0.56035694, 0.72904483, 0.79505001, -0.29354031],
 [-0.59517708, 0.08172389, 0.38847584, 0.21164882, -0.09421962, -0.6612324, -0.7419197, -0.11201114],
 [-0.90998187, 0.10939955, -0.00644353, -0.50790748, 0.69847656, -0.35544255, -0.78111919, 0.76442594],
 [ 0.0696891, -0.43123797, 0.87752935, -0.53844923, -0.05382915, -0.77473227, 0.37893365, 0.4033205 ]])

y1 = jnp.array([[0.81025244],[0.12633403],[-0.27351843],[-0.22738992],[-0.04246509],[-0.74659567],[0.29532475],[0.29147134]])

print(A1.shape)
print(y1.shape)


import jax.numpy as jnp

A2 = jnp.array([[0.8847363,  0.20735656, 0.3773889,  0.23544965, 0.58455062, 0.60455535],
 [0.22066618, 0.10340076, 0.82985727, 0.51556355, 0.16158017, 0.01554002],
 [0.77804031, 0.53846337, 0.52636412, 0.52573696, 0.05653821, 0.9502298 ],
 [0.28905543, 0.25418091, 0.68436999, 0.36737675, 0.11333675, 0.50065588],
 [0.0923724,  0.40013578, 0.10427759, 0.88367601, 0.04567698, 0.5614461 ],
 [0.76325149, 0.05564603, 0.73700669, 0.78701047, 0.3065009,  0.81391347],
 [0.42297042, 0.06445234, 0.37385898, 0.95497206, 0.98407816, 0.28076653],
 [0.10806804, 0.76714286, 0.82931698, 0.25355806, 0.09899629, 0.47661276],
 [0.55615413, 0.32609653, 0.84413152, 0.73315836, 0.58309715, 0.84786528],
 [0.95206656, 0.1132698,  0.39265378, 0.75970375, 0.08369203, 0.65761839]])

y2 = jnp.array([[0.5806555 ], [0.47827308], [0.61024271], [0.15632305], [0.93126525], [0.91945009], [0.1717938 ],
                [0.92275104], [0.78164574], [0.71781675]])

print(A2.shape)
print(y2.shape)


# Part 3.2: Gradient Descent using JAX autograd

import jax
import jax.numpy as jnp

# Loss function: J(x) = ||Ax - y||^2
def loss(x, A, y):
    r = A @ x - y          # residual vector
    return jnp.sum(r**2)   # squared error

# Gradient of loss w.r.t. x
grad_loss = jax.grad(loss, argnums=0)

# Gradient descent solver
def solve_gd(A, y, lr=1e-3, iters=20000, tol=1e-10):
    A = jnp.asarray(A, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32).reshape(-1,)   # ensure 1D target

    x = jnp.zeros((A.shape[1],), dtype=A.dtype)          # initialize x

    for _ in range(iters):
        g = grad_loss(x, A, y)                            # compute gradient
        if jnp.linalg.norm(g) < tol:                      # convergence check
            break
        x = x - lr * g                                    # update step

    return x

# ---- Case 1 ----
x1_star = solve_gd(A1, y1, lr=0.001, iters=500000)
print("Case 1: x* =", x1_star)

# ---- Case 2 ----
x2_star = solve_gd(A2, y2, lr=0.001, iters=200000)
print("Case 2: x* =", x2_star)
