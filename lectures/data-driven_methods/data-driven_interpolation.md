#### Data-Driven Rational Interpolation

Consider a SISO LTI system

$$
\begin{equation}\tag{$\Sigma$}
  \begin{aligned}
    E \dot{x}(t) & = A x(t) + B u(t), \\
    y(t) & = C x(t).
  \end{aligned}
\end{equation}
$$

where $ E, A \in \mathbb{R}^{n \times n}, B \in \mathbb{R}^{n}, C \in \mathbb{R}^{1 \times n} $

The corresponding transfer function is

$$
H(s) = C {\left(s E - A\right)}^{-1} B.
$$

$H$ is a matrix-valued rational function.


***Goal:*** For some Laplace-variables/frequencies of interest $s_1,\ldots,s_N$ compute a rational function $\hat{H}$ such that 

$$
    H(s_i) = \hat{H}(s_i)
$$


If we know $E, A, B, C$ then we use rational Krylov subspaces and do projection: `LTIBHIReductor`.

If we don't know $E, A, B, C$ but can evaluate $H(s)$ for arbitrary $s$ then we can use `TFBHIReductor`.


### Data-Driven Setting

If we have neither access to $E, A, B, C$ nor $H(s)$ ***but a dataset***

$$
\begin{equation}
  \begin{aligned}
    H(s_1),\ldots,H(s_N) \\
    s_1,\ldots,s_N \in \mathbb{C}
  \end{aligned}
\end{equation}
$$

then we are in the ***data-driven*** setting.

Data can come from
- Real-world measurements
- Data from previous simulations


### Loewner Interpolation Framework

1.    Split the data into left and right partition
    $$
    \begin{aligned}
    \{ s_1,\ldots,s_N \} \quad &\rightarrow \quad \{ \lambda_1,\ldots,\lambda_k \} \cup \{ \mu_1,\ldots,\mu_{N-k} \} \\
    H(s_1),\ldots,H(s_N) \quad &\rightarrow \quad \{ H(\lambda_1),\ldots,H(\lambda_k) \} \cup \{ H(\mu_1),\ldots,H(\mu_{N-k}) \} \\
    \end{aligned}
    $$
2.    Compute the Loewner matrix pencil
    $$
    \mathbb{L} = \begin{bmatrix}
    \frac{H(\lambda_1) - H(\mu_1)}{\lambda_1 - \mu_1} & \cdots & \frac{H(\lambda_1) - H(\mu_{N-k})}{\lambda_1 - \mu_{N-k}} \\ 
        \vdots & & \vdots \\
    \frac{H(\lambda_k) - H(\mu_1)}{\lambda_k - \mu_1} & \cdots & \frac{H(\lambda_k) - H(\mu_{N-k})}{\lambda_k - \mu_{N-k}}
    \end{bmatrix}, \quad \mathbb{L}_s = \begin{bmatrix}
    \frac{\lambda_1 H(\lambda_1) - \mu_1 H(\mu_1)}{\lambda_1 - \mu_1} & \cdots & \frac{\lambda_1 H(\lambda_1) - \mu_{N-k} H(\mu_{N-k})}{\lambda_1 - \mu_{N-k}} \\ 
        \vdots & & \vdots \\
    \frac{\lambda_k H(\lambda_k) - \mu_1 H(\mu_1)}{\lambda_k - \mu_1} & \cdots & \frac{\lambda_k H(\lambda_k) - \mu_{N-k} H(\mu_{N-k})}{\lambda_k - \mu_{N-k}}
    \end{bmatrix} \in \mathbb{R}^{k \times (N - k)}
    $$
3.    Compute the rank-revealing SVDs
    $$
    Y_1 \Sigma_1 X_1^* = \begin{bmatrix} \mathbb{L} & \mathbb{L}_s \end{bmatrix}, Y_2 \Sigma_2 X_2^* = \begin{bmatrix} \mathbb{L} \\ \mathbb{L}_s \end{bmatrix}
    $$
4.    Form the LTI ROM via
    $$
    \begin{aligned}
        \hat{E} &:= -Y_1^\top \mathbb{L}_s X_2 \\
        \hat{A} &:= -Y_1^\top \mathbb{L} X_2 \\
        \hat{B} &:= Y_1^\top \begin{bmatrix} H(\mu_1) & \cdots & H(\mu_{N-k}) \end{bmatrix}^\top \\
        \hat{C} &:= \begin{bmatrix} H(\lambda_1) & \cdots & H(\lambda_k) \end{bmatrix} X_2 \\
    \end{aligned}
    $$


The reduced order model will be of order $r$ where
$$
    r = \operatorname{rank}\left(\begin{bmatrix} \mathbb{L} & \mathbb{L}_s \end{bmatrix}\right).
$$
The reduced order transfer function $\hat{H}$ will interpolate $H$ for all samples $\{ s_1,\ldots,s_N \}$ 


There are many practical (typically problem or objective dependent) questions:

- How large should the left vs right partition be?
- How do we pick which samples go into the left vs right partition?
- How do we choose the tolerance for the numerical rank in the rank-revealing SVD?
- How do we do it for the MIMO case (i.e., interpolation of matrix-valued functions)?

```python
import scipy.io as spio
from pymor.models.iosys import LTIModel

mats = spio.loadmat('beam.mat')
fom = LTIModel.from_matrices(mats['A'], mats['B'].todense(), mats['C'].todense())
```

```python
import numpy as np

s = np.logspace(-3, 3, 50)
Hs = np.array([fom.transfer_function.eval_tf(ss) for ss in 1j*s])
```

```python
from pymor.reductors.loewner import LoewnerReductor

loewner = LoewnerReductor(1j*s,Hs)
```

It holds $H(\overline{s}) = \overline{H(s)}$.

If we have data for $H(s)$ we can artifically increase our data set by adding complex conjugates. This allows for computing ***real realizations***.

```python
rom = loewner.reduce()
```

```python
rom
```

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(constrained_layout=True)
fom.transfer_function.mag_plot(np.logspace(-3, 3, 200), ax=ax, label='FOM')
rom.transfer_function.mag_plot(np.logspace(-3, 3, 200), ax=ax, label='ROM', linestyle='dashed')
ax.legend()
```

### Exercise

- Change the `partitioning` attribute of `loewner` to `'half-half`' using `loewner.with_`.
- Compute an order $5$ reduced model with the `loewner` reductor (i.e., pass $r=5$ to the `reduce` method).
- Plot the solution.
- Repeat the previous steps using `partitioning = 'even-odd'`.


```python
loewner = loewner.with_(...)
...
```

### Rational Approximation
The Loewner framework will yield a rational approximation if
- The truncation tolerance for the SVD of Loewner matrices is too small
- The partitioning size $k$ is too small


### AAA Idea
1.    Construct $\hat{H}$ such that it interpolates $\lambda_1,\ldots,\lambda_k$ and approximates other data in a least squares sense.
    $$
    \{ s_1,\ldots,s_N \} \quad \rightarrow \quad \begin{cases} \text{Interpolate: } & \{ \lambda_1,\ldots,\lambda_k \} \\ \text{LS Fit:} & \{ \mu_1,\ldots,\mu_{N-k} \} \end{cases}
    $$
2.    Start with $k=1$ interpolation point and successively increase the interpolation set via greedy selection:
    $$
      \lambda_{k+1} = \operatorname{arg max} \lVert H(\mu_j) - \hat{H}(\mu_j) \rVert
    $$
3.    Finish once error over training set is low enough.

```python
from pymor.reductors.aaa import PAAAReductor

aaa = PAAAReductor(1j*s,Hs)
```

```python
rom = aaa.reduce(tol=1e-3)
```

```python
rom
```

```python
fig, ax = plt.subplots(constrained_layout=True)
fom.transfer_function.mag_plot(np.logspace(-3, 3, 200), ax=ax, label='FOM')
rom.mag_plot(np.logspace(-3, 3, 200), ax=ax, label='ROM', linestyle='dashed')
```

```python

```
