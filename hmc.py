import time, sys
import numpy as np
import matplotlib.pyplot as plt
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["JAX_PLATFORMS"] = "cpu"
from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import jax.scipy.linalg as jla
from jax import lax
from jax import jit
import pandas as pd

eps = 0.1
d = 2
M = 2 * jnp.eye(d) # mass matrix
L = jnp.linalg.cholesky(M)
I = jnp.eye(M.shape[0], dtype=M.dtype)
M_inv = jla.cho_solve((L, True), I)
seed = 1234
key = jax.random.PRNGKey(seed)


# logγ(x) = -U(x), assuming γ(x) = exp(-U(x))
log_potential = jit(lambda x : -0.5 * jnp.linalg.norm(x)**2)
# log exp(- 0.5 * p^T M^-1 p)
log_proposal = jit(lambda p : -0.5 * p @ M_inv @ p)


Hamiltionian = lambda x, p : -(log_potential(x) + log_proposal(p))

# ∇logγ(x) = -∇U(x)
@jit
def grad_lp(x):
    return jax.grad(log_potential)(x)

def kick(x, p):
    return x, p + (eps/2) * grad_lp(x)

def drift(x, p):
    return x + eps * p, p

def flip(x, p):
    return x, -p

def leapfrog(x, p):
    x, p = kick(x, p)
    x, p = drift(x, p)
    x, p = kick(x, p)
    return x, p


key, subkey1, subkey2 = jax.random.split(key, 3)
x0 = jax.random.multivariate_normal(subkey1, mean=jnp.zeros(d), cov=jnp.eye(d))
p0 = jax.random.multivariate_normal(subkey2, mean=jnp.zeros(d), cov=M)
print("x: ", x0, "p: ", p0)
print(leapfrog(x0, p0))
