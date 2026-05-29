import time, sys
import numpy as np
import matplotlib.pyplot as plt
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import jax.scipy.linalg as jla
from jax import lax
from jax import jit
import pandas as pd
# import utils
# from manifold_sampler import ManifoldSampler

eps = 1e-1
d = 1
M = 2 * jnp.eye(d) # mass matrix
L = jnp.linalg.cholesky(M)
I = jnp.eye(M.shape[0], dtype=M.dtype)
M_inv = jla.cho_solve((L, True), I)

s = 1
Sigma = s * jnp.eye(d) # covariance matrix for target Gaussian
L = jnp.linalg.cholesky(Sigma)
I = jnp.eye(Sigma.shape[0], dtype=Sigma.dtype)
Sigma_inv = jla.cho_solve((L, True), I)

seed = 1234
key = jax.random.PRNGKey(seed)


# logγ(x) = -U(x), assuming γ(x) = exp(-U(x))
log_potential = jit(lambda x : -0.5 * (x @ Sigma_inv @ x))
# log exp(- 0.5 * p^T M^-1 p)
log_proposal = jit(lambda p : -0.5 * p @ M_inv @ p)

Hamiltonian = jit(lambda x, p : -(log_potential(x) + log_proposal(p)))

# ∇logγ(x) = -∇U(x)
@jit
def grad_lp(x):
    return jax.grad(log_potential)(x)

def kick(s):
    x, p = s
    return x, p + (eps/2) * grad_lp(x)

def drift(s):
    x, p = s
    return x + eps * (M_inv @ p), p

def flip(s):
    x, p = s
    return x, -p

def leapfrog_step(s):
    x, p = s
    x, p = kick((x, p))
    x, p = drift((x, p))
    x, p = kick((x, p))
    return x, p

@jit(static_argnames=["L"])
def leapfrog(s0, L):
    x0, p0 = s0
    xs = jnp.empty(shape=(L+1, d))
    ps = jnp.empty(shape=(L+1, d))
    xs = xs.at[0].set(x0)
    ps = ps.at[0].set(p0)

    Hs = jnp.empty(shape=(L+1, ))
    Hs = Hs.at[0].set(Hamiltonian(x0, p0))

    init_carry = (xs, ps, Hs, 1)

    def body(carry, _):
        xs, ps, Hs, i = carry
        x, p = xs[i-1], ps[i-1]
        new_x, new_p = leapfrog_step((x, p))
        new_H = Hamiltonian(new_x, new_p)
        xs = xs.at[i].set(new_x)
        ps = ps.at[i].set(new_p)
        Hs = Hs.at[i].set(new_H)
        new_carry = (xs, ps, Hs, i+1)
        return new_carry, _

    final_carry, _ = lax.scan(body, init_carry, xs=None, length=L)
    final_xs, final_ps, final_Hs, _ = final_carry
    return final_xs, final_ps, final_Hs


key, subkey1, subkey2 = jax.random.split(key, 3)
x0 = jax.random.multivariate_normal(subkey1, mean=jnp.zeros(d), cov=Sigma)
p0 = jax.random.multivariate_normal(subkey2, mean=jnp.zeros(d), cov=M)

# print(leapfrog_step((x0, p0)))
# print(leapfrog_step(flip(leapfrog_step((x0, p0)))))
# print(leapfrog((x0, p0), 5))

L = 50
xs, ps, hs = leapfrog((x0, p0), L)
# Exact Hamiltonian contour
lb, ub = np.min((xs, ps)), np.max((xs, ps))
x_grid = np.linspace(lb-1, ub+1, 300)
p_grid = np.linspace(lb-1, ub+1, 300)
X, P = np.meshgrid(x_grid, p_grid) # X, P has shape (300, 300)
# Z expects both inputs has shape (300, 300, d), so that each input[i,j] has shape (d, )
Z = jax.vmap(jax.vmap(Hamiltonian))(X[..., None], P[..., None]) # [..., None] = [:, :, None]: add one more dimension


# Exact solution for comparison
# t = np.linspace(0, eps * L, 500)
t = eps * np.arange(L + 1)
Sigma_inv_scalar = Sigma_inv.flatten()[0]
M_inv_scalar = M_inv.flatten()[0]
omega = jnp.sqrt(Sigma_inv_scalar * M_inv_scalar)
x_exact = x0 * np.cos(omega * t) + (M_inv_scalar * p0 / omega) * np.sin(omega * t)
p_exact = p0 * np.cos(omega * t) - (Sigma_inv_scalar * x0 / omega) * np.sin(omega * t)


plt.figure(figsize=(7, 6))
plt.contour(X, P, Z, levels=20)
plt.plot(x_exact, p_exact, color="red", alpha=0.5, label="Exact Hamiltonian flow")
plt.plot(xs, ps, color="blue", marker="o", markersize=1, alpha=0.3, label="Leapfrog")
plt.scatter([x0], [p0], s=80, label="Start")
plt.xlabel("x")
plt.ylabel("p")
plt.title("Exact flow vs leapfrog in phase space")
plt.axis("equal")
plt.legend()
plt.show()


print("leapfrog solutions to Hamiltonian equations: ", xs.reshape(-1))
print("Analytic solution to Hamiltonian equations: ", x_exact)
print("difference of leapfrog and analytic solutions: ", jnp.abs(xs.reshape(-1) - x_exact))

# Hamiltonian error over time
plt.figure(figsize=(7, 4))
plt.plot(np.arange(L + 1), hs - hs[0], marker="o", markersize=3)
plt.axhline(0, linestyle="--")
plt.xlabel("Leapfrog step")
plt.ylabel(r"$H(q_n,p_n)-H(q_0,p_0)$")
plt.title("Hamiltonian error along leapfrog trajectory")
plt.show()



# # manifold sampler on Hamiltonian
# tol = 1e-8
# nmax = 50

# def q_numeric(z):
#     x = z[:d]
#     p = z[d:]
#     return Hamiltonian(x, p)
# q = jit(lambda z : jnp.array([q_numeric(z)]))

# @jit
# def grad_q(z):
#     g = jax.grad(q_numeric)
#     return jnp.reshape(g(z), (-1,1))


# @jax.jit
# def log_f(z):
#     Gz = grad_q(z)  # Jacobian of q, shape: n x m  = 2 x 1 (dim_ambient, dim_constraint) = (2,1)
#     return -jnp.log(jnp.linalg.norm(Gz))


# sampler = ManifoldSampler(None, log_f, grad_q, eps, nmax, tol, "newton")
# q_level = Hamiltonian(x0, p0)
# start = jnp.concatenate([x0, p0])
# key, samples, accepts, tangents = utils.run_chain(sampler, key, start, 1, q, q_level, grad_q, mode="fixed")
# print("samples: ", samples)
