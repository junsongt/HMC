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
import solver

jax.config.update("jax_enable_x64", True)


eps = 1e-1
d = 1
M = 5 * jnp.eye(d)  # mass matrix
L = jnp.linalg.cholesky(M)
I = jnp.eye(M.shape[0], dtype=M.dtype)
M_inv = jla.cho_solve((L, True), I)

s = 2
Sigma = s * jnp.eye(d)  # covariance matrix for target Gaussian
L = jnp.linalg.cholesky(Sigma)
I = jnp.eye(Sigma.shape[0], dtype=Sigma.dtype)
Sigma_inv = jla.cho_solve((L, True), I)

seed = 1234
key = jax.random.PRNGKey(seed)


# logγ(x) = -U(x), assuming γ(x) = exp(-U(x))
log_potential = jit(lambda x: -0.5 * (x @ Sigma_inv @ x))
# log exp(- 0.5 * p^T M^-1 p)
log_proposal = jit(lambda p: -0.5 * p @ M_inv @ p)

Hamiltonian = jit(lambda x, p: -(log_potential(x) + log_proposal(p)))


# ∇logγ(x) = -∇U(x)
@jit
def grad_lp(x):
    return jax.grad(log_potential)(x)


def kick(s):
    x, p = s
    return x, p + (eps / 2) * grad_lp(x)


def drift(s):
    x, p = s
    return x + eps * (M_inv @ p), p


def flip(s):
    x, p = s
    return x, -p


def leapfrog(s):
    x, p = s
    x, p = kick((x, p))
    x, p = drift((x, p))
    x, p = kick((x, p))
    return x, p


@jit(static_argnames=["L"])
def leapfrog_integrate(s0, L):
    x0, p0 = s0
    xs = jnp.empty(shape=(L + 1, d))
    ps = jnp.empty(shape=(L + 1, d))
    xs = xs.at[0].set(x0)
    ps = ps.at[0].set(p0)

    Hs = jnp.empty(shape=(L + 1,))
    Hs = Hs.at[0].set(Hamiltonian(x0, p0))

    init_carry = (xs, ps, Hs, 1)

    def body(carry, _):
        xs, ps, Hs, i = carry
        x, p = xs[i - 1], ps[i - 1]
        new_x, new_p = leapfrog((x, p))
        new_H = Hamiltonian(new_x, new_p)
        xs = xs.at[i].set(new_x)
        ps = ps.at[i].set(new_p)
        Hs = Hs.at[i].set(new_H)
        new_carry = (xs, ps, Hs, i + 1)
        return new_carry, _

    final_carry, _ = lax.scan(body, init_carry, xs=None, length=L)
    final_xs, final_ps, final_Hs, _ = final_carry
    return final_xs, final_ps, final_Hs


key, subkey1, subkey2 = jax.random.split(key, 3)
x0 = jax.random.multivariate_normal(subkey1, mean=jnp.zeros(d), cov=Sigma)
p0 = jax.random.multivariate_normal(subkey2, mean=jnp.zeros(d), cov=M)

print("initial state: ", jnp.concatenate([x0, p0]))

# print(leapfrog((x0, p0)))
# print(leapfrog(flip(leapfrog((x0, p0)))))
# print(leapfrog((x0, p0), 5))

L = 10
xs, ps, hs = leapfrog_integrate((x0, p0), L)
# Exact Hamiltonian contour
lb, ub = np.min((xs, ps)), np.max((xs, ps))
x_grid = np.linspace(lb - eps, ub + eps, 300)
p_grid = np.linspace(lb - eps, ub + eps, 300)
X, P = np.meshgrid(x_grid, p_grid)  # X, P has shape (300, 300)
# Z expects both inputs has shape (300, 300, d), so that each input[i,j] has shape (d, )
Z = jax.vmap(jax.vmap(Hamiltonian))(X[..., None], P[..., None])  # [..., None] = [:, :, None]: add one more dimension


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

leapfrog_solutions = jnp.hstack([xs, ps])
analytic_solutions = jnp.hstack([x_exact.reshape(-1, 1), p_exact.reshape(-1, 1)])


# print("leapfrog solutions to Hamiltonian equations: ", leapfrog_solutions)  # same as jnp.concatenate([xs, ps], axis=1)
# print("Analytic solution to Hamiltonian equations: ", analytic_solutions)
# print(
#     "norms of difference of leapfrog and analytic solutions: ",
#     jax.vmap(jnp.linalg.norm)(jnp.abs(leapfrog_solutions - analytic_solutions)),
# )


# manifold sampler on Hamiltonian
tol = 1e-8
nmax = 200


def q_numeric(z):
    x = z[:d]
    p = z[d:]
    return Hamiltonian(x, p)


q = jit(lambda z: jnp.array([q_numeric(z)]))


@jit
def grad_q(z):
    g = jax.grad(q_numeric)
    return jnp.reshape(g(z), (-1, 1))


@jax.jit
def log_f(z):
    Gz = grad_q(z)  # Jacobian of q, shape: n x m  = (dim_ambient, dim_constraint)
    return -jnp.log(jnp.linalg.norm(Gz))


q_level = Hamiltonian(x0, p0)


def project_check(proj_obj, constraints) -> bool:
    if constraints is None:
        return proj_obj["flag"]

    def false_fn():
        return jnp.array(False)

    def true_fn():
        y = proj_obj["pt"]
        return jnp.all(constraints(y) >= 0)

    return lax.cond(proj_obj["flag"], true_fn, false_fn)


@jit(static_argnames=["q", "gradient", "nmax", "tol"])
def rs_project(z, tz, q, q_level, gradient, nmax, tol, constraints=None):
    Gz = gradient(z)
    proj_obj = solver.newton_solve(z + tz, q, q_level, gradient, Gz, nmax, tol)

    def proj_success_fn():
        return proj_obj["pt"], True

    def proj_failure_fn():
        return z, False

    return lax.cond(project_check(proj_obj, constraints), proj_success_fn, proj_failure_fn)


z0 = jnp.concatenate([x0, p0])
tz0 = eps * jnp.concatenate([M_inv @ p0, grad_lp(x0)])
print("tangent vector proposed:  ", tz0)
z1, flag = rs_project(z0, tz0, q, q_level, grad_q, nmax, tol)
print("at time eps, analytic solution: ", analytic_solutions[1])
print("at time eps, leapfrog solution: ", leapfrog_solutions[1])
print("at time eps, rs projected solution: ", z1, flag)

x1 = z1[:d]
p1 = z1[d:]
print(jnp.concatenate([x1, p1]))
tz1 = eps * jnp.concatenate([M_inv @ p1, grad_lp(x1)])
print("tangent proposed: ", tz1)
print(tz1 @ grad_q(z1))
z2, flag = rs_project(z1, tz1, q, q_level, grad_q, nmax, tol)
print("at time 2eps, rs projected solution: ", z2, flag)


@jit(static_argnames=["L"])
def rs_integrate(s0, L):
    x0, p0 = s0
    xs = jnp.empty(shape=(L + 1, d))
    ps = jnp.empty(shape=(L + 1, d))
    xs = xs.at[0].set(x0)
    ps = ps.at[0].set(p0)

    Hs = jnp.empty(shape=(L + 1,))
    Hs = Hs.at[0].set(Hamiltonian(x0, p0))

    init_carry = (xs, ps, Hs, 1)

    def body(carry, _):
        xs, ps, Hs, i = carry
        x, p = xs[i - 1], ps[i - 1]
        z = jnp.concatenate([x, p])
        tz = eps * jnp.concatenate([M_inv @ p, grad_lp(x)])
        new_state, new_flag = rs_project(z, tz, q, q_level, grad_q, nmax, tol)
        new_x = new_state[:d]
        new_p = new_state[d:]
        new_H = Hamiltonian(new_x, new_p)
        xs = xs.at[i].set(new_x)
        ps = ps.at[i].set(new_p)
        Hs = Hs.at[i].set(new_H)
        new_carry = (xs, ps, Hs, i + 1)
        out = new_flag
        return new_carry, out

    final_carry, outs = lax.scan(body, init_carry, xs=None, length=L)
    final_xs, final_ps, final_Hs, _ = final_carry
    return final_xs, final_ps, final_Hs, outs


xs, ps, hs, flags = rs_integrate((x0, p0), L)
print("Hamiltonian by rs: ", hs)
print("projection flags: ", flags)
rs_solutions = jnp.hstack([xs, ps])

print("analytic solution: ", analytic_solutions)
print("leapfrog solution: ", leapfrog_solutions)
print("rs projected solution: ", rs_solutions)

# # Hamiltonian error over time
# plt.figure(figsize=(7, 4))
# plt.plot(np.arange(L + 1), hs - hs[0], marker="o", markersize=3)
# plt.axhline(0, linestyle="--")
# plt.xlabel("Leapfrog step")
# plt.ylabel(r"$H(q_n,p_n)-H(q_0,p_0)$")
# plt.title("Hamiltonian error along leapfrog trajectory")
# plt.show()
