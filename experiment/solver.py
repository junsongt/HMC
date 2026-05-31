import os, sys
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import lax
from functools import partial


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def is_invalid(x):
    return jnp.any(jnp.isnan(x)) | jnp.any(jnp.isinf(x))


@partial(jax.jit, static_argnames=("q", "G", "nmax", "eps")) # q, G are static function arguments. jit expects function arguments to be arrays/scalars/pytrees unless we mark them static
def newton_solve(v, q, q_level, G, Gx, nmax, eps):
    """
    JAX version of the classic Newton solver.

    Solves for a in
        g(a) = q(v + Gx @ a) = 0

    where Gx = G(x) is fixed, and G(y) is evaluated at the current iterate y.

    Parameters
    ----------
    v : array, shape (m,)
        Base point.
    q : callable
        Constraint/map function q(y), returns shape (d,)
    G : callable
        Jacobian-like function G(y), returns shape (m, d)
    Gx : array, shape (m, d)
        Fixed Jacobian at x
    nmax : int
        Maximum number of iterations
    eps : float
        Tolerance on ||q(y)||

    Returns
    -------
    dict with:
        "pt"   : final point y
        "flag" : whether convergence was achieved
    """
    d = Gx.shape[1]
    a = jnp.zeros(d, dtype=v.dtype)
    y = v + Gx @ a
    qval = q(y) - q_level

    invalid = is_invalid(v) | is_invalid(qval)

    def early_return():
        return {"pt": v, "flag": False}

    def run_solver():
        Gy = G(y)
        qerr = jnp.linalg.norm(qval)
        state0 = {
            "i": jnp.array(1),
            "a": a,
            "y": y,
            "qval": qval,
            "Gy": Gy,
            "qerr": qerr,
            "invalid": is_invalid(Gy),
        }

        def cond_fun(state):
            return (state["i"] <= nmax) & (state["qerr"] > eps) & (~state["invalid"])

        def body_fun(state):
            M = state["Gy"].T @ Gx
            rhs = -state["qval"]

            delta_a = jnp.linalg.solve(M, rhs)
            invalid_delta = is_invalid(delta_a)

            a_new = state["a"] + delta_a
            y_new = v + Gx @ a_new
            qval_new = q(y_new) - q_level
            Gy_new = G(y_new)

            invalid_new = (state["invalid"] | invalid_delta | is_invalid(y_new) | is_invalid(qval_new) | is_invalid(Gy_new))

            qerr_new = jnp.linalg.norm(qval_new)

            return {
                "i": state["i"] + 1,
                "a": a_new,
                "y": y_new,
                "qval": qval_new,
                "Gy": Gy_new,
                "qerr": qerr_new,
                "invalid": invalid_new,
            }

        statef = lax.while_loop(cond_fun, body_fun, state0)
        found = (statef["qerr"] <= eps) & (~statef["invalid"])
        return {"pt": statef["y"], "flag": found}

    return lax.cond(invalid, early_return, run_solver)


@partial(jax.jit, static_argnames=("q", "nmax", "eps")) # G_x, L_x as JAX arrays must be hashable, hence don't mark them static
def quasi_newton_solve(v, q, q_level, G_x, L_x, nmax, eps):
    """
    JAX version of the quasi-Newton solver.

    Solves for a in
        g(a) = q(v + G_x @ a) = 0

    using the approximation
        ∇g(a) ≈ G_x^T G_x

    and Cholesky factor L_x such that
        G_x^T G_x = L_x L_x^T
    or equivalently according to your convention.

    Here we follow your original code:
        solve L_x z = -qval
        solve L_x.T delta_a = z

    Parameters
    ----------
    v : array, shape (m,)
        Base point.
    q : callable
        Constraint/map function q(y), returns shape (d,)
    G_x : array, shape (m, d)
        Fixed Jacobian at x
    L_x : array, shape (d, d)
        Cholesky factor used in the triangular solves
    nmax : int
        Maximum number of iterations
    eps : float
        Tolerance on ||q(y)||

    Returns
    -------
    dict with:
        "pt"   : final point y
        "flag" : whether convergence was achieved
    """
    d = G_x.shape[1]
    a = jnp.zeros(d, dtype=v.dtype)
    y = v + G_x @ a
    qval = q(y) - q_level

    invalid = is_invalid(v) | is_invalid(L_x) | is_invalid(qval)

    def early_return():
        return {"pt": v, "flag": False}

    def run_solver():
        qerr = jnp.linalg.norm(qval)

        state0 = {
            "i": jnp.array(1),
            "a": a,
            "y": y,
            "qval": qval,
            "qerr": qerr,
            "invalid": False,
        }

        def cond_fun(state):
            return (state["i"] <= nmax) & (state["qerr"] > eps) & (~state["invalid"])

        def body_fun(state):
            z = jla.solve_triangular(L_x, -state["qval"], lower=True)
            delta_a = jla.solve_triangular(L_x.T, z, lower=False)

            invalid_delta = is_invalid(delta_a)

            a_new = state["a"] + delta_a
            y_new = v + G_x @ a_new
            qval_new = q(y_new) - q_level

            invalid_new = state["invalid"] | invalid_delta | is_invalid(y_new) | is_invalid(qval_new)

            qerr_new = jnp.linalg.norm(qval_new)

            return {
                "i": state["i"] + 1,
                "a": a_new,
                "y": y_new,
                "qval": qval_new,
                "qerr": qerr_new,
                "invalid": invalid_new,
            }

        statef = lax.while_loop(cond_fun, body_fun, state0)
        found = (statef["qerr"] <= eps) & (~statef["invalid"])
        return {"pt": statef["y"], "flag": found}

    return lax.cond(invalid, early_return, run_solver)
