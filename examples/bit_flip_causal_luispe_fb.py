# bit_flip_causal_luispe_fb.py
# Feedback-agnostic SME dynamics (tools.one_step / run_trajectories).
# Only fb-specific pieces live here: Omega1 / Omega2 builder + setup.
# Luispe EC jumps C4–C6 with SBDB-style dΩ.

import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial

jax.config.update("jax_enable_x64", True)

from tools import (
    I, X, Z,
    bracket, antibracket,
    channel_pack,
    run_trajectories,
)

# =============================================================================
# FEEDBACK-SPECIFIC: choose dΩ via Omega1 / Omega2 (SBDB example)
# =============================================================================


def build_omega_sbdb(A, B, PQ, gamma):
    """
    SBDB dΩ = Σ (Omega1 dt + Omega2 dy): cancel open EC processes at PQ.

        Omega1 = i (γ/2) {A,B} + i γ [A² + ½[A,B], PQ]
        Omega2 = -i B - i [A, PQ]

    A, B: (n_ec, N, N), PQ: (N, N), gamma: (n_ec,)
    Returns Omega1, Omega2 each (n_ec, N, N).
    """
    g = jnp.asarray(gamma, dtype=A.real.dtype)[:, None, None]
    A2 = jnp.matmul(A, A)
    commAB = bracket(A, B)
    anticomAB = antibracket(A, B)

    gamma_fb = 1

    Omega1 = gamma_fb * (
        1j * (g / 2.0) * anticomAB
        + 1j * g * bracket(A2 + 0.5 * commAB, PQ)
    )
    Omega2 = gamma_fb * (-1j * B - 1j * bracket(A, PQ))
    return Omega1, Omega2


# =============================================================================
# OPERATORS / SETUP
# =============================================================================

n = 3
N = 2**n

T = 10.0
L = 1000
dt = T / L
n_traj = 50

gamma_error = jnp.array([1e-3, 2e-3, 3e-3], dtype=jnp.float64)
gamma_ec = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float64)

X1 = jnp.kron(jnp.kron(X, I), I)
X2 = jnp.kron(jnp.kron(I, X), I)
X3 = jnp.kron(jnp.kron(I, I), X)

e000 = jnp.zeros((N,), dtype=jnp.complex128).at[0].set(1.0)
e111 = jnp.zeros((N,), dtype=jnp.complex128).at[-1].set(1.0)
PQ = e000[:, None] * jnp.conj(e000)[None, :] + e111[:, None] * jnp.conj(e111)[None, :]

C_error = jnp.stack([X1, X2, X3], axis=0)

C4 = 0.5 * antibracket(PQ, X1) + X1 @ PQ @ X1
C5 = 0.5 * antibracket(PQ, X2) + X2 @ PQ @ X2
C6 = 0.5 * antibracket(PQ, X3) + X3 @ PQ @ X3
C_ec = jnp.stack([C4, C5, C6], axis=0)

pack_error = channel_pack(C_error)
pack_ec = channel_pack(C_ec)

Omega1, Omega2 = build_omega_sbdb(
    pack_ec["A"], pack_ec["B"], PQ, gamma_ec
)

q = e000
psi0 = q / jnp.linalg.norm(q)
P0 = psi0[:, None] * jnp.conj(psi0)[None, :]
rho0 = jnp.broadcast_to(P0[None, :, :], (n_traj, N, N))


@partial(jit, static_argnames=("n_steps",))
def run(rho0, key, n_steps):
    return run_trajectories(
        rho0,
        key,
        n_steps,
        pack_error,
        pack_ec,
        gamma_error,
        gamma_ec,
        Omega1,
        Omega2,
        dt,
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    key = random.PRNGKey(5)
    rho_all = run(rho0, key, L)
    rho_all.block_until_ready()

    F_all = jnp.real(jnp.einsum("stij,ji->st", rho_all, PQ))
    times = dt * jnp.arange(1, L + 1)
    mean_F = jnp.mean(F_all, axis=1)

    plt.figure(2)
    plt.semilogy(times, 1.0 - jnp.abs(mean_F), "k")
    plt.xlabel("time")
    plt.ylabel("1 - F")
    plt.tight_layout()
    plt.show()
