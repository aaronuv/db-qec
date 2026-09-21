"""Causal / Ito / Strato checks for Luispe feedback.

Convention: A already includes √γ; Lindblad jump L = A − i Ω₂; monitor A; dW² = dt.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

jax.config.update("jax_enable_x64", True)

from tools import (
    I,
    X,
    antibracket,
    bracket,
    build_HF,
    channel_pack,
    run_trajectories_ito,
    run_trajectories_ito_nomonitor,
    run_trajectories_strato,
)


def build_omega_luispe(PQ, R, A, gamma_ec):
    """
    Luispe feedback with rate already in A (= √γ a):
        Ω₂ = √γ · i [PQ, R] / 2
        Ω₁ = −½ {A, Ω₂}
    so H_F = ½ (Ω₂ A + A Ω₂) + Ω₁ = 0 for Hermitian A.
    """
    n_ec = R.shape[0]
    PQ_rep = jnp.broadcast_to(PQ, (n_ec, PQ.shape[-2], PQ.shape[-1]))
    sqrt_g = jnp.sqrt(gamma_ec)[:, None, None]
    Omega2 = 1j * sqrt_g * bracket(PQ_rep, R) / 2.0
    Omega1 = -0.5 * antibracket(A, Omega2)
    return Omega1, Omega2


def build_omega_sbdb(A, B, PQ, gamma):
    """
    SBDB dΩ for Strato causal FB on Luispe EC jumps:
        Omega1 = i (γ/2) {A,B} + i γ [A² + ½[A,B], PQ]
        Omega2 = -i B - i [A, PQ]
    """
    g = jnp.asarray(gamma, dtype=A.real.dtype)[:, None, None]
    A2 = jnp.matmul(A, A)
    commAB = bracket(A, B)
    anticomAB = antibracket(A, B)
    Omega1 = 1j * (g / 2.0) * anticomAB + 1j * g * bracket(A2 + 0.5 * commAB, PQ)
    Omega2 = -1j * B - 1j * bracket(A, PQ)
    return Omega1, Omega2


def _luispe_ops(gamma_ec=None, lamb=1.0):
    """3-qubit bit-flip Luispe operators with √γ absorbed into A."""
    n = 3
    N = 2**n
    if gamma_ec is None:
        gamma_ec = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64)

    X1 = jnp.kron(jnp.kron(X, I), I)
    X2 = jnp.kron(jnp.kron(I, X), I)
    X3 = jnp.kron(jnp.kron(I, I), X)

    e000 = jnp.zeros((N,), dtype=jnp.complex128).at[0].set(1.0)
    e111 = jnp.zeros((N,), dtype=jnp.complex128).at[-1].set(1.0)
    PQ = e000[:, None] * jnp.conj(e000)[None, :] + e111[:, None] * jnp.conj(e111)[None, :]

    R = jnp.stack([X1, X2, X3], axis=0)
    PQ_rep = jnp.broadcast_to(PQ, (3, N, N))
    a_ec = 0.5 * antibracket(PQ_rep, R) + lamb * jnp.matmul(R, jnp.matmul(PQ_rep, R))
    C_ec = jnp.sqrt(gamma_ec)[:, None, None] * a_ec
    C_error = jnp.stack([X1, X2, X3], axis=0)  # scale at call site if needed
    return e000, PQ, R, C_error, C_ec, channel_pack(C_error), channel_pack(C_ec)


def _rho0_from_codespace(e000, n_traj, var=0.0, key=None):
    q = e000
    if var > 0.0:
        if key is None:
            key = random.PRNGKey(0)
        noise = random.normal(key, q.shape, dtype=q.dtype) * jnp.sqrt(var)
        psi0 = q + noise
        psi0 = psi0 / jnp.linalg.norm(psi0)
    else:
        psi0 = q
    P0 = psi0[:, None] * jnp.conj(psi0)[None, :]
    N = e000.shape[0]
    return jnp.broadcast_to(P0[None, :, :], (n_traj, N, N))


def test_luispe_fb_HF_zero():
    """For Luispe FB, H_F vanishes with absorbed-rate A and Ω."""
    gamma_ec = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64)
    _, PQ, R, _, C_ec, _, pack_ec = _luispe_ops(gamma_ec)
    Omega1, Omega2 = build_omega_luispe(PQ, R, C_ec, gamma_ec)
    A_ec = pack_ec["A"] + pack_ec["B"]
    HF = build_HF(Omega1, Omega2, A_ec)
    np.testing.assert_allclose(HF, 0.0, atol=1e-12)


def test_luispe_fb_HF_zero_expanded_jumps():
    """H_F = 0 for EC jumps on {I, X1, X2, X3} with √γ in A."""
    n = 3
    N = 2**n
    lamb = 1.0
    gamma_ec = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float64)

    X1 = jnp.kron(jnp.kron(X, I), I)
    X2 = jnp.kron(jnp.kron(I, X), I)
    X3 = jnp.kron(jnp.kron(I, I), X)
    Id3 = jnp.eye(N, dtype=jnp.complex128)

    e000 = jnp.zeros((N,), dtype=jnp.complex128).at[0].set(1.0)
    e111 = jnp.zeros((N,), dtype=jnp.complex128).at[-1].set(1.0)
    PQ = e000[:, None] * jnp.conj(e000)[None, :] + e111[:, None] * jnp.conj(e111)[None, :]

    R = jnp.stack([Id3, X1, X2, X3], axis=0)
    PQ_rep = jnp.broadcast_to(PQ, (R.shape[0], N, N))
    a_ec = 0.5 * antibracket(PQ_rep, R) + lamb * jnp.matmul(R, jnp.matmul(PQ_rep, R))
    C_ec = jnp.sqrt(gamma_ec)[:, None, None] * a_ec
    pack_ec = channel_pack(C_ec)

    Omega1, Omega2 = build_omega_luispe(PQ, R, C_ec, gamma_ec)
    A_ec = pack_ec["A"] + pack_ec["B"]
    HF = build_HF(Omega1, Omega2, A_ec)
    np.testing.assert_allclose(HF, 0.0, atol=1e-12)


def test_luispe_fb_zero_error_infidelity_zero_ito():
    """Deterministic Luispe step (dW=0, γ_E=0) keeps codespace fidelity = 1."""
    from tools import one_step_ito

    gamma_ec = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64)
    gamma_error = jnp.zeros(3, dtype=jnp.float64)
    dt = 0.01

    e000, PQ, R, C_error, C_ec, _, pack_ec = _luispe_ops(gamma_ec)
    pack_error = channel_pack(jnp.zeros_like(C_error))
    Omega1, Omega2 = build_omega_luispe(PQ, R, C_ec, gamma_ec)
    rho0 = _rho0_from_codespace(e000, n_traj=4, var=0.0)

    dw_e = jnp.zeros((4, 3), dtype=jnp.float64)
    dw_c = jnp.zeros((4, 3), dtype=jnp.float64)
    rho1 = one_step_ito(
        rho0, dw_e, dw_c, pack_error, pack_ec,
        gamma_error, gamma_ec, Omega1, Omega2, dt,
    )
    F = jnp.real(jnp.einsum("tij,ji->t", rho1, PQ))
    np.testing.assert_allclose(1.0 - F, 0.0, atol=1e-10)

    # Effective jump L = A − i Ω₂ annihilates |000⟩
    A_ec = pack_ec["A"] + pack_ec["B"]
    L = A_ec - 1j * Omega2
    np.testing.assert_allclose(L @ e000, 0.0, atol=1e-12)


def test_luispe_fb_zero_error_infidelity_zero_ito_nomonitor():
    """Deterministic nomonitor Luispe step (dW=0, γ_E=0) keeps fidelity = 1."""
    from tools import one_step_ito_nomonitor

    gamma_ec = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64)
    gamma_error = jnp.zeros(3, dtype=jnp.float64)
    dt = 0.01

    e000, PQ, R, C_error, C_ec, _, pack_ec = _luispe_ops(gamma_ec)
    pack_error = channel_pack(jnp.zeros_like(C_error))
    Omega1, Omega2 = build_omega_luispe(PQ, R, C_ec, gamma_ec)
    rho0 = _rho0_from_codespace(e000, n_traj=4, var=0.0)

    dw_c = jnp.zeros((4, 3), dtype=jnp.float64)
    rho1 = one_step_ito_nomonitor(
        rho0, dw_c, pack_error, pack_ec,
        gamma_error, gamma_ec, Omega1, Omega2, dt,
    )
    F = jnp.real(jnp.einsum("tij,ji->t", rho1, PQ))
    np.testing.assert_allclose(1.0 - F, 0.0, atol=1e-10)


def test_luispe_fb_zero_error_infidelity_zero_strato():
    """Luispe EC + SBDB FB (Strato) with var=0 and gamma_error=0 keeps fidelity = 1."""
    gamma_ec = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64)
    gamma_error = jnp.zeros(3, dtype=jnp.float64)
    n_traj, T, L = 8, 1.0, 100
    dt = T / L

    # Strato path still uses dimensionless pack + gamma multipliers in the SME
    n = 3
    N = 2**n
    X1 = jnp.kron(jnp.kron(X, I), I)
    X2 = jnp.kron(jnp.kron(I, X), I)
    X3 = jnp.kron(jnp.kron(I, I), X)
    e000 = jnp.zeros((N,), dtype=jnp.complex128).at[0].set(1.0)
    e111 = jnp.zeros((N,), dtype=jnp.complex128).at[-1].set(1.0)
    PQ = e000[:, None] * jnp.conj(e000)[None, :] + e111[:, None] * jnp.conj(e111)[None, :]
    C_error = jnp.stack([X1, X2, X3], axis=0)
    C_ec = jnp.stack(
        [
            0.5 * antibracket(PQ, X1) + X1 @ PQ @ X1,
            0.5 * antibracket(PQ, X2) + X2 @ PQ @ X2,
            0.5 * antibracket(PQ, X3) + X3 @ PQ @ X3,
        ],
        axis=0,
    )
    pack_error = channel_pack(C_error)
    pack_ec = channel_pack(C_ec)
    Omega1, Omega2 = build_omega_sbdb(
        pack_ec["A"], pack_ec["B"], PQ, gamma_ec
    )
    rho0 = _rho0_from_codespace(e000, n_traj, var=0.0)

    rho_all = run_trajectories_strato(
        rho0,
        random.PRNGKey(0),
        L,
        pack_error,
        pack_ec,
        gamma_error,
        gamma_ec,
        Omega1,
        Omega2,
        dt,
    )
    F_all = jnp.real(jnp.einsum("stij,ji->st", rho_all, PQ))
    np.testing.assert_allclose(1.0 - F_all, 0.0, atol=1e-10)
