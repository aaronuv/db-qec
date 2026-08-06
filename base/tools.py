import jax.numpy as jnp
from jax import lax, random

# --- Pauli matrices (dense; N=2 is tiny) ---
I = jnp.eye(2, dtype=jnp.complex128)
X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

e = (I, X, Y, Z)  # e[0]=I, e[1]=X, e[2]=Y, e[3]=Z

sigmap = 0.5 * (X + 1j * Y)
sigmam = 0.5 * (X - 1j * Y)


def bracket(A, B):
    """Commutator [A, B] = AB - BA. Supports leading batch dims."""
    return jnp.matmul(A, B) - jnp.matmul(B, A)


def antibracket(A, B):
    """Anticommutator {A, B} = AB + BA. Supports leading batch dims."""
    return jnp.matmul(A, B) + jnp.matmul(B, A)


def sigmax(i, n):
    """X on qubit i (1-based, like MATLAB), identity elsewhere. Shape (2^n, 2^n)."""
    ops = [X if k == i else I for k in range(1, n + 1)]
    out = ops[0]
    for op in ops[1:]:
        out = jnp.kron(out, op)
    return out


def sigmay(i, n):
    """Y on qubit i (1-based), identity elsewhere."""
    ops = [Y if k == i else I for k in range(1, n + 1)]
    out = ops[0]
    for op in ops[1:]:
        out = jnp.kron(out, op)
    return out


def sigmaz(i, n):
    """Z on qubit i (1-based), identity elsewhere."""
    ops = [Z if k == i else I for k in range(1, n + 1)]
    out = ops[0]
    for op in ops[1:]:
        out = jnp.kron(out, op)
    return out


def build_qubit(theta, phi):
    """Single-qubit state |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩."""
    return jnp.array(
        [jnp.cos(theta / 2), jnp.exp(1j * phi) * jnp.sin(theta / 2)],
        dtype=jnp.complex128,
    )


def split_AB(C):
    """C -> (A, B) with A = (C+C†)/2, B = (C-C†)/2. Leading channel axis OK."""
    Cd = jnp.conj(jnp.swapaxes(C, -1, -2))
    return 0.5 * (C + Cd), 0.5 * (C - Cd)


def channel_pack(C):
    """Precompute A, B, A2, [A,B], {A,B} for a stack of jump operators."""
    A, B = split_AB(C)
    return {
        "A": A,
        "B": B,
        "A2": jnp.matmul(A, A),
        "commAB": bracket(A, B),
        "anticomAB": antibracket(A, B),
    }


def measurement_increments(rho, A, gamma, dw, dt):
    """dy_i = 2 γ_i Tr(A_i ρ) dt + dw_i"""
    expect = jnp.real(jnp.einsum("cij,tji->tc", A, rho))
    return 2.0 * gamma[None, :] * expect * dt + dw


def build_dhsb(gamma, anticomAB, B, dy, dt):
    """
    SB process dH^SB (state-independent operator process).

    gamma:     (n_ch,)
    anticomAB: (n_ch, N, N)
    B:         (n_ch, N, N)
    dy:        (n_traj, n_ch)
    dt:        scalar

    Returns: (n_traj, N, N)
        sum_i [ -i γ_i/2 {A_i,B_i} dt + i B_i dy_i ]
    """
    drift = (-1j * gamma * 0.5 * dt)[None, :, None, None] * anticomAB[None, ...]
    innov = (1j * dy)[..., None, None] * B[None, ...]
    return jnp.sum(drift + innov, axis=1)


def build_dhdb(gamma, Ato2, commAB, A, dy, dt):
    """
    DB process dH^DB (state-independent operator process — NOT yet commutated).

    The full Hamiltonian without feedback is
        dH = dH^SB + i [dH^DB, ρ]

    gamma:  (n_ch,)
    Ato2:   (n_ch, N, N)
    commAB: (n_ch, N, N)
    A:      (n_ch, N, N)
    dy:     (n_traj, n_ch)
    dt:     scalar

    Returns: (n_traj, N, N)
        sum_i ( -γ_i A_i² dt - ½ γ_i [A_i,B_i] dt + dy_i A_i )
    """
    ops = (
        (-gamma * dt)[None, :, None, None] * Ato2[None, ...]
        - (0.5 * gamma * dt)[None, :, None, None] * commAB[None, ...]
        + dy[..., None, None] * A[None, ...]
    )
    return jnp.sum(ops, axis=1)


def assemble_dH(dH_sb, dH_db, rho, dOmega):
    """
    Full Hamiltonian process:
        dH = dH^SB + i [dH^DB, ρ] + dΩ

    dH_sb, dH_db, rho, dOmega: (n_traj, N, N)
    (pass dOmega = 0 for no feedback)
    """
    return dH_sb + 1j * bracket(dH_db, rho) + dOmega


def build_domega(Omega1, Omega2, dy, dt):
    """
    Generic feedback increment from caller-supplied coefficient arrays:
        dΩ = Σ_a (Omega1_a dt + Omega2_a dy_a)

    Omega1, Omega2: (n_ch, N, N)   — built outside tools (fb-specific)
    dy:             (n_traj, n_ch)
    dt:             scalar

    Returns: (n_traj, N, N)
    """
    drift = dt * jnp.sum(Omega1, axis=0)              # (N, N), broadcasts
    innov = jnp.einsum("cij,tc->tij", Omega2, dy)      # (n_traj, N, N)
    return drift + innov


def build_h2(B, A, rho):
    """
    Dy-coefficient of the open EC Hamiltonian process:
        H^(2),EC = i B + i [A, ρ]

    B, A: (n_ec, N, N)
    rho:  (n_traj, N, N)
    Returns: (n_ec, n_traj, N, N)
    """
    A_b = A[:, None, :, :]
    B_b = B[:, None, :, :]
    r = rho[None, :, :, :]
    return 1j * (B_b + bracket(A_b, r))


def project_pure(rho):
    """
    Rank-1 projector onto the dominant eigenspace of rho (MATLAB eigs + v*v').
    rho: (..., N, N) -> (..., N, N)
    """
    herm = 0.5 * (rho + jnp.conj(jnp.swapaxes(rho, -1, -2)))
    _, evecs = jnp.linalg.eigh(herm)
    v = evecs[..., -1]
    return v[..., :, None] * jnp.conj(v[..., None, :])


def build_omega_corrections(rho, Omega2, H2, gamma_ec, dt):
    """
    Feedback / causality corrections (sum over EC channels):
        - (γ/2) [Ω^(2), [Ω^(2), ρ]] dt - γ [Ω^(2), [H^(2),EC, ρ]] dt

    Omega2:   (n_ec, N, N)
    H2:       (n_ec, n_traj, N, N) from build_h2
    gamma_ec: (n_ec,)
    rho:      (n_traj, N, N)
    Returns:  (n_traj, N, N)
    """
    Om = Omega2[:, None, :, :]
    r = rho[None, :, :, :]

    ad_Om = bracket(Om, r)
    ad2_Om = bracket(Om, ad_Om)
    ad_H = bracket(H2, r)
    ad_Om_H = bracket(Om, ad_H)

    g = gamma_ec[:, None, None, None]
    corr = -0.5 * g * ad2_Om * dt - g * ad_Om_H * dt
    return jnp.sum(corr, axis=0)


def update_rho(rho, dH_sb, dH_db, dOmega, Omega2, A_ec, B_ec, gamma_ec, dt):
    """
    Stratonovich midpoint + pure-state projection for
        dρ = -i [dH^E + dH^EC + dΩ, ρ]
             - (γ^EC/2) [Ω^(2), [Ω^(2), ρ]] dt
             - γ^EC [Ω^(2), [H^(2),EC, ρ]] dt

    with dH(ρ) = dH^SB + i [dH^DB, ρ] + dΩ and
         H^(2),EC = i B + i [A, ρ]  (open EC dy-coefficients).

    Feedback enters only through dΩ / Ω^(2) (caller-supplied Omega1, Omega2).
    dH is rebuilt at the pivot because of i[dH_DB, ρ] and H^(2)(ρ).
    """
    dH = assemble_dH(dH_sb, dH_db, rho, dOmega)
    d_comm = -1j * bracket(dH, rho)
    H2 = build_h2(B_ec, A_ec, rho)
    d_corr = build_omega_corrections(rho, Omega2, H2, gamma_ec, dt)

    rho_pivot = project_pure(rho + d_comm)

    dH_p = assemble_dH(dH_sb, dH_db, rho_pivot, dOmega)
    d_comm_p = -1j * bracket(dH_p, rho_pivot)
    H2_p = build_h2(B_ec, A_ec, rho_pivot)
    d_corr_p = build_omega_corrections(rho_pivot, Omega2, H2_p, gamma_ec, dt)

    drho = 0.5 * (d_comm + d_comm_p + d_corr + d_corr_p)
    return project_pure(rho + drho)


def one_step(
    rho,
    dw_error,
    dw_ec,
    pack_error,
    pack_ec,
    gamma_error,
    gamma_ec,
    Omega1,
    Omega2,
    dt,
):
    """
    One Stratonovich step (all trajectories) for the agnostic feedback SME:
        dρ = -i [dH^E + dH^EC + dΩ, ρ]
             - (γ^EC/2) [Ω^(2), [Ω^(2), ρ]] dt
             - γ^EC [Ω^(2), [H^(2),EC, ρ]] dt

    with dΩ = build_domega(Omega1, Omega2, dy_ec, dt).
    Omega1 / Omega2 are the only feedback-specific inputs (caller-supplied).

    rho:         (n_traj, N, N)
    dw_error:    (n_traj, n_error)
    dw_ec:       (n_traj, n_ec)
    pack_*:      channel_pack dicts
    gamma_*:     (n_ch,)
    Omega1/2:    (n_ec, N, N)
    dt:          scalar

    Returns: rho_next (n_traj, N, N)
    """
    dy_error = measurement_increments(
        rho, pack_error["A"], gamma_error, dw_error, dt
    )
    dy_ec = measurement_increments(
        rho, pack_ec["A"], gamma_ec, dw_ec, dt
    )

    dH_sb = build_dhsb(
        gamma_error, pack_error["anticomAB"], pack_error["B"], dy_error, dt
    ) + build_dhsb(
        gamma_ec, pack_ec["anticomAB"], pack_ec["B"], dy_ec, dt
    )
    dH_db = build_dhdb(
        gamma_error, pack_error["A2"], pack_error["commAB"], pack_error["A"], dy_error, dt
    ) + build_dhdb(
        gamma_ec, pack_ec["A2"], pack_ec["commAB"], pack_ec["A"], dy_ec, dt
    )
    dOmega = build_domega(Omega1, Omega2, dy_ec, dt)

    return update_rho(
        rho,
        dH_sb,
        dH_db,
        dOmega,
        Omega2,
        pack_ec["A"],
        pack_ec["B"],
        gamma_ec,
        dt,
    )


def run_trajectories_strato(
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
):
    """
    Scan one_step over time for a batch of trajectories.

    Returns rho_all: (n_steps, n_traj, N, N)
    """
    n_traj = rho0.shape[0]
    key_e, key_c = random.split(key)
    dw_error = jnp.sqrt(gamma_error * dt) * random.normal(
        key_e, (n_steps, n_traj, gamma_error.shape[0])
    )
    dw_ec = jnp.sqrt(gamma_ec * dt) * random.normal(
        key_c, (n_steps, n_traj, gamma_ec.shape[0])
    )

    def body(rho, dw_pair):
        dw_e, dw_c = dw_pair
        rho = one_step(
            rho,
            dw_e,
            dw_c,
            pack_error,
            pack_ec,
            gamma_error,
            gamma_ec,
            Omega1,
            Omega2,
            dt,
        )
        return rho, rho

    _, rho_all = lax.scan(body, rho0, (dw_error, dw_ec))
    return rho_all


# --- Ito SME (Wiseman–Milburn feedback form) ---------------------------------


def dissipator_D(C, rho):
    """
    Per-channel Lindblad dissipator
        D[C]ρ = C ρ C† − ½ {C† C, ρ}

    C:   (n_ch, N, N)
    rho: (n_traj, N, N)
    Returns: (n_ch, n_traj, N, N)
    """
    Cd = jnp.conj(jnp.swapaxes(C, -1, -2))
    Cr = jnp.matmul(C[:, None, :, :], rho[None, :, :, :])
    C_rho_Cd = jnp.matmul(Cr, Cd[:, None, :, :])
    CdC = jnp.matmul(Cd, C)
    anticom = antibracket(CdC[:, None, :, :], rho[None, :, :, :])
    return C_rho_Cd - 0.5 * anticom


def superop_H(C, rho):
    """
    Per-channel measurement (innovation) superoperator
        H[C]ρ = C ρ + ρ C† − Tr((C + C†) ρ) ρ

    C:   (n_ch, N, N)
    rho: (n_traj, N, N)
    Returns: (n_ch, n_traj, N, N)
    """
    Cd = jnp.conj(jnp.swapaxes(C, -1, -2))
    Cr = jnp.matmul(C[:, None, :, :], rho[None, :, :, :])
    rCd = jnp.matmul(rho[None, :, :, :], Cd[:, None, :, :])
    expect = jnp.real(jnp.einsum("cij,tji->ct", C + Cd, rho))
    return Cr + rCd - expect[:, :, None, None] * rho[None, :, :, :]


def build_HF(Omega1, Omega2, J):
    """
    Feedback Hamiltonian (summed over EC channels)
        H_F = Σ_a [ ½ (Ω₂_a J_a + J_a† Ω₂_a) + Ω₁_a ]

    Omega1, Omega2, J: (n_ec, N, N)
    Returns: (N, N)
    """
    Jd = jnp.conj(jnp.swapaxes(J, -1, -2))
    return jnp.sum(
        0.5 * (jnp.matmul(Omega2, J) + jnp.matmul(Jd, Omega2)) + Omega1,
        axis=0,
    )


def one_step_ito(
    rho,
    dw_error,
    dw_ec,
    pack_error,
    pack_ec,
    gamma_error,
    gamma_ec,
    Omega1,
    Omega2,
    dt,
):
    """
    One Ito / Euler–Maruyama step for the feedback SME
        dρ = −i [H_F, ρ] dt + Σ γ D[L] ρ dt + Σ H[L] ρ dW

    with error jumps L = J^E, EC jumps L = C = J^C − i Ω₂, and
        H_F = ½ (Ω₂ J^C + (J^C)† Ω₂) + Ω₁
    (sums over channels understood).

    Same argument layout as one_step. dw_* ~ N(0, γ dt).
    """
    J_error = pack_error["A"] + pack_error["B"]
    J_ec = pack_ec["A"] + pack_ec["B"]
    C_ec = J_ec - 1j * Omega2

    HF = build_HF(Omega1, Omega2, J_ec)

    drift = (
        -1j * bracket(HF[None, :, :], rho)
        + jnp.einsum("c,ctij->tij", gamma_error, dissipator_D(J_error, rho))
        + jnp.einsum("c,ctij->tij", gamma_ec, dissipator_D(C_ec, rho))
    )
    diff = (
        jnp.einsum("tc,ctij->tij", dw_error, superop_H(J_error, rho))
        + jnp.einsum("tc,ctij->tij", dw_ec, superop_H(C_ec, rho))
    )
    return project_pure(rho + drift * dt + diff)


def run_trajectories_ito(
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
):
    """
    Scan one_step_ito over time for a batch of trajectories.

    Drop-in Ito counterpart of run_trajectories (same signature / return).
    Returns rho_all: (n_steps, n_traj, N, N)
    """
    n_traj = rho0.shape[0]
    key_e, key_c = random.split(key)
    dw_error = jnp.sqrt(gamma_error * dt) * random.normal(
        key_e, (n_steps, n_traj, gamma_error.shape[0])
    )
    dw_ec = jnp.sqrt(gamma_ec * dt) * random.normal(
        key_c, (n_steps, n_traj, gamma_ec.shape[0])
    )

    def body(rho, dw_pair):
        dw_e, dw_c = dw_pair
        rho = one_step_ito(
            rho,
            dw_e,
            dw_c,
            pack_error,
            pack_ec,
            gamma_error,
            gamma_ec,
            Omega1,
            Omega2,
            dt,
        )
        return rho, rho

    _, rho_all = lax.scan(body, rho0, (dw_error, dw_ec))
    return rho_all
