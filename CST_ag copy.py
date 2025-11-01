
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable

###############################################################################
# 1. Model configuration
###############################################################################

# We'll index players as 0,1,2 = [US, CN, EU] for convenience.
N_PLAYERS = 3
US, CN, EU = 0, 1, 2  # useful named indices


@dataclass
class Params:
    # Capability growth efficiency
    alpha: float  # dK/dt contribution from acceleration effort (or scalar for all players)

    # Safety growth efficiency
    gamma: float  # dS/dt contribution from safety effort

    # Spillover strength for safety via trust
    eta: float    # multiplier on T * avg_other_S

    # Trust formation / decay
    beta: float      # how fast joint verification effort builds trust
    delta_T: float   # trust decay rate

    # Safety effectiveness for diagnostic debt
    theta: float  # how much safety cancels capability in the debt metric

    # Two-phase capability growth (recursive self-improvement)
    K_threshold: float  # capability level where recursive self-improvement kicks in
    beta_dim: float     # diminishing returns strength before threshold (higher = more diminishing)

    # Do agents *care* about debt in their payoff? (kept for future)
    lam: float = 0.5  # lambda=0 means they ignore debt when choosing actions
    
    # If alpha_player is provided, it overrides alpha per player
    alpha_player: np.ndarray = None  # shape (3,) per-player capability growth efficiency


@dataclass
class State:
    # State vector:
    # K[i] = capability stock for bloc i
    # S[i] = safety stock for bloc i
    # T    = global trust/verification stock
    K: np.ndarray  # shape (3,)
    S: np.ndarray  # shape (3,)
    T: float       # scalar


@dataclass
class Controls:
    # Per-bloc effort allocations at a given time t
    # aX[i] = acceleration effort for bloc i (0..1)
    # aS[i] = safety effort for bloc i (0..1)
    # aV[i] = verification/coop effort for bloc i (0..1)
    aX: np.ndarray  # shape (3,)
    aS: np.ndarray  # shape (3,)
    aV: np.ndarray  # shape (3,)


###############################################################################
# 2. Control policy
###############################################################################
# For now, policy is just a function of time t (and maybe state later).
# Later you can implement best-response: aX_i(t) = argmax payoff_i(...)
#
# Signature: policy_fn(t, y_vector) -> Controls
#
# y_vector is the flattened state we'll pass to the ODE solver. We'll provide
# a helper to unpack that into (K,S,T) if you want state-feedback policies.


def unpack_state(y: np.ndarray) -> State:
    """
    y is a flat array of length 3(K) + 3(S) + 1(T) = 7
    ordering: [K_US, K_CN, K_EU, S_US, S_CN, S_EU, T]
    """
    K = y[0:3]
    S = y[3:6]
    T = y[6]
    return State(K=K, S=S, T=T)


def simple_scenario_policy_builder(
    mode: str = "arms_race"
) -> Callable[[float, np.ndarray], Controls]:
    """
    Returns a policy function.
    - 'arms_race': high accel, low safety, low verification
    - 'treaty': moderate accel, higher safety, decent verification
    You can extend this to depend on current state for adaptive behavior.
    """

    def policy_fn(t: float, y: np.ndarray) -> Controls:
        if mode == "arms_race":
            aX = np.array([0.9, 0.9, 0.9])
            aS = np.array([0.1, 0.1, 0.1])
            aV = np.array([0.05, 0.05, 0.05])

        elif mode == "treaty":
            aX = np.array([0.6, 0.6, 0.6])
            aS = np.array([0.5, 0.5, 0.5])
            aV = np.array([0.5, 0.5, 0.5])

        else:
            # default fallback (you can raise instead)
            aX = np.array([0.8, 0.8, 0.8])
            aS = np.array([0.2, 0.2, 0.2])
            aV = np.array([0.1, 0.1, 0.1])

        return Controls(aX=aX, aS=aS, aV=aV)

    return policy_fn


###############################################################################
# 3. System dynamics (ODE right-hand side)
###############################################################################
# Continuous-time approximation of:
#
# Two-phase capability growth (recursive self-improvement):
#   If K_i < K_threshold:
#     dK_i/dt = alpha * aX_i / (1 + beta_dim * K_i)  [diminishing returns]
#   If K_i >= K_threshold:
#     dK_i/dt = alpha * aX_i * K_i  [exponential/compounding growth]
#
# dS_i/dt = gamma * aS_i
#           + eta * T * avg_other_S_i
#
# dT/dt   = beta * mean(aV_i) - delta_T * T
#
# avg_other_S_i = (sum_j S_j - S_i)/(N_PLAYERS-1)
#
# No decay in K or S (can add later).
# No resource depletion in K.
# No feedback from "risk" into behavior (lam=0 baseline).
#
# We'll feed this to solve_ivp.


def rhs_ode(
    t: float,
    y: np.ndarray,
    params: Params,
    policy_fn: Callable[[float, np.ndarray], Controls],
) -> np.ndarray:
    """
    Compute dy/dt for the flattened state y.
    """
    st = unpack_state(y)
    ctrl = policy_fn(t, y)

    dK = np.zeros(N_PLAYERS)
    dS = np.zeros(N_PLAYERS)

    # --- Capability dynamics (two-phase growth)
    # Phase 1 (K < K_threshold): Diminishing returns
    #   dK_i/dt = alpha_i * aX_i / (1 + beta_dim * K_i)
    # Phase 2 (K >= K_threshold): Recursive self-improvement (compounding)
    #   dK_i/dt = alpha_i * aX_i * K_i
    # Use per-player alpha if provided, otherwise use scalar alpha for all
    alpha_vec = params.alpha_player if params.alpha_player is not None else np.array([params.alpha] * N_PLAYERS)
    
    for i in range(N_PLAYERS):
        if st.K[i] < params.K_threshold:
            # Diminishing returns phase
            dK[i] = alpha_vec[i] * ctrl.aX[i] / (1.0 + params.beta_dim * st.K[i])
        else:
            # Recursive self-improvement phase (exponential growth)
            dK[i] = alpha_vec[i] * ctrl.aX[i] * st.K[i]

    # --- Safety dynamics
    # dS_i/dt = gamma * aS_i + eta * T * avg_other_S_i
    S_sum = np.sum(st.S)
    for i in range(N_PLAYERS):
        avg_other_S = (S_sum - st.S[i]) / (N_PLAYERS - 1)
        spillover = params.eta * st.T * avg_other_S
        dS[i] = params.gamma * ctrl.aS[i] + spillover

    # --- Trust dynamics
    # dT/dt = beta * mean(aV_i) - delta_T * T
    mean_aV = np.mean(ctrl.aV)
    dT = params.beta * mean_aV - params.delta_T * st.T

    # pack derivative
    dydt = np.zeros_like(y)
    dydt[0:3] = dK
    dydt[3:6] = dS
    dydt[6] = dT

    return dydt


###############################################################################
# 4. Diagnostics
###############################################################################
# Safety debt / overhang D_i(t) = max(0, K_i - theta * S_i)
# This is NOT fed back into dynamics yet (lam=0).
# You can compute it after integration to see how dangerous each bloc is.


def safety_debt(state: State, params: Params) -> np.ndarray:
    debt = state.K - params.theta * state.S
    debt = np.maximum(debt, 0.0)
    return debt


###############################################################################
# 5. Simulation helper
###############################################################################


def simulate(
    t_span: tuple[float, float],
    y0: np.ndarray,
    params: Params,
    policy_fn: Callable[[float, np.ndarray], Controls],
    t_eval: np.ndarray = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
):
    """
    Wrapper around solve_ivp.
    - t_span: (t0, tf)
    - y0: initial state vector [K0_US, K0_CN, K0_EU, S0_US, S0_CN, S0_EU, T0]
    - params: Params(...)
    - policy_fn: decides aX, aS, aV
    - t_eval: times at which to sample solution (np.linspace recommended)
    """

    sol = solve_ivp(
        fun=lambda t, y: rhs_ode(t, y, params, policy_fn),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        vectorized=False,
        rtol=rtol,
        atol=atol,
    )
    return sol


###############################################################################
# 6. Nash / best response hooks (future extension)
###############################################################################
# Right now, policy_fn ignores the state and doesn't optimize anything.
# Later you'll want each bloc i to solve:
#
#   maximize_i  U_i = K_i(t+dt) - lam * Debt_i(t+dt)
#
# subject to control bounds (aX_i, aS_i, aV_i in [0,1]),
# holding other blocs' controls fixed.
#
# That becomes a myopic best response. You can wrap a fixed-point search
# at each timestep to approximate Nash. You don't have to change rhs_ode for
# that; you just swap policy_fn.


###############################################################################
# 7. Example usage / quick demo
###############################################################################
if __name__ == "__main__":
    # --- Define parameters
    params = Params(
        alpha=1.0,       # capability growth rate per accel effort
        gamma=0.5,       # safety growth rate per safety effort
        eta=0.2,         # spillover strength (trust -> shared safety)
        beta=0.3,        # trust build rate from verification effort
        delta_T=0.1,     # trust decay
        theta=0.8,       # how effective safety is at offsetting capability
        K_threshold=8.0,   # AGI threshold for recursive self-improvement
        beta_dim=0.3,    # diminishing returns strength (higher = more diminishing)
        lam=0.0,         # for future payoff weighting, currently unused
    )

    # --- Initial conditions:
    # Everyone below AGI threshold, with ~10% safety coverage
    K0 = np.array([5.0, 4.0, 3.0])   # starting capability levels (US ahead, all pre-AGI)
    S0 = np.array([0.5, 0.4, 0.3])   # starting safety levels (~10% S/K ratio)
    T0 = 0.1                         # low verification regime
    y0 = np.concatenate([K0, S0, [T0]])

    # --- Time horizon
    t0, tf = 0.0, 50.0
    t_eval = np.linspace(t0, tf, 501)

    # --- Choose scenario
    mode = "arms_race"
    policy_fn = simple_scenario_policy_builder(mode=mode)

    # --- Run simulation
    sol = simulate(
        t_span=(t0, tf),
        y0=y0,
        params=params,
        policy_fn=policy_fn,
        t_eval=t_eval,
    )

    # --- Unpack results for convenience
    K_path = sol.y[0:3, :]   # shape (3, len(t_eval))
    S_path = sol.y[3:6, :]
    T_path = sol.y[6, :]

    # --- Compute safety debt over time for each bloc
    debt_path = np.zeros_like(K_path)
    for ti in range(len(t_eval)):
        st_t = State(
            K=K_path[:, ti],
            S=S_path[:, ti],
            T=T_path[ti],
        )
        debt_path[:, ti] = safety_debt(st_t, params)

    # --- Print some summary stats
    print("Final capabilities K:", K_path[:, -1])
    print("Final safety S:", S_path[:, -1])
    print("Final trust T:", T_path[-1])
    print("Final safety debt:", debt_path[:, -1])

    import matplotlib.pyplot as plt

    plt.plot(t_eval, K_path[US], label="US K")
    plt.plot(t_eval, K_path[CN], label="CN K")
    plt.plot(t_eval, K_path[EU], label="EU K")
    plt.legend()
    plt.savefig("plots/CST.png")
