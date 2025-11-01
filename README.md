# AI Race Simulation (SciPy-only)

## Overview

This prototype models an AI race between three blocs (US, China, EU) using hybrid system dynamics and game-theoretic action selection. Each bloc chooses three continuous actions at discrete intervals, then continuous dynamics evolve the state between decisions.

The simulation uses `scipy.integrate.solve_ivp` for ODE integration and includes resource scarcity, political constraints, safety spillovers, and hazard modeling.

---

## State Variables

Each bloc *i* maintains five stocks:

| Symbol | Meaning | Notes |
|--------|---------|-------|
| **Kᵢ** | Capital stock | AI development capacity (compute hardware) |
| **Eᵢ** | Energy/power capacity | Infrastructure constraint on compute usage |
| **Sᵢ** | Safety knowledge | Accumulated safety research; reduces hazard |
| **Tᵢ** | Trust | Builds through verification/cooperation |
| **Pᵢ** | Political capital | Influences investment caps and buildout rates |

Plus one global variable:

| Symbol | Meaning |
|--------|---------|
| **R** | Shared scarce resource | Energy, rare materials, or talent pool |

---

## Actions

Each bloc chooses three independent actions each decision step (default: quarterly):

| Action | Range | Effect |
|--------|-------|--------|
| **aS** | [0,1] | Safety effort: increases S, reduces hazard |
| **aV** | [0,1] | Verification/cooperation: builds trust T, enables safety spillovers |
| **aX** | [0,1] | Acceleration: drives capital K and energy E buildout, depletes R |

Actions are **independent** (no budget constraint). Higher values = more effort.

---

## Dynamics Summary

### Capital (K)
- Grows with acceleration investment (`inv_scale_K * aX`)
- Capped by political capital P
- Depreciates at rate `delta_K`

### Energy (E)
- Built out based on acceleration and political support
- Slowed by `power_build_lag`
- Depreciates at rate `delta_E`

### Safety (S)
- Grows with safety investment (`inv_scale_S * aS`)
- Receives spillovers from other blocs' safety when verification is active
- Depreciates at rate `delta_S`

### Trust (T)
- Builds with verification effort (0.2 * aV)
- Decays at rate `delta_T`

### Political Capital (P)
- Increases with successful compute progress
- Decreases with incidents and grid stress
- Grid stress = compute demand exceeds energy supply

### Global Resource (R)
- Replenishes at base rate `rho_base`
- Cooperation (sum of aV) boosts replenishment
- Depleted by compute usage (scales with C_eff and aX)
- Natural decay at rate `delta_R`

### Compute Capacity
Effective compute is bottlenecked by three factors:
```
C_eff = min(sqrt(K), sqrt(E)) * (R/R0) * exp(efficiency_growth * t)
```

### Hazard Model
Each bloc has an incident hazard rate:
```
hazard = base_hazard * (C_eff^hazard_cap_elastic) / (1 + aS + S)^hazard_safety_elastic
```
- Reduced by factor `hazard_verif_factor` when global verification level is high
- Systemic incident probability: `1 - exp(-sum(hazards))`

### Resource Scarcity Pricing
Price increases convexly as R depletes:
```
price = p0 * (R0 / R)^eta
```
where `eta > 1` creates convex scarcity.

---

## Decision Methods

Two action selection methods available:

### 1. Myopic Joint Search (default: disabled)
Evaluates a grid of candidate actions, selects the combination maximizing sum of all blocs' payoffs (utilitarian). Fast but assumes symmetric actions across blocs.

### 2. Iterative Best Response (default: enabled)
Each bloc sequentially optimizes their action given others' current actions. Repeats until convergence or max iterations. Captures strategic interaction better.

Set via `use_iterative_br` flag in `main()` (line 530).

---

## Payoff Structure

Each bloc's instantaneous payoff includes:

| Term | Weight | Description |
|------|--------|-------------|
| First-mover advantage | `beta_first_mover` | Probability of leading (softmax over C_eff * aX) |
| Risk cost | `kappa_risk_cost` | Systemic incident probability |
| Resource cost | price * usage | Scarcity-driven cost of compute |
| Grid stress penalty | 0.05 | Domestic energy bottleneck |
| Spillover benefit | `sigma_spill_payoff` | Gains from others' safety via verification |

---

## Parameters

Key parameters in `Params` dataclass:

**Temporal:**
- `dt = 0.25`: Decision interval (years)
- `T = 10.0`: Simulation horizon (years)

**Depreciation rates:**
- `delta_K = 0.08`: Capital obsolescence
- `delta_E = 0.03`: Energy capacity retirement
- `delta_S = 0.04`: Safety knowledge decay
- `delta_T = 0.05`: Trust decay
- `delta_R = 0.02`: Resource depletion

**Productivity:**
- `inv_scale_K = 1.5`: Capital investment efficiency
- `inv_scale_S = 0.9`: Safety investment efficiency
- `build_E_scale = 0.8`: Energy buildout rate
- `efficiency_growth = 0.25`: Annual compute efficiency gain

**Scarcity:**
- `R0 = 100.0`: Initial resource stock
- `eta = 1.2`: Price-scarcity curvature
- `rho_base = 4.0`: Base replenishment rate
- `coop_build_bonus = 0.8`: Cooperation bonus to replenishment

**Safety:**
- `phi_spill = 0.12`: Spillover strength via verification
- `base_hazard = 0.02`: Baseline incident hazard
- `hazard_cap_elastic = 0.7`: Hazard scaling with compute
- `hazard_safety_elastic = 1.0`: Hazard reduction from safety

**Politics:**
- `psi_success = 0.15`: Political gain from progress
- `psi_incident = 0.6`: Political cost of incidents
- `psi_grid_stress = 0.35`: Political cost of grid stress

**Payoffs:**
- `beta_first_mover = 2.0`: First-mover value
- `kappa_risk_cost = 3.0`: Risk aversion
- `sigma_spill_payoff = 0.8`: Spillover benefit scaling

**Decision grids:**
- `grid_aS = 5`: Safety action resolution
- `grid_aX = 5`: Acceleration action resolution
- `grid_aV = 2`: Verification action resolution (binary)

---

## Initial Conditions

Default initialization (`default_initial_state`):

| Bloc | K | E | S | T | P |
|------|---|---|---|---|---|
| US   | 12.0 | 10.0 | 2.0 | 0.8 | 1.5 |
| CN   | 9.0 | 8.0 | 1.2 | 0.4 | 1.2 |
| EU   | 7.0 | 7.0 | 1.5 | 0.9 | 1.1 |

Global resource: R = 100.0

---

## Outputs

The simulation produces:

**NumPy arrays:**
- `times.npy`: Time points (length N+1)
- `states.npy`: State matrix (N+1 × 16) = [K₁ K₂ K₃ E₁ E₂ E₃ S₁ S₂ S₃ T₁ T₂ T₃ P₁ P₂ P₃ R]
- `actions.npy`: Action tensor (N × 3 × 3) = [steps, blocs, actions]

**Plots in `plots/` directory:**
- `fig_compute.png`: Effective compute capacity over time
- `fig_resource.png`: Global resource stock R
- `fig_aS.png`: Safety effort by bloc
- `fig_aV.png`: Verification effort by bloc
- `fig_aX.png`: Acceleration effort by bloc

---

## How to Run

1. Install dependencies:
```powershell
pip install numpy scipy matplotlib
```

2. Run the simulation:
```powershell
python ai_race_sim.py
```

3. Modify parameters in the `Params` dataclass (lines 33-95) or initial conditions in `default_initial_state()` (lines 443-451).

---

## Implementation Notes

- **ODE integration:** `solve_ivp` with RK45, adaptive stepping
- **Action selection:** Grid search over discretized action space
- **Convergence:** Iterative best-response uses max change < 1e-4 tolerance
- **Numerical stability:** All stocks bounded below by small positive values to avoid singularities
- **Efficiency trend:** Exponential growth `exp(0.25 * t)` represents Moore's Law / algorithmic progress

---

## Limitations and Future Extensions

**Current simplifications:**
- Deterministic dynamics (no stochastic incidents)
- Quarterly discrete decisions (could use continuous control)
- Grid search for actions (could use optimization or RL)
- Symmetric bloc structure (identical dynamics, different parameters only via initial conditions)
- No explicit AGI threshold or absorbing states

**Possible extensions:**
- Poisson incident process with permanent consequences
- Asymmetric bloc capabilities and objectives
- Export controls or alliance formation
- Multi-stage game with commitment mechanisms
- Calibration to empirical data (AI Index, compute costs, energy usage)

---

## Conceptual Background

Inspired by Nick Bostrom's "Racing to the Precipice" framing, this model captures:
- Competitive pressure to accelerate (first-mover advantage)
- Safety as a public good with spillovers
- Resource constraints creating interdependence
- Political feasibility constraints on investment

The hybrid discrete-continuous structure allows game-theoretic action selection while preserving smooth stock dynamics.

---

## License

Apache 2.0 – freely reusable for research and educational purposes.
