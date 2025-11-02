# System Dynamics Game-Theoretic Model of the AI Development Race

## Overview

Continuous-time ODE model of AI race dynamics between three blocs (US, China, EU) using `scipy.integrate.solve_ivp`. The model tracks capability growth, safety investment, and trust formation with game-theoretic policy selection.

Key features:
- Two-phase capability growth (diminishing returns → recursive self-improvement at threshold)
- Safety spillovers mediated by trust/verification
- Nash equilibrium computation via iterated best response
- Budget-constrained actions: aX + aS + aV = 1

---

## State Variables

Per-bloc state (i ∈ {US, CN, EU}):

| Symbol | Description |
|--------|-------------|
| **Kᵢ** | Capability stock (AI development level) |
| **Sᵢ** | Safety stock (accumulated safety research) |

Global state:

| Symbol | Description |
|--------|-------------|
| **T** | Trust/verification level (scalar, shared across blocs) |

Total state dimension: 7 (3K + 3S + 1T)

---

## Actions

Each bloc allocates effort across three activities with budget constraint:

| Action | Range | Constraint | Effect |
|--------|-------|-----------|--------|
| **aXᵢ** | [0,1] | aXᵢ + aSᵢ + aVᵢ = 1 | Acceleration effort: increases capability K |
| **aSᵢ** | [0,1] | | Safety effort: increases safety S |
| **aVᵢ** | [0,1] | | Verification/cooperation: builds global trust T |

Budget constraint enforced in best-response and Bostrom minimal policies.

## Dynamics

### Capability (K)

Two-phase growth model representing transition from diminishing returns to recursive self-improvement:

**Phase 1 (K < K_threshold):** Diminishing returns (pre-AGI)
```
dKᵢ/dt = α * aXᵢ / (1 + β_dim * Kᵢ)
```

**Phase 2 (K > K_threshold):** Exponential growth (recursive self-improvement)
```
dKᵢ/dt = α * aXᵢ * Kᵢ
```

Smooth transition via tanh centered at K_threshold with width `transition_width`. Physically represents transition from "superhuman coder" to "superhuman AI researcher".

### Safety (S)

Direct investment plus spillovers from other blocs mediated by trust:
```
dSᵢ/dt = γᵢ * aSᵢ + η * T * avg_other_Sᵢ
```
where `avg_other_Sᵢ = (Σⱼ Sⱼ - Sᵢ) / (N-1)` and γᵢ is bloc-specific safety productivity.

No depreciation (knowledge accumulates).

### Trust (T)

Global verification regime built by cooperation, decays without maintenance:
```
dT/dt = β * mean(aVᵢ) - δ_T * T
```

Higher trust enables safety spillovers across blocs.

## Policy Modes

Three policy implementations available:

### 1. Simple Scenario Policy
Fixed actions independent of state. Modes:
- `arms_race`: high acceleration (0.9), low safety (0.1), low verification (0.05)
- `treaty`: moderate acceleration (0.6), higher safety (0.5), higher verification (0.5)

### 2. Best Response Policy (Nash Equilibrium)
Iterative best response to compute approximate Nash equilibrium:
- Each bloc optimizes given others' current actions via grid search
- Supports lookahead with exponential discounting (strategic)
- Optional `phase1_only_lookahead`: agents below threshold ignore exponential growth in projections
- Randomized player ordering each iteration for better convergence
- Converges when max action change < 1e-4

### 3. Bostrom Minimal Information Policy
Myopic optimization representing informationally limited agents (Bostrom 2014, "Racing to the Precipice"):
- Agents observe only own state (Kᵢ, Sᵢ, T)
- No knowledge of others' payoffs or strategies
- No lookahead - instantaneous payoff gradient maximization
- Simulates agents who know their utility function but can't predict future states

## Diagnostics and Payoffs

### Safety Debt
Diagnostic metric (not fed back into dynamics when λ=0):
```
Dᵢ = max(0, Kᵢ - θ * Sᵢ)
```
Represents capability "overhang" relative to safety preparation.

### Payoff Structure
Each bloc's utility function:
```
Uᵢ = Kᵢ - λ * Dᵢ + ω * T
```

| Term | Coefficient | Interpretation |
|------|-------------|----------------|
| Kᵢ | 1 | Direct benefit from capability |
| Dᵢ | λ | Cost of safety debt (λ=0: ignore, λ>1: danger exceeds capability benefit) |
| T | ω | Direct value of trust/cooperation |

When λ=0, agents ignore safety debt in decision-making (not in diagnostics).

## Parameters

`Params` dataclass fields:

**Capability dynamics:**
- `alpha`: Capability growth rate per acceleration effort
- `K_threshold`: Capability level triggering recursive self-improvement (e.g., 14.5)
- `beta_dim`: Diminishing returns strength in phase 1 (higher = stronger diminishing returns)
- `transition_width`: Smooth transition zone width between phases

**Safety dynamics:**
- `gamma`: Safety growth rate per safety effort, bloc-specific array [US, CN, EU] (e.g., [0.15, 0.025, 0.10])
- `eta`: Spillover strength from other blocs' safety via trust
- `theta`: Safety effectiveness at offsetting capability in debt metric

**Trust dynamics:**
- `beta`: Trust formation rate from verification effort
- `delta_T`: Trust decay rate

**Payoff parameters:**
- `lam` (λ): Weight on safety debt in utility (0=ignore, >1=danger exceeds benefit)
- `omega` (ω): Direct value of trust/cooperation in utility

## Initial Conditions

State vector y0 = [K_US, K_CN, K_EU, S_US, S_CN, S_EU, T]

Example (hardcoded in CST.py, lines 681-694):
- K0 normalized relative to US: [1.0, 0.03, 0.0036] (representing Grok 4, Qwen 3 Max, Mistral Large 2)
- S0: [0.2, 0.12, 0.15]
- T0: 0.1

Can also load from JSON calibration file (`calibration_from_real_data.json`) with real compute estimates.

## Outputs

**Plots in `plots/` directory:**
- `CST_<policy_mode>.png`: State trajectories (K, S, T, debt)
- `CST_actions_<policy_mode>.png`: Action trajectories (aX, aS, aV)

**Console output:**
- Final state values
- Final actions
- Action budget verification (sum should equal 1.0)

---

## How to Run

1. Install dependencies:
```powershell
pip install numpy scipy matplotlib
```

2. Run simulation:
```powershell
python CST.py
```

3. Configure:
   - Set `use_calibration_file = True/False` (line 648) to load JSON parameters or use hardcoded values
   - Set `policy_mode = "best_response" / "bostrom_minimal" / "scenario"` (lines 702-703)
   - Modify `Params` fields (lines 664-675) or simulation horizon `tf` (line 697)

4. Best response options (lines 705-728):
   - `action_grid`: Discretization fineness (e.g., [0.0, 0.1, ..., 1.0])
   - `use_lookahead`: Strategic forward simulation vs single-step payoff
   - `lookahead_years`: Planning horizon (e.g., 1-2 years)
   - `discount_rate`: Exponential discount for future payoffs
   - `phase1_only_lookahead`: Simulate agents ignoring exponential growth in projections

## Implementation Notes

- **ODE integration:** `solve_ivp` with RK45 (default), adaptive stepping, rtol=1e-6, atol=1e-8
- **Action selection:** Discrete grid search with budget constraint aX + aS + aV = 1
- **Nash convergence:** Randomized player ordering + warm start caching for stability
- **Smooth phase transition:** tanh blending avoids discontinuities at K_threshold
- **State unpacking:** Flat array y → State(K, S, T) for readability

## Key Assumptions

1. **No depreciation/decay:** K and S accumulate indefinitely (only T decays)
2. **Deterministic dynamics:** No stochastic shocks or incidents
3. **Symmetric structure:** All blocs use same dynamics (asymmetry only in γ, initial conditions, and trust growth rate)
4. **Budget constraint:** Enforced in best-response/Bostrom policies, not in scenario policy
5. **Global trust:** Single scalar T (not bilateral trust matrix)
6. **Threshold uncertainty:** Optional `phase1_only_lookahead` simulates agents not knowing when exponential growth begins
7. **Continuous actions:** Policy evaluated at each ODE step (not discrete decision points)

## Limitations and Extensions

**Current scope:**
- 3-player symmetric game
- Continuous-time approximation (no explicit timesteps in dynamics)
- No resource constraints, energy bottlenecks, or political feasibility limits
- Safety debt is diagnostic only unless λ > 0

**Possible extensions:**
- Stochastic capability breakthroughs or safety failures
- Asymmetric bloc objectives and constraint sets
- Multi-stage commitment mechanisms or treaties
- Calibration to historical AI progress data
- Absorbing states (AGI threshold reached, catastrophic failure)



## License

Apache 2.0
