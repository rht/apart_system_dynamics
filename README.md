# AI Race Simulation (SciPy-only)

## Overview

This prototype models a simplified **AI race** between three blocs — the **USA**, **China**, and the **EU** — who each decide how much to invest in:
- **Safety research (a_S)**: reduces risk but slows progress.
- **Verification/Alignment (a_V)**: ensures safety progress is effective.
- **Acceleration (a_X)**: increases effective compute and capability.

The model is based on the logic of **Nick Bostrom’s “Racing to the Precipice”**, but uses continuous **system dynamics** with resource constraints. It’s implemented entirely using SciPy’s `solve_ivp` for ODE integration.

---

## Conceptual structure

| Symbol | Meaning | Notes |
|---------|----------|-------|
| **Kᵢ** | Capital/Compute stock of bloc *i* | Proxy for available AI development capacity. |
| **Eᵢ** | Efficiency of AI R&D | How much output each unit of compute produces. |
| **Sᵢ** | Safety level | Higher means lower chance of catastrophe. |
| **Tᵢ** | Technology maturity | Captures frontier closeness; saturates near AGI. |
| **Pᵢ** | Political pressure / Willingness | Influences aggressiveness. |
| **R** | Shared global resource pool | Represents energy, compute hardware, data, and attention. |

Each bloc selects actions **a = (a_S, a_V, a_X)** subject to \(a_S + a_V + a_X = 1\).  
These decisions affect the evolution of their internal state and the global resource pool.

---

## Equations (simplified)

```
dK_i/dt = α * a_X_i * R - δ * K_i
dE_i/dt = β * (1 - a_S_i) * log(K_i) - γ * E_i
dS_i/dt = η * a_S_i * (1 - S_i) - ξ * a_X_i * S_i
dT_i/dt = λ * E_i * K_i * (1 - T_i)
dR/dt  = -ρ * Σ_i (a_X_i * K_i)
```

---

## Assumptions

- All blocs start with comparable but slightly differing initial capabilities.
- Safety and acceleration trade off — focusing on one weakens the other.
- Resource use is global: aggressive acceleration by one bloc depletes everyone’s available compute and energy.
- Each bloc acts **myopically** — maximizing short-term progress rather than long-term payoff (a simplification).
- Cooperation scenarios can later be represented by shared objectives or side payments.

---

## Parameters and their interpretations

| Parameter | Meaning | Typical range | Notes |
|------------|----------|----------------|-------|
| **α** | Capital accumulation rate | 0.1–0.5 | How much new compute is added per unit of acceleration effort. |
| **β** | Efficiency gain coefficient | 0.05–0.2 | Effect of R&D and scale on algorithmic efficiency. |
| **γ** | Efficiency decay rate | 0.01–0.1 | Obsolescence of methods. |
| **η** | Safety improvement rate | 0.02–0.2 | Speed at which safety investment improves robustness. |
| **ξ** | Safety erosion from speed | 0.02–0.1 | How fast safety erodes when racing. |
| **λ** | Tech maturation rate | 0.1–0.5 | Rate at which capabilities approach the AGI frontier. |
| **ρ** | Resource depletion rate | 0.001–0.02 | Scarcity of compute, energy, or talent. |
| **δ** | Capital depreciation | 0.01–0.05 | Reflects maintenance or hardware turnover. |

---

## Glossary

- **Bloc**: A major geopolitical actor (US, China, EU).
- **Myopic optimization**: Each bloc chooses short-term best action rather than planning ahead.
- **Cooperation vs Competition**: Can be simulated by changing whether blocs share payoffs or not.
- **Frontier / AGI threshold**: When Tᵢ ≈ 1, AGI-level capability is reached.
- **Safety debt**: The gap between capability growth and safety growth; large debt increases risk.

---

## Outputs

The script produces:
- Time series of compute, safety, and resource levels.
- Saved `.npy` arrays (`times.npy`, `states.npy`, `actions.npy`).
- Figures for each key variable (`fig_compute.png`, `fig_resource.png`, etc.).

---

## How to run

1. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib
   ```

2. Run the simulation:
   ```bash
   python ai_race_sim.py
   ```

3. Inspect results and modify parameters near the top of the script.

---

## Next steps for refinement

1. **Parameter realism**  
   - Calibrate α, β, ρ, etc. using public AI Index and compute-cost datasets.  
   - Replace arbitrary constants with empirically grounded ones.

2. **Decision realism**  
   - Replace myopic strategy with a best-response or reinforcement-learning loop.

3. **Safety-risk mapping**  
   - Link Sᵢ and Tᵢ to a measurable incident probability.

4. **Scenario control**  
   - Add configuration files for cooperation, export controls, or resource shocks.

---

## Citation

This model builds conceptually on:
> Bostrom, Nick. “Racing to the Precipice: A Model of Artificial Intelligence Development.” (2013)

---

## License

MIT License – freely reusable for research and educational purposes.
