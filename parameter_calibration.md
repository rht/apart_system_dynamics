# Parameter Calibration Guide: Translating Model Parameters to Real-World Metrics

## Overview
This guide shows how to map the abstract CST.py parameters to measurable, real-world AI development metrics as of 2025.

## 🆕 **Automated Real-Data Pipeline (November 2025)**

We now have an **automated data pipeline** that fetches real-world data and calibrates parameters:

### Quick Start:
```bash
# 1. Fetch real data from public sources
python fetch_real_data.py

# 2. Calibrate model parameters from real data
python calibrate_from_real_data.py

# Output: calibration_from_real_data.json with all 17 parameters
```

### Data Sources (Live):
- **Epoch AI**: 507 models with training compute (https://epochai.org/data/)
- **arXiv**: 366+ AI safety papers (via arXiv API)
- **GitHub**: Open source model releases (via GitHub API)
- **World Bank**: R&D spending data (via World Bank API)

### ⚠️ **TRANSPARENCY: What's Real vs. Estimated**

**Directly from Real Data (High Confidence):**
- ✅ **K0 (capabilities)**: Computed from actual Epoch AI training FLOPs
- ✅ **alpha (growth rate)**: Measured from real model trajectory (2023-2025)
- ✅ **K_threshold (AGI)**: Extrapolated from current frontier models
- ✅ **T0 (trust)**: Based on actual open source release rates

**Estimated from Indirect Proxies (Medium Confidence):**
- 🟡 **gamma (safety growth)**: Inferred from arXiv safety paper growth rate
- 🟡 **eta (spillover)**: Scaled from trust level (assumes spillover ∝ trust)

**Literature-Based Assumptions (Low Confidence - MADE UP):**
- 🔴 **S0 (safety levels)**: Assumed 1.5% of capability (NO direct data on safety investment)
- 🔴 **beta_dim (diminishing returns)**: Hardcoded 0.35 (based on reported compute bottlenecks)
- 🔴 **theta (safety effectiveness)**: Hardcoded 0.7 (no empirical measure exists)
- 🔴 **beta (trust build)**: Hardcoded 0.3 (no direct observation of trust dynamics)
- 🔴 **delta_T (trust decay)**: Hardcoded 0.2 (geopolitical judgment call)
- 🔴 **lambda (safety concern)**: Hardcoded 0.4 (qualitative assessment)

### Why Some Parameters Can't Be Measured:

**Missing Public Data:**
- Safety team sizes (companies don't publish)
- Actual safety R&D budgets (proprietary)
- Red-teaming effectiveness (rarely published)

**Inherently Unobservable:**
- How much safety cancels capability risk (theta) - no ground truth
- Trust build/decay rates (beta, delta_T) - only proxy indicators
- Decision-maker preferences (lambda) - private information

**Recommendation:** Run **sensitivity analysis** on the red-flagged (🔴) parameters to understand which assumptions matter most for your conclusions.

### Latest Calibration (November 2025):
Based on real data as of Nov 1, 2025:
- **α = 0.50** ✅ (from actual capability growth trajectory)
- **K_threshold = 20.9** ✅ (extrapolated from current frontier)
- **K0 = [13.1, 9.8, 9.6]** ✅ (US, China, EU from Epoch AI compute data)
- **T0 = 0.50** ✅ (from 100% open source rate in sample)
- **gamma = 1.0** 🟡 (from safety paper exponential growth)
- **beta_dim = 0.35** 🔴 (ASSUMED from literature)
- **theta = 0.7** 🔴 (ASSUMED - no empirical basis)
- **eta = 0.4** 🟡 (scaled from trust)
- **beta = 0.3** 🔴 (ASSUMED)
- **delta_T = 0.2** 🔴 (ASSUMED)
- **lambda = 0.4** 🔴 (ASSUMED)

See `calibration_from_real_data.json` for full calibrated parameters.

---

## Manual Calibration Approach (If Data Unavailable)

The sections below describe how to manually estimate parameters if the automated pipeline is unavailable or you want to understand the methodology.

---

## 1. Capability Parameters

### **`alpha` - Capability Growth Efficiency**
**What it means**: How much capability improves per unit of acceleration effort.

**Real-world proxies**:
- **Compute scaling laws**: Current models show ~4x performance for 10x compute (OpenAI scaling laws)
- **Budget efficiency**: Capability gain per $1B invested
- **Timeline**: Months to train next-gen model (GPT-4 took ~4 months)

**Calibration approach**:
```python
# Measure: Capability gain / Investment
# Example: If $1B training run improves benchmark by 10 points
alpha = capability_gain / (compute_fraction * time_invested)

# Historical data:
# GPT-3 (2020): ~$4M training, 175B params
# GPT-4 (2023): ~$100M training, ~1.7T params (estimated)
# Improvement: ~10x params for 25x cost → alpha ≈ 0.4
```

**Suggested range**: `alpha = 0.5 - 2.0`
- Lower bound: Linear scaling only (current regime)
- Upper bound: Some algorithmic efficiency gains

---

### **`K_threshold` - AGI Threshold**
**What it means**: The capability level where recursive self-improvement becomes possible.

**Real-world proxies**:
- **Benchmark performance**: When AI matches human expert performance across domains
- **Economic productivity**: AI can perform 50%+ of cognitive work
- **Meta-learning capability**: AI can improve its own architecture/training

**Calibration approach**:
```python
# Normalize current capabilities to 0-100 scale:
# GPT-3.5 (2023): K ≈ 5
# GPT-4 (2024): K ≈ 8
# Claude 3.5, Gemini 1.5 (2024-25): K ≈ 10
# Human-level AGI: K ≈ 20-30
# Superintelligence: K > 30

# Suggested: K_threshold = 15-25
# This is when AI can meaningfully contribute to AI research
```

**Key indicators for threshold**:
- AI can autonomously find and fix bugs in AI training code
- AI can propose novel architectures that work
- AI researchers use AI for >50% of their work
- AI can run experiments and interpret results

**Suggested range**: `K_threshold = 15 - 25`

---

### **`beta_dim` - Diminishing Returns Strength**
**What it means**: How quickly progress slows before AGI threshold (higher = stronger diminishing returns).

**Real-world proxies**:
- **Data bottleneck**: Running out of quality training data
- **Compute limitations**: Hardware constraints on training
- **Architectural limits**: Current transformer paradigm hitting walls

**Calibration approach**:
```python
# Observe: Progress from GPT-4 → GPT-5 vs GPT-3 → GPT-4
# If each generation requires 3-5x more compute for 50% improvement
# Then beta_dim ≈ 0.2 - 0.5

# Formula: dK/dt = alpha * effort / (1 + beta_dim * K)
# At K=10 with beta_dim=0.3: growth is 1/(1+3) = 25% of baseline
```

**Current evidence (2024-25)**:
- Frontier model improvements slowing down
- Requiring more compute per capability gain
- Data quality becoming critical bottleneck

**Suggested range**: `beta_dim = 0.2 - 0.5`

---

## 2. Safety Parameters

### **`gamma` - Safety Growth Efficiency**
**What it means**: How much safety improves per unit of safety effort.

**Real-world proxies**:
- **RLHF effectiveness**: Reduction in harmful outputs per training iteration
- **Alignment tax**: Safety work as % of total effort
- **Red-team success rate**: Reduction in successful jailbreaks

**Calibration approach**:
```python
# Measure safety improvements in practice:
# GPT-3.5 → GPT-4: ~40% reduction in harmful content (per OpenAI)
# Claude 2 → Claude 3: ~50% improvement on safety benchmarks
# Effort: ~10-20% of total training budget on safety

# gamma ≈ safety_improvement / safety_effort_fraction
# Example: 0.4 improvement / 0.15 effort = gamma ≈ 2.7

# But diminishing returns exist, so use conservative estimate
```

**Suggested range**: `gamma = 0.3 - 1.0`
- Lower: Safety work is less efficient than capability work
- Upper: Safety scales well with investment

---

### **`theta` - Safety Effectiveness Factor**
**What it means**: How much safety mitigates capability risk (in safety debt formula: `debt = K - theta * S`).

**Real-world meaning**:
- **theta = 1.0**: Perfect safety - 1 unit safety cancels 1 unit capability risk
- **theta = 0.5**: Need 2x safety to cancel capability risk
- **theta = 2.0**: Safety is super-effective (1 unit cancels 2 units risk)

**Calibration approach**:
```python
# Think of it as: "How many safety researchers per capability researcher?"
# Current ratios at major labs:
# - OpenAI: ~20-30% on alignment (theta ≈ 0.3-0.5)
# - Anthropic: ~40-50% on safety (theta ≈ 0.6-0.8)
# - Academic consensus: Need 1:1 ratio minimum (theta ≈ 1.0)

# Conservative estimate: theta = 0.5 - 1.0
# Optimistic: theta = 1.0 - 1.5
```

**Suggested range**: `theta = 0.5 - 1.2`

---

## 3. Cooperation Parameters

### **`eta` - Safety Spillover Strength**
**What it means**: How much safety knowledge spreads between blocs through trust.

**Real-world proxies**:
- **Research publication rate**: Papers published vs kept secret
- **Model release practices**: Open vs closed models
- **Safety standard adoption**: Voluntary adoption of safety protocols

**Calibration approach**:
```python
# Measure information sharing:
# - Open research era (2015-2020): eta ≈ 0.5-0.8 (high sharing)
# - Current era (2024-25): eta ≈ 0.2-0.4 (more secretive)
# - Cold war scenario: eta ≈ 0.0-0.1 (minimal sharing)

# Formula: dS_i/dt includes term: eta * T * avg_other_S
# If trust T=0.5 and others have S=1.0, spillover = eta * 0.5 * 1.0
```

**Current indicators**:
- Meta/LLaMA: Open weights (high spillover)
- OpenAI GPT-4: Closed (low spillover)
- Anthropic Claude: Research papers but closed model (medium)

**Suggested range**: `eta = 0.1 - 0.5`
- Lower: Arms race, secretive
- Upper: Strong cooperation

---

### **`beta` - Trust Build Rate**
**What it means**: How fast trust accumulates from verification/cooperation efforts.

**Real-world proxies**:
- **Treaty effectiveness**: Time to establish international AI agreements
- **Verification regime**: Compute monitoring, model registries
- **Track record**: Years of cooperation needed to build trust

**Calibration approach**:
```python
# Think in terms of half-life:
# beta = 0.1: Takes ~10 time units of full verification effort to build trust
# beta = 1.0: Trust builds quickly from cooperation

# Real-world analogs:
# - Nuclear arms treaties: Took decades (beta ≈ 0.05-0.1)
# - Climate agreements: Medium trust (beta ≈ 0.2-0.3)
# - AI safety orgs (Partnership on AI): Few years (beta ≈ 0.3-0.5)
```

**Suggested range**: `beta = 0.1 - 0.5`
- Lower: Hard to build trust (default assumption)
- Upper: Quick trust building (optimistic)

---

### **`delta_T` - Trust Decay Rate**
**What it means**: How fast trust erodes without active maintenance.

**Real-world proxies**:
- **Geopolitical tensions**: US-China tech competition
- **Defection events**: Broken commitments, leaked models
- **Media cycles**: Public perception changes

**Calibration approach**:
```python
# Half-life of trust:
# delta_T = 0.1: Trust decays slowly (half-life ~7 time units)
# delta_T = 0.5: Trust decays quickly (half-life ~1.4 time units)

# Examples:
# - Stable alliances (NATO): delta_T ≈ 0.05-0.1
# - Business partnerships: delta_T ≈ 0.2-0.4
# - AI race context (high stakes): delta_T ≈ 0.3-0.6
```

**Suggested range**: `delta_T = 0.1 - 0.5`

---

## 4. Payoff Parameter

### **`lam` (lambda) - Safety Debt Concern**
**What it means**: How much actors care about safety debt in decision-making.

**Real-world proxies**:
- **Public pressure**: Regulatory scrutiny, media attention
- **Existential risk awareness**: Board-level concern about catastrophic risk
- **Liability**: Legal/financial consequences of accidents

**Calibration approach**:
```python
# lam = 0.0: Pure race - only care about capability lead
# lam = 0.5: Balanced - trade off capability for safety
# lam = 1.0: Safety-first - equal weight to risk reduction

# Current estimates by actor:
# - Meta/Google (2023-24): lam ≈ 0.1-0.3 (capability-focused)
# - OpenAI/Anthropic: lam ≈ 0.4-0.6 (safety-conscious)
# - Academia/safety orgs: lam ≈ 0.7-1.0 (safety-first)

# Trajectory: lam likely increases as capabilities approach AGI
```

**Suggested range**: `lam = 0.2 - 0.8`
- Use time-dependent: `lam(t) = 0.2 + 0.6 * (K_avg / K_threshold)`

---

## 5. Initial Conditions (as of 2025)

### **Capability (K)**
```python
# Benchmark-based normalization:
# K = 0: No AI capability
# K = 10: Current frontier models (GPT-4, Claude 3.5, Gemini)
# K = 20: Human-level AGI
# K = 30+: Superintelligence

# November 2025 estimates:
K0 = np.array([
    12.0,  # US (OpenAI, Anthropic, Meta) - slightly ahead
    10.0,  # China (Alibaba, Baidu, DeepSeek) - catching up
    8.0    # EU (Mistral, DeepMind) - smaller but advanced
])
```

**Calibration method**:
- Aggregate benchmark performance (MMLU, HumanEval, etc.)
- Compute capacity (training flops)
- Deployment scale (API usage)

### **Safety (S)**
```python
# Safety investment as % of capability:
# S/K = 0.1 means 10% safety investment
# S/K = 0.5 means 50% safety investment (very high)

# Current estimates:
S0 = np.array([
    0.15,  # US: ~12-15% on safety (OpenAI's superalignment team was 20%)
    0.08,  # China: Lower published safety focus
    0.12   # EU: Strong regulatory focus (AI Act)
])

# Absolute values scaled to K
S0 = K0 * np.array([0.015, 0.008, 0.015])
```

### **Trust (T)**
```python
# T = 0.0: Complete distrust (cold war)
# T = 0.5: Moderate cooperation (current?)
# T = 1.0: Full cooperation (unlikely)

# November 2025:
T0 = 0.15  # Low trust due to:
           # - US-China tech competition
           # - Export controls on chips
           # - But some cooperation (Partnership on AI, academic exchange)
```

---

## 6. Time Units Calibration

**Model time vs real time**:
```python
# One time unit ≈ 6-12 months of AI development
# Rationale:
# - Major model releases: ~6-12 months apart
# - Policy cycles: Annual reviews
# - Research iteration: 6-12 months for major advances

# Simulation horizons:
t_span = (0, 20)  # 10-20 years of AI development
t_span = (0, 10)  # 5-10 years (to AGI?)
```

---

## 7. Example: Realistic Calibration (2025 Baseline)

```python
from CST import Params
import numpy as np

# Conservative/realistic parameters based on current evidence
params_realistic = Params(
    # Capability
    alpha=1.0,           # Moderate scaling efficiency
    K_threshold=18.0,    # AGI at ~1.8x current frontier
    beta_dim=0.35,       # Significant diminishing returns currently
    
    # Safety
    gamma=0.5,           # Safety work is half as efficient as capability
    theta=0.7,           # Need ~1.4x safety to offset capability
    
    # Cooperation
    eta=0.25,            # Moderate spillover (some open research)
    beta=0.3,            # Medium trust building rate
    delta_T=0.2,         # Moderate trust decay
    
    # Incentives
    lam=0.4,             # Moderate safety concern (growing over time)
)

# Initial conditions (November 2025)
K0 = np.array([12.0, 10.0, 8.0])   # US ahead, China close, EU behind
S0 = np.array([0.18, 0.08, 0.12])  # US & EU more safety-focused
T0 = 0.15                           # Low but non-zero trust

# Time: 10 years (20 time units @ 6 months per unit)
t_span = (0, 20)
```

---

## 8. Sensitivity Analysis Recommendations

**Which parameters matter most?**
1. **`K_threshold`**: Determines when explosive growth starts (critical!)
2. **`alpha`**: Controls overall race speed
3. **`eta` and `beta`**: Determine if cooperation can work
4. **`lam`**: Whether safety is prioritized

**Uncertainty analysis**:
```python
# Run scenarios with:
# - Optimistic: High cooperation (eta=0.5, beta=0.5, low delta_T)
# - Pessimistic: Arms race (eta=0.1, beta=0.2, high delta_T)
# - Realistic: Mix of both (parameters above)
```

---

## 9. Data Sources for Calibration

**Publicly available data**:
1. **Compute estimates**: Epoch AI database
2. **Model capabilities**: HELM, BIG-bench, OpenLLM leaderboard
3. **Safety investment**: Company reports, academic papers
4. **Cooperation indicators**: 
   - Partnership on AI activities
   - Model release practices (open vs closed)
   - International AI governance discussions
   - Export control policies

**Update frequency**: Recalibrate every 6-12 months as new models release

---

## 10. Validation Approach

**Backtesting** (2020-2025):
```python
# Set parameters based on 2020 state
# Run simulation forward
# Compare to actual 2025 state
# Adjust parameters to match reality
```

**Forward validation**:
- Track model: Does it predict next 6-12 months?
- Adjust based on surprises (breakthrough algorithms, policy changes)

---

## Summary Table

| Parameter | Range | Current Best Estimate | Key Uncertainty |
|-----------|-------|----------------------|-----------------|
| `alpha` | 0.5-2.0 | 1.0 | Algorithmic breakthroughs |
| `K_threshold` | 15-25 | 18 | Definition of AGI |
| `beta_dim` | 0.2-0.5 | 0.35 | Data/compute bottlenecks |
| `gamma` | 0.3-1.0 | 0.5 | Safety work efficiency |
| `theta` | 0.5-1.2 | 0.7 | How much safety is enough |
| `eta` | 0.1-0.5 | 0.25 | Geopolitical dynamics |
| `beta` | 0.1-0.5 | 0.3 | Trust formation speed |
| `delta_T` | 0.1-0.5 | 0.2 | Geopolitical stability |
| `lam` | 0.2-0.8 | 0.4 (→0.8) | Growing safety awareness |

**Next steps**: Run sensitivity analysis on these ranges to see which parameters drive outcomes most strongly.

---

## 11. Using the Automated Pipeline

### Fetch Real Data:
```bash
python fetch_real_data.py
```
This downloads:
- 500+ AI models from Epoch AI with training compute
- Safety research papers from arXiv
- Open source releases from GitHub
- R&D spending from World Bank

Output: `fetched_real_data.json`

### Calibrate Parameters:
```bash
python calibrate_from_real_data.py
```
This processes the fetched data and outputs:
- 9 calibrated model parameters
- 7 initial conditions (K0, S0, T0)

Output: `calibration_from_real_data.json`

### Use in Simulations:
```python
import json
import numpy as np
from CST import Params, simulate

# Load calibrated parameters
with open('calibration_from_real_data.json', 'r') as f:
    cal = json.load(f)

# Create Params object
params = Params(**cal['params'])

# Initial conditions
K0 = np.array(cal['initial_conditions']['K0'])
S0 = np.array(cal['initial_conditions']['S0'])
T0 = cal['initial_conditions']['T0']
y0 = np.concatenate([K0, S0, [T0]])

# Run simulation
sol = simulate(t_span=(0, 20), y0=y0, params=params, ...)
```

### Update Frequency:
- **Monthly**: Re-run `fetch_real_data.py` as new models are announced
- **Quarterly**: Review and validate calibration against actual events
- **Annually**: Major recalibration as AI capabilities evolve
