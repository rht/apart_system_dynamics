import numpy as np
import matplotlib.pyplot as plt
from CST_ag import Params, simulate, simple_scenario_policy_builder, unpack_state

# Let's run a couple scenarios and plot the full trajectories to see what's happening

t0, tf = 0.0, 20.0  # Shorter time to see detail
t_eval = np.linspace(t0, tf, 201)

scenarios_to_check = {
    'baseline': {
        'K0': np.array([5.0, 4.0, 3.0]),
        'S0': np.array([0.5, 0.4, 0.3]),
        'T0': 0.1,
        'params': Params(alpha=1.0, gamma=0.5, eta=0.2, beta=0.3, delta_T=0.1, 
                        theta=0.8, K_threshold=8.0, beta_dim=0.3, lam=0.0),
        'mode': 'arms_race'
    },
    'hard_alignment': {
        'K0': np.array([5.0, 4.0, 3.0]),
        'S0': np.array([0.5, 0.4, 0.3]),
        'T0': 0.1,
        'params': Params(alpha=1.0, gamma=0.5, eta=0.2, beta=0.3, delta_T=0.1,
                        theta=0.8, K_threshold=12.0, beta_dim=0.3, lam=0.0),
        'mode': 'arms_race'
    },
    'treaty': {
        'K0': np.array([5.0, 4.0, 3.0]),
        'S0': np.array([0.5, 0.4, 0.3]),
        'T0': 0.1,
        'params': Params(alpha=1.0, gamma=0.5, eta=0.2, beta=0.3, delta_T=0.1,
                        theta=0.8, K_threshold=8.0, beta_dim=0.3, lam=0.0),
        'mode': 'treaty'
    }
}

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Detailed Trajectory Check', fontsize=14, fontweight='bold')

colors = ['blue', 'red', 'green']
player_names = ['US', 'CN', 'EU']

for col_idx, (name, scenario) in enumerate(scenarios_to_check.items()):
    print(f"\n{'='*60}")
    print(f"Checking: {name.upper()}")
    print(f"{'='*60}")
    
    y0 = np.concatenate([scenario['K0'], scenario['S0'], [scenario['T0']]])
    policy_fn = simple_scenario_policy_builder(mode=scenario['mode'])
    
    # Get the controls to verify what actions are being taken
    test_controls = policy_fn(0, y0)
    print(f"Controls being used:")
    print(f"  aX (acceleration): {test_controls.aX}")
    print(f"  aS (safety):       {test_controls.aS}")
    print(f"  aV (verification): {test_controls.aV}")
    
    sol = simulate(
        t_span=(t0, tf),
        y0=y0,
        params=scenario['params'],
        policy_fn=policy_fn,
        t_eval=t_eval,
    )
    
    K_path = sol.y[0:3, :]
    S_path = sol.y[3:6, :]
    T_path = sol.y[6, :]
    
    # Check when US crosses threshold 50
    us_k = K_path[0, :]
    cross_idx = np.where(us_k >= 50.0)[0]
    if len(cross_idx) > 0:
        cross_time = sol.t[cross_idx[0]]
        cross_k = us_k[cross_idx[0]]
        cross_s = S_path[0, cross_idx[0]]
        print(f"\nUS reaches K=50 at t={cross_time:.2f} years")
        print(f"  K={cross_k:.2f}, S={cross_s:.3f}, S/K={cross_s/cross_k:.4f} ({100*cross_s/cross_k:.2f}%)")
    else:
        print(f"\nUS does not reach K=50 in {tf} years")
        print(f"  Final K={us_k[-1]:.2f}, Final S={S_path[0,-1]:.3f}")
    
    # Plot capability
    ax1 = axes[0, col_idx]
    for i in range(3):
        ax1.plot(sol.t, K_path[i, :], color=colors[i], label=player_names[i], linewidth=2)
    ax1.axhline(y=scenario['params'].K_threshold, color='orange', linestyle='--', 
                label=f"AGI threshold={scenario['params'].K_threshold}")
    ax1.axhline(y=50, color='red', linestyle='--', label='Win threshold=50')
    ax1.set_ylabel('Capability (K)')
    ax1.set_title(f'{name.replace("_", " ").title()}')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 50)
    
    # Plot safety
    ax2 = axes[1, col_idx]
    for i in range(3):
        ax2.plot(sol.t, S_path[i, :], color=colors[i], label=player_names[i], linewidth=2)
    ax2.set_ylabel('Safety (S)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 3)
    
    # Plot safety ratio for US
    ax3 = axes[2, col_idx]
    us_safety_ratio = S_path[0, :] / K_path[0, :]
    ax3.plot(sol.t, 100 * us_safety_ratio, color='purple', linewidth=2)
    ax3.set_ylabel('US Safety/Capability %')
    ax3.set_xlabel('Time (years)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 20)
    ax3.set_ylim(0, 100)
    ax3.axhline(y=5, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plots/trajectory_verification.png', dpi=150, bbox_inches='tight')
print(f"\n{'='*60}")
print("Plot saved to plots/trajectory_verification.png")

