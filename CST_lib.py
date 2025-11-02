import json

import matplotlib.pyplot as plt
import numpy as np

US, CN, EU = 0, 1, 2  # useful named indices

def plot_results(t_eval, K_path, params, suffix=""):
    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, K_path[US], label="US K")
    plt.plot(t_eval, K_path[CN], label="CN K")
    plt.plot(t_eval, K_path[EU], label="EU K")
    plt.axhline(
        params.K_threshold, label="K threshold", color="black", linestyle="dashed"
    )
    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("Capability (K)")
    plt.title(f"Capability Evolution ({suffix})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/CST_main_{suffix}.png")
    plt.close()


def plot_actions(t_eval, aX_path, aS_path, aV_path, suffix=""):
    """
    Plot actions over time using step plots.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot aX (acceleration)
    axes[0].step(t_eval, aX_path[US], label="US", where="post", linewidth=2)
    axes[0].step(t_eval, aX_path[CN], label="CN", where="post", linewidth=2)
    axes[0].step(t_eval, aX_path[EU], label="EU", where="post", linewidth=2)
    axes[0].set_ylabel("Acceleration Effort (aX)")
    axes[0].set_title("Action Evolution Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.05, 1.05])

    # Plot aS (safety)
    axes[1].step(t_eval, aS_path[US], label="US", where="post", linewidth=2)
    axes[1].step(t_eval, aS_path[CN], label="CN", where="post", linewidth=2)
    axes[1].step(t_eval, aS_path[EU], label="EU", where="post", linewidth=2)
    axes[1].set_ylabel("Safety Effort (aS)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.05, 1.05])

    # Plot aV (verification)
    axes[2].step(t_eval, aV_path[US], label="US", where="post", linewidth=2)
    axes[2].step(t_eval, aV_path[CN], label="CN", where="post", linewidth=2)
    axes[2].step(t_eval, aV_path[EU], label="EU", where="post", linewidth=2)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Verification Effort (aV)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-0.05, 1.05])

    plt.suptitle(suffix)
    plt.tight_layout()
    fname = f"plots/CST_actions_{suffix}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved action plot to {fname}")


def load_calibration_from_json(json_path, Params):
    """
    Load model parameters and initial conditions from a JSON calibration file.

    Args:
        json_path: Path to the JSON file containing calibration data

    Returns:
        tuple of (Params, y0) where y0 is the initial state vector
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Load parameters
    params_dict = data["params"]
    params = Params(
        alpha=params_dict["alpha"],
        gamma=params_dict["gamma"],
        eta=params_dict["eta"],
        beta=params_dict["beta"],
        delta_T=params_dict["delta_T"],
        theta=params_dict["theta"],
        K_threshold=params_dict["K_threshold"],
        beta_dim=params_dict["beta_dim"],
        transition_width=params_dict.get(
            "transition_width", 1.0
        ),  # default if not in JSON
        lam=params_dict["lam"],
    )

    # Load initial conditions
    ic = data["initial_conditions"]
    K0 = np.array(ic["K0"])
    S0 = np.array(ic["S0"])
    T0 = ic["T0"]
    y0 = np.concatenate([K0, S0, [T0]])

    # Print metadata if available
    if "metadata" in data:
        meta = data["metadata"]
        print(f"Loaded calibration from: {json_path}")
        print(f"  Calibration date: {meta.get('calibration_date', 'N/A')}")
        print(f"  Target year: {meta.get('target_year', 'N/A')}")
        print(f"  Data sources: {', '.join(meta.get('data_sources', []))}")

    return params, y0
