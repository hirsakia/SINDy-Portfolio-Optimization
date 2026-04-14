"""
SINDy Portfolio Optimization — v8
Visualization: cumulative return curves and weight area charts.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_results(all_results, split="2015-01-01"):
    """Plot cumulative returns and weight histories for all universes at a given OOS split."""
    n_u = len(all_results)
    if n_u == 0:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(2, n_u, figsize=(6 * n_u, 8))
    if n_u == 1:
        axes = axes.reshape(2, 1)

    for col, (uname, ur) in enumerate(all_results.items()):
        oos = ur["oos"].get(split, {})
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # Top: cumulative returns
        for sname, sr in oos.items():
            if "curve" in sr:
                ax_top.plot(sr["curve"], label=f"{sname} (S={sr['sharpe']:.2f})")
        ax_top.set_title(f"{uname}\nOOS from {split}")
        ax_top.legend(fontsize=7)
        ax_top.set_ylabel("Cumulative Return")
        ax_top.grid(True, alpha=0.3)

        # Bottom: SINDy CVaR weight history (if available)
        if "SINDy CVaR" in oos and "weights_hist" in oos["SINDy CVaR"]:
            wh = oos["SINDy CVaR"]["weights_hist"]
            if wh:
                dates = [d for d, _ in wh]
                weights_arr = np.array([w for _, w in wh])
                ax_bot.stackplot(range(len(dates)), weights_arr.T, alpha=0.7)
                ax_bot.set_title(f"SINDy CVaR Weights — {uname}")
                ax_bot.set_ylabel("Weight")
                ax_bot.set_xlim(0, len(dates) - 1)
                # Show a few date labels
                n_labels = min(5, len(dates))
                step = max(1, len(dates) // n_labels)
                ax_bot.set_xticks(range(0, len(dates), step))
                ax_bot.set_xticklabels(
                    [str(dates[i].date()) if hasattr(dates[i], 'date') else str(dates[i])
                     for i in range(0, len(dates), step)],
                    rotation=45, fontsize=7,
                )
                ax_bot.grid(True, alpha=0.3)
        else:
            ax_bot.text(0.5, 0.5, "No SINDy CVaR data", ha='center', va='center',
                        transform=ax_bot.transAxes)

    plt.tight_layout()
    plt.show()
