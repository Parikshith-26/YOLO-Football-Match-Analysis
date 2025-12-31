# utils/speed_plot.py
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend("Agg")

def smooth(signal, window=7):
    """Simple moving average smoothing."""
    if len(signal) < window:
        return signal
    return np.convolve(signal, np.ones(window)/window, mode='same')

def plot_player_speed(tracks, save_path):
    """
    Create a clean, modern speed chart:
    - Only top 5 fastest players
    - Smooth curves
    - Neon colors on white background
    - Dashboard-style analytics theme
    """

    # -----------------------------------------
    # Extract speeds per player
    # -----------------------------------------
    players = {}
    for fi, frame in enumerate(tracks.get("players", [])):
        for pid, pdata in frame.items():
            sp = float(pdata.get("speed", 0.0))
            players.setdefault(pid, []).append(sp)

    if not players:
        fig = plt.figure(figsize=(10, 3))
        plt.text(0.5, 0.5, "No speed data", ha="center", va="center")
        plt.axis("off")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return

    # -----------------------------------------
    # Compute max speed for ranking players
    # -----------------------------------------
    max_speeds = {pid: max(vals) if len(vals) else 0 for pid, vals in players.items()}

    # pick top 5 fastest players
    top_players = sorted(max_speeds.items(), key=lambda x: x[1], reverse=True)[:5]
    top_ids = [pid for pid, _ in top_players]

    # -----------------------------------------
    # Modern UI color palette (neon cyan/green/purple/orange)
    # -----------------------------------------
    colors = ["#00cc66", "#00aaff", "#ffaa00", "#cc33ff", "#ff3366"]

    # -----------------------------------------
    # PLOT
    # -----------------------------------------
    plt.figure(figsize=(12, 3.5))
    ax = plt.gca()

    # white clean background
    ax.set_facecolor("#ffffff")

    # soft grey grid
    ax.grid(True, color="#cccccc", alpha=0.4, linewidth=0.6)

    for idx, pid in enumerate(top_ids):
        arr = players[pid]

        # smooth line
        arr_smooth = smooth(np.array(arr), window=9)

        ax.plot(
            arr_smooth,
            label=f"Player {pid}",
            linewidth=2.2,
            color=colors[idx % len(colors)],
            alpha=0.95
        )

    ax.set_title("Player Speed Over Time", fontsize=13, color="#333333", pad=10)
    ax.set_xlabel("Frame", color="#333333")
    ax.set_ylabel("Speed (units)", color="#333333")

    # modern legend
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()

