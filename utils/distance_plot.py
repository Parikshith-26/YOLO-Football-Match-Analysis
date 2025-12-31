# utils/distance_plot.py
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend("Agg")

def plot_distance_covered(tracks, save_path):
    """
    Creates a clean bar chart showing total distance covered by each player.
    Only shows players who actually moved (distance > 0).
    """

    # -----------------------------------------
    # Extract final distance per player
    # -----------------------------------------
    dist_map = {}

    for frame in tracks.get("players", []):
        for pid, pdata in frame.items():
            d = float(pdata.get("distance", 0.0))
            dist_map[pid] = max(dist_map.get(pid, 0.0), d)

    if not dist_map:
        fig = plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No distance data", ha="center", va="center")
        plt.axis("off")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return

    # Sort by distance
    sorted_items = sorted(dist_map.items(), key=lambda x: x[1], reverse=True)

    # Limit to top 12 players for clarity
    sorted_items = sorted_items[:12]

    players = [f"P{pid}" for pid, _ in sorted_items]
    distances = [round(d, 2) for _, d in sorted_items]

    # -----------------------------------------
    # PLOT
    # -----------------------------------------
    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", color="#cccccc", alpha=0.4, linewidth=0.6)

    # Neon green bars
    ax.bar(players, distances, color="#00cc66", alpha=0.85)

    ax.set_title("Total Distance Covered (Top Players)", fontsize=13, color="#333333", pad=10)
    ax.set_ylabel("Distance (units)", color="#333333")

    # Tilt x labels for readability
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()

