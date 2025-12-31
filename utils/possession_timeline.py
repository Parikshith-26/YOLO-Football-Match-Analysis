# utils/possession_timeline.py
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend("Agg")

def plot_possession_timeline(team_control, save_path):
    """
    Creates clean rectangular-style possession blocks:
    - Green = Team 1
    - Blue = Team 2
    - Matches the screenshot style provided
    """
    arr = np.array(team_control, dtype=int)

    if arr.size == 0:
        fig = plt.figure(figsize=(10, 2))
        plt.text(0.5, 0.5, "No possession data", ha="center", va="center")
        plt.axis("off")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return

    x = np.arange(len(arr))

    # Identify blocks (rectangles)
    team1_mask = arr == 1
    team2_mask = arr == 2

    # Create figure similar to screenshot
    fig, ax = plt.subplots(figsize=(14, 2.8))

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Draw TEAM 1 rectangles
    start = None
    for i in range(len(arr)):
        if team1_mask[i] and start is None:
            start = i
        if (not team1_mask[i] or i == len(arr) - 1) and start is not None:
            end = i if not team1_mask[i] else i + 1
            ax.fill_between(range(start, end), 0.6, 1.3,
                            color="#00cc66", alpha=0.45)
            start = None

    # Draw TEAM 2 rectangles
    start = None
    for i in range(len(arr)):
        if team2_mask[i] and start is None:
            start = i
        if (not team2_mask[i] or i == len(arr) - 1) and start is not None:
            end = i if not team2_mask[i] else i + 1
            ax.fill_between(range(start, end), 1.7, 2.4,
                            color="#4ac0ff", alpha=0.45)
            start = None

    # Y axis labels like screenshot
    ax.set_yticks([1.0, 2.0])
    ax.set_yticklabels(["Team 1", "Team 2"], fontsize=9)

    ax.set_xlabel("Frames", fontsize=10)
    ax.set_title("Ball Possession Timeline", fontsize=12)

    ax.set_xlim(0, len(arr))
    ax.set_ylim(0.5, 2.6)

    # Remove borders for a clean card look
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()

