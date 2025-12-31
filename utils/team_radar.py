# utils/team_radar.py
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend("Agg")

def team_radar(team_metrics, save_path):
    """
    team_metrics example:
    {
      "labels": ["Possession", "Avg Speed", "Total Distance"],
      "team1": [val1, val2, val3],
      "team2": [val1, val2, val3]
    }
    Values should be raw numbers; function normalizes to max for plotting.
    """
    labels = team_metrics.get("labels", [])
    t1 = np.array(team_metrics.get("team1", []), dtype=float)
    t2 = np.array(team_metrics.get("team2", []), dtype=float)
    if len(labels) == 0 or t1.size == 0 or t2.size == 0:
        fig = plt.figure(figsize=(2,2))
        plt.text(0.5,0.5,"No data",ha="center",va="center")
        plt.axis("off")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return

    # normalize by maximum per metric (avoid zero division)
    max_vals = np.maximum(t1, t2)
    max_vals[max_vals == 0] = 1.0
    t1n = t1 / max_vals
    t2n = t2 / max_vals

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    t1n = np.concatenate([t1n, [t1n[0]]])
    t2n = np.concatenate([t2n, [t2n[0]]])
    angles += angles[:1]

    plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, t1n, color="#00aa44", linewidth=2)
    ax.fill(angles, t1n, color="#00aa44", alpha=0.25)
    ax.plot(angles, t2n, color="#0077cc", linewidth=2)
    ax.fill(angles, t2n, color="#0077cc", alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_yticklabels([])
    plt.title("Team Comparison", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
