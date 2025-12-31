"""Team assignment using KMeans clustering."""
from sklearn.cluster import KMeans
import numpy as np


class TeamAssigner:
    """Assign players to teams based on shirt colors."""

    def __init__(self):
        """Initialize team assigner."""
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_color_cache = {}

    def get_player_color(self, frame, bbox, player_id=None):
        """
        Get dominant color of player's shirt with caching and safety checks.

        Args:
            frame: Video frame
            bbox: Player bounding box
            player_id: Optional player id for caching

        Returns:
            Dominant color (BGR) as numpy array
        """
        if player_id and player_id in self.player_color_cache:
            return self.player_color_cache[player_id]

        x1, y1, x2, y2 = map(int, bbox)
        # Clip bbox to frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return np.array([0, 0, 0])

        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return np.array([0, 0, 0])

        # Take top half where shirt color is likely present
        top_half = image[: max(1, image.shape[0] // 2), :]

        # If ROI too small, just take mean color
        if top_half.size < 10:
            color = top_half.reshape(-1, 3).mean(axis=0)
            color = np.array(color)
            if player_id:
                self.player_color_cache[player_id] = color
            return color

        # KMeans: reasonable parameters, no duplicate keyword args
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=3, max_iter=50)
        pixels = top_half.reshape(-1, 3)
        kmeans.fit(pixels)

        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])

        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        if player_id:
            self.player_color_cache[player_id] = player_color

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Assign team colors based on player detections from a single reference frame.
        """
        player_colors = []
        ids = []

        for pid, pdata in player_detections.items():
            bbox = pdata.get("bbox")
            if bbox is None:
                continue
            color = self.get_player_color(frame, bbox, pid)
            player_colors.append(color)
            ids.append(pid)

        if len(player_colors) < 2:
            # Fallback: assign default colors if not enough data
            self.team_colors[1] = np.array([255, 0, 0])
            self.team_colors[2] = np.array([0, 255, 0])
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=100)
        kmeans.fit(np.vstack(player_colors))

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        # Optionally pre-populate player_team_dict for those seen in the reference frame
        for pid, color in zip(ids, player_colors):
            team_id = int(kmeans.predict(np.array(color).reshape(1, -1))[0]) + 1
            self.player_team_dict[pid] = team_id

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Get team assignment for a player.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox, player_id)
        team_id = int(self.kmeans.predict(np.array(player_color).reshape(1, -1))[0]) + 1

        # Manual override if desired (keeps parity with original)
        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id