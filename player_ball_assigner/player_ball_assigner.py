"""Player ball assignment logic."""
import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    """Assign ball possession to players."""
    
    def __init__(self):
        """Initialize player ball assigner."""
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players, ball_bbox):
        """
        Assign ball to the closest player.
        
        Args:
            players: Dictionary of player tracks
            ball_bbox: Ball bounding box
            
        Returns:
            Player ID who has the ball, or -1 if no player is close enough
        """
        ball_position = get_center_of_bbox(ball_bbox)
        
        minimum_distance = 99999
        assigned_player = -1
        
        for player_id, player in players.items():
            player_bbox = player['bbox']
            
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)
            
            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id
        
        return assigned_player
