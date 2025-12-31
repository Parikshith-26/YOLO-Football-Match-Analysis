"""Perspective transformation for converting pixel coordinates to meters."""
import numpy as np
import cv2


class ViewTransformer:
    """Transform pixel coordinates to real-world coordinates."""
    
    def __init__(self):
        """Initialize view transformer."""
        court_width = 68
        court_length = 23.32
        
        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ])
        
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])
        
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)
        
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
    
    
    def transform_points_batch(self, points):
        """
        Transform multiple points at once (vectorized).
        
        Args:
            points: Nx2 array of (x,y) points
            
        Returns:
            Nx2 array of transformed points
        """
        points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points_array, self.perspective_transformer)
        return transformed.reshape(-1, 2)
    
    def add_transformed_position_to_tracks_fast(self, tracks):
        """
        Vectorized transformation for all frames at once.
        """
        for obj_type, obj_tracks in tracks.items():
            # Collect all positions for this object type
            positions = []
            valid_indices = []
            
            for frame_num, frame_tracks in enumerate(obj_tracks):
                for track_id, track_info in frame_tracks.items():
                    pos = track_info.get('position_adjusted')
                    if pos:
                        positions.append(pos)
                        valid_indices.append((frame_num, track_id))
            
            if not positions:
                continue
            
            # Batch transform all at once
            positions_array = np.array(positions, dtype=np.float32)
            transformed_all = self.transform_points_batch(positions_array)
            
            # Assign back
            for i, (frame_num, track_id) in enumerate(valid_indices):
                trans_pos = transformed_all[i].tolist()
                tracks[obj_type][frame_num][track_id]['position_transformed'] = trans_pos
        
        return tracks

    # Backwards-compatible wrapper
    def add_transformed_position_to_tracks(self, tracks):
        """
        Backwards-compatible API: delegate to vectorized implementation.
        Preserves original method name expected elsewhere in the codebase.
        """
        return self.add_transformed_position_to_tracks_fast(tracks)