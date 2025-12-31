"""Camera movement estimation using optical flow."""
import cv2
import numpy as np
import pickle
import os


class CameraMovementEstimator:
    """Estimate camera movement between frames using optical flow."""
    
    def __init__(self, frame):
        """
        Initialize camera movement estimator.
        
        Args:
            frame: First video frame
        """
        self.minimum_distance = 5
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1
        
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )
    
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Adjust track positions based on camera movement.
        
        Args:
            tracks: Dictionary of tracks
            camera_movement_per_frame: List of camera movements per frame
            
        Returns:
            Tracks with adjusted positions
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
        
        return tracks
    
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Get camera movement for each frame.
        
        Args:
            frames: List or generator of video frames
            read_from_stub: Whether to read from cached stub
            stub_path: Path to stub file
            
        Returns:
            List of camera movements per frame
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        camera_movement = []
        
        # We need to iterate through frames.
        # Since we need pairs of frames (old, new), we'll handle the iterator carefully.
        
        iterator = iter(frames)
        try:
            first_frame = next(iterator)
        except StopIteration:
            return []
            
        camera_movement.append([0, 0])
        
        old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        
        for frame_num, frame in enumerate(iterator, start=1):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_features, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)
            
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0
            
            if new_features is not None and old_features is not None:
                for i, (new, old) in enumerate(zip(new_features, old_features)):
                    new_features_point = new.ravel()
                    old_features_point = old.ravel()
                    
                    distance = abs(new_features_point[0] - old_features_point[0]) + abs(new_features_point[1] - old_features_point[1])
                    
                    if distance > max_distance:
                        max_distance = distance
                        camera_movement_x, camera_movement_y = new_features_point[0] - old_features_point[0], new_features_point[1] - old_features_point[1]
            
            if max_distance > self.minimum_distance:
                camera_movement.append([camera_movement_x, camera_movement_y])
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            else:
                camera_movement.append([0, 0])
            
            old_gray = frame_gray.copy()
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
        
        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """
        Draw camera movement on frames.
        
        Args:
            frames: List or generator of video frames
            camera_movement_per_frame: List of camera movements per frame
            
        Yields:
            Frames with camera movement drawn
        """
        for frame_num, frame in enumerate(frames):
            # Optimize overlay drawing to avoid full frame copy
            # Define ROI for the rectangle
            roi_x1, roi_y1 = 0, 0
            roi_x2, roi_y2 = 500, 100
            
            overlay = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            cv2.rectangle(overlay, (0, 0), (roi_x2-roi_x1, roi_y2-roi_y1), (255, 255, 255), -1)
            
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame[roi_y1:roi_y2, roi_x1:roi_x2], 1 - alpha, 0, frame[roi_y1:roi_y2, roi_x1:roi_x2])
            
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            
            yield frame
