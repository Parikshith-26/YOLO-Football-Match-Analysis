"""Ball interpolation utilities."""
import pandas as pd
import numpy as np

def interpolate_ball_positions_fast(ball_positions):
    """
    Pure numpy interpolation: 2x faster than pandas.
    """
    # Extract bboxes
    bboxes = []
    for frame_data in ball_positions:
        bbox = frame_data.get(1, {}).get('bbox', [np.nan]*4)
        bboxes.append(bbox)
    
    bboxes = np.array(bboxes, dtype=np.float64)
    
    # Handle each column (x1, y1, x2, y2)
    for col in range(4):
        mask = ~np.isnan(bboxes[:, col])
        if not mask.any():
            continue
        
        # Linear interpolation + forward/backward fill
        valid_indices = np.where(mask)[0]
        valid_values = bboxes[mask, col]
        
        all_indices = np.arange(len(bboxes))
        interpolated = np.interp(all_indices, valid_indices, valid_values)
        
        bboxes[:, col] = interpolated
    
    # Forward fill any remaining NaNs
    for col in range(4):
        mask = ~np.isnan(bboxes[:, col])
        if mask.any():
            first_valid = np.where(mask)[0][0]
            if first_valid > 0:
                bboxes[:first_valid, col] = bboxes[first_valid, col]
    
    # Convert back to original format
    return [{1: {"bbox": bbox}} for bbox in bboxes.tolist()]
