"""Utility functions for bounding box operations."""
import cv2


def get_center_of_bbox(bbox):
    """
    Get center point of bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    """
    Get width of bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Width of bounding box
    """
    return bbox[2] - bbox[0]


def measure_distance(p1, p2):
    """
    Measure Euclidean distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        Distance between points
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def measure_xy_distance(p1, p2):
    """
    Measure x and y distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        Tuple (x_distance, y_distance)
    """
    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox):
    """
    Get foot position (bottom center) of bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple (x, y) of foot position
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def draw_ellipse(frame, bbox, color, track_id=None):
    """
    Draw ellipse at the bottom of bounding box.
    
    Args:
        frame: Video frame
        bbox: Bounding box [x1, y1, x2, y2]
        color: Color tuple (B, G, R)
        track_id: Optional track ID to display
        
    Returns:
        Frame with ellipse drawn
    """
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)
    
    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )
    
    if track_id is not None:
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15
        
        cv2.rectangle(frame,
                     (int(x1_rect), int(y1_rect)),
                     (int(x2_rect), int(y2_rect)),
                     color,
                     cv2.FILLED)
        
        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10
        
        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    
    return frame


def draw_triangle(frame, bbox, color):
    """
    Draw triangle above bounding box (for ball).
    
    Args:
        frame: Video frame
        bbox: Bounding box [x1, y1, x2, y2]
        color: Color tuple (B, G, R)
        
    Returns:
        Frame with triangle drawn
    """
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)
    
    triangle_points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20],
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
    
    return frame


import numpy as np
