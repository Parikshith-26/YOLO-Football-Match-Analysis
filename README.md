# Football Analysis System

An AI/ML football analysis system that uses YOLO for object detection, KMeans for team assignment, optical flow for camera movement tracking, and perspective transformation to analyze player movements in football videos.

## Features

- **Object Detection & Tracking**: Detect and track players, referees, and footballs using YOLOv8/v5
- **Team Assignment**: Automatically assign players to teams based on shirt colors using KMeans clustering
- **Ball Interpolation**: Smooth ball trajectory by interpolating missing detections
- **Camera Movement Estimation**: Track camera movement using optical flow
- **Perspective Transformation**: Convert pixel coordinates to real-world measurements
- **Speed & Distance Calculation**: Measure player speed (km/h) and distance covered (meters)
- **Ball Possession Statistics**: Calculate team ball possession percentages

## Installation

1. Install Python 3.x (3.8 or higher recommended)

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained YOLO model:
   - Download from: https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing
   - Place in `models/` directory

4. Download sample video (optional):
   - Download from: https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing
   - Place in `input_videos/` directory

## Usage

Run the analysis on a video:
```bash
python main.py
```

The processed video will be saved to `output_videos/` directory.

## Project Structure

```
Football/
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
├── trackers/                        # Object detection and tracking
│   ├── __init__.py
│   └── tracker.py
├── team_assigner/                   # Team assignment logic
│   ├── __init__.py
│   └── team_assigner.py
├── camera_movement_estimator/       # Camera movement tracking
│   ├── __init__.py
│   └── camera_movement_estimator.py
├── view_transformer/                # Perspective transformation
│   ├── __init__.py
│   └── view_transformer.py
├── speed_and_distance_estimator/    # Speed and distance calculations
│   ├── __init__.py
│   └── speed_and_distance_estimator.py
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── video_utils.py
│   ├── bbox_utils.py
│   └── ball_interpolation.py
├── models/                          # YOLO model files
├── input_videos/                    # Input videos
├── output_videos/                   # Processed videos
└── stubs/                           # Cached tracking data
```

## Modules

### Tracker
Detects and tracks players, referees, and footballs using YOLO object detection model.

### Team Assigner
Uses KMeans clustering on shirt colors to assign players to teams.

### Camera Movement Estimator
Estimates camera movement between frames using optical flow to accurately measure player movement.

### View Transformer
Applies perspective transformation to convert pixel measurements to real-world meters.

### Speed and Distance Estimator
Calculates player speed and distance covered based on transformed coordinates.

## Credits

Based on the tutorial by Abdullah Tarek: https://github.com/abdullahtarek/football_analysis
