# Football Analysis System - Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning repositories)

## Installation Steps

### 1. Install Python Dependencies

Open PowerShell or Command Prompt and navigate to the project directory:

```powershell
cd "c:\Users\PAVAN HAMBAR\Football"
```

Install the required packages:

```powershell
python -m pip install ultralytics opencv-python numpy pandas matplotlib scikit-learn supervision
```

**Alternative**: If you encounter issues, install packages one by one:

```powershell
python -m pip install ultralytics
python -m pip install opencv-python
python -m pip install numpy
python -m pip install pandas
python -m pip install matplotlib
python -m pip install scikit-learn
python -m pip install supervision
```

### 2. Download the Pre-trained YOLO Model

The project uses a custom-trained YOLOv5 model for detecting players, referees, and footballs.

**Option 1: Download from Google Drive**
1. Visit: https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing
2. Click "Download" to download the model file
3. Rename the downloaded file to `best.pt`
4. Move the file to: `c:\Users\PAVAN HAMBAR\Football\models\best.pt`

**Option 2: Use a different YOLO model**
- You can use any YOLOv5 or YOLOv8 model trained on football/soccer datasets
- Place the model file in the `models/` directory
- Update the model path in `main.py` (line 22) to match your model filename

### 3. Download Sample Video

To test the system, you'll need a football video.

**Option 1: Download the tutorial's sample video**
1. Visit: https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing
2. Click "Download" to download the video
3. Rename the file to `08fd33_4.mp4`
4. Move the file to: `c:\Users\PAVAN HAMBAR\Football\input_videos\08fd33_4.mp4`

**Option 2: Use your own football video**
- Place any football/soccer video in the `input_videos/` directory
- Update the video path in `main.py` (line 20) to match your video filename
- Supported formats: .mp4, .avi, .mov, .mkv

### 4. Verify Installation

Check that all packages are installed correctly:

```powershell
python -c "import ultralytics; import cv2; import numpy; import pandas; import matplotlib; import sklearn; import supervision; print('All packages installed successfully!')"
```

### 5. Run the Analysis

Once everything is set up, run the main script:

```powershell
python main.py
```

**Expected behavior:**
- The script will process the video frame by frame
- Progress will be displayed in the console
- Processed video will be saved to `output_videos/output_video.avi`
- Tracking data will be cached in `stubs/` directory for faster re-runs

### 6. View the Output

Open the output video:

```powershell
start output_videos\output_video.avi
```

## Troubleshooting

### Issue: "No module named 'ultralytics'"
**Solution**: Install ultralytics package:
```powershell
python -m pip install ultralytics
```

### Issue: "No module named 'cv2'"
**Solution**: Install opencv-python:
```powershell
python -m pip install opencv-python
```

### Issue: "Model file not found"
**Solution**: Make sure the model file is in the correct location:
- Check that `models/best.pt` exists
- Verify the path in `main.py` matches your model filename

### Issue: "Video file not found"
**Solution**: Make sure the video file is in the correct location:
- Check that `input_videos/08fd33_4.mp4` exists
- Verify the path in `main.py` matches your video filename

### Issue: "Out of memory" or slow processing
**Solution**: 
- Reduce video resolution or length
- Process fewer frames at a time (modify batch_size in `tracker.py`)
- Close other applications to free up RAM

### Issue: "CUDA not available" warning
**Solution**: This is normal if you don't have an NVIDIA GPU. The system will use CPU instead (slower but functional).

## Project Structure

```
Football/
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
├── SETUP_GUIDE.md                   # This file
├── README.md                        # Project documentation
├── trackers/                        # Object detection and tracking
├── team_assigner/                   # Team assignment logic
├── player_ball_assigner/            # Ball possession logic
├── camera_movement_estimator/       # Camera movement tracking
├── view_transformer/                # Perspective transformation
├── speed_and_distance_estimator/    # Speed and distance calculations
├── utils/                           # Utility functions
├── models/                          # YOLO model files (you need to add)
│   └── best.pt                      # Pre-trained model (download required)
├── input_videos/                    # Input videos (you need to add)
│   └── 08fd33_4.mp4                 # Sample video (download required)
├── output_videos/                   # Processed videos (generated)
└── stubs/                           # Cached tracking data (generated)
```

## Next Steps

1. **Install dependencies** (Step 1)
2. **Download model** (Step 2)
3. **Download sample video** (Step 3)
4. **Run the analysis** (Step 5)
5. **View results** (Step 6)

## Additional Resources

- Original GitHub Repository: https://github.com/abdullahtarek/football_analysis
- YouTube Tutorial: https://youtu.be/neBZ6huolkg
- YOLO Documentation: https://docs.ultralytics.com/
- OpenCV Documentation: https://docs.opencv.org/

## Support

If you encounter any issues not covered in this guide, please check:
1. The original GitHub repository's issues section
2. The YouTube video comments
3. Python and package versions compatibility
