import os
import numpy as np
import cv2

from utils import read_video, save_video, get_video_info
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator


def process_video(input_path, output_path):
    """Process one video using tracking + stubs + fast pipeline."""
    print(f"\n==============================")
    print(f"PROCESSING: {input_path}")
    print("==============================")

    try:
        # --------------------------------------
        # 1. VIDEO INFO
        # --------------------------------------
        video_info = get_video_info(input_path)
        if not video_info:
            print(f"❌ ERROR: Could not read video info for {input_path}")
            return
        
        total_frames = video_info["total_frames"]
        fps = video_info["fps"]
        print(f"Video: {total_frames} frames | {fps} FPS")

        # Load all frames
        video_frames = read_video(input_path)
        filename = os.path.basename(input_path)

        # --------------------------------------
        # Prepare stub filenames
        # --------------------------------------
        track_stub = f"stubs/track_stubs_{filename}.pkl"
        cam_stub   = f"stubs/cam_stub_{filename}.pkl"

        tracker = Tracker("models/best.pt")

        # --------------------------------------
        # 2. TRACKING (FAST IF STUB EXISTS)
        # --------------------------------------
        print("Pass 1: Tracking...")

        tracks = tracker.get_object_tracks(
            video_frames,
            read_from_stub=True,
            stub_path=track_stub
        )

        tracker.add_position_to_tracks(tracks)

        # Ensure length matches video
        for key in tracks:
            while len(tracks[key]) < total_frames:
                tracks[key].append({})
            while len(tracks[key]) > total_frames:
                tracks[key].pop()

        # --------------------------------------
        # 3. CAMERA MOVEMENT (FAST IF STUB EXISTS)
        # --------------------------------------
        print("Pass 2: Camera Movement Estimation...")

        first_frame = video_frames[0]
        cme = CameraMovementEstimator(first_frame)

        camera_movements = cme.get_camera_movement(
            video_frames,
            read_from_stub=True,
            stub_path=cam_stub
        )

        cme.add_adjust_positions_to_tracks(tracks, camera_movements)

        # --------------------------------------
        # 4. VIEW TRANSFORMATION
        # --------------------------------------
        vt = ViewTransformer()
        vt.add_transformed_position_to_tracks(tracks)

        # --------------------------------------
        # 5. BALL INTERPOLATION
        # --------------------------------------
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # --------------------------------------
        # 6. SPEED & DISTANCE
        # --------------------------------------
        speed_calc = SpeedAndDistanceEstimator()
        speed_calc.add_speed_and_distance_to_tracks(tracks)

        # --------------------------------------
        # 7. TEAM ASSIGNMENT
        # --------------------------------------
        print("Pass 3: Team Assignment...")

        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(first_frame, tracks["players"][0])

        for i, frame in enumerate(video_frames):
            for pid, pdata in tracks["players"][i].items():
                team = team_assigner.get_player_team(frame, pdata["bbox"], pid)
                pdata["team"] = team
                pdata["team_color"] = team_assigner.team_colors[team]

        # --------------------------------------
        # 8. BALL POSSESSION
        # --------------------------------------
        pba = PlayerBallAssigner()
        team_ball_control = []

        for i, players in enumerate(tracks["players"]):
            ball_bbox = tracks["ball"][i][1]["bbox"]
            pid = pba.assign_ball_to_player(players, ball_bbox)

            if pid != -1:
                players[pid]["has_ball"] = True
                team_ball_control.append(players[pid]["team"])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

        team_ball_control = np.array(team_ball_control)

        # --------------------------------------
        # 9. DRAW OUTPUT
        # --------------------------------------
        print("Pass 4: Drawing Output...")

        frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
        frames = cme.draw_camera_movement(frames, camera_movements)
        frames = speed_calc.draw_speed_and_distance(frames, tracks)

        # --------------------------------------
        # 10. SAVE OUTPUT (MP4 for speed)
        # --------------------------------------
        print(f"Saving output → {output_path}")
        save_video(frames, output_path)
        print(f"✅ DONE: {output_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ ERROR processing {input_path}: {e}")


def main(source_path):
    """Run analysis on a directory or single video."""
    print(f"Source: {source_path}")

    if os.path.isdir(source_path):  # MULTIPLE VIDEOS
        for f in os.listdir(source_path):
            if f.endswith(".mp4"):
                input_path = os.path.join(source_path, f)
                output_path = os.path.join("output_videos", f"processed_{f}")
                process_video(input_path, output_path)

    elif os.path.isfile(source_path):  # SINGLE VIDEO
        output_path = "output_videos/processed_single.mp4"
        process_video(source_path, output_path)

    else:
        print("❌ ERROR: Invalid source path.")


if __name__ == "__main__":
    src = "input_videos"
    os.makedirs("output_videos", exist_ok=True)
    main(src)
