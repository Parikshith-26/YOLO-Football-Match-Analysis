import os
import json
import numpy as np
import traceback

from utils.video_utils import read_video, save_video, get_video_info
from utils.speed_plot import plot_player_speed
from utils.distance_plot import plot_distance_covered
from utils.possession_timeline import plot_possession_timeline
from utils.team_radar import team_radar
from utils.pdf_report import make_report_data

from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "output_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _norm(p: str) -> str:
    return p.replace("\\", "/")


def _save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


def _ensure_png(path):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(2, 1))
    plt.text(0.5, 0.5, "No data", ha="center", va="center")
    plt.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def process_video(input_path, output_path=None):
    """
    Full updated pipeline with FIXED ball-owner tracking.
    """

    try:
        print(f"Processing {input_path}...")
        video_info = get_video_info(input_path)
        if not video_info:
            print("ERROR reading video info")
            return None

        total_frames = int(video_info.get("total_frames", 0))
        fps = int(video_info.get("fps", 25))

        frames = read_video(input_path)

        # ------------------------- TRACKING -------------------------
        tracker = Tracker(os.path.join(BASE_DIR, "models", "best.pt"))
        print("Tracking...")

        tracks = tracker.get_object_tracks(
            frames,
            read_from_stub=True,
            stub_path=os.path.join(BASE_DIR, "stubs", f"track_stubs_{os.path.basename(input_path)}.pkl")
        )

        tracker.add_position_to_tracks(tracks)

        # Normalize track lists
        for k in tracks:
            while len(tracks[k]) < total_frames:
                tracks[k].append({})
            while len(tracks[k]) > total_frames:
                tracks[k].pop()

        # ------------------------- CAMERA -------------------------
        print("Camera movement estimation...")
        cam_est = CameraMovementEstimator(frames[0] if frames else None)

        cam_movements = cam_est.get_camera_movement(
            frames,
            read_from_stub=True,
            stub_path=os.path.join(BASE_DIR, "stubs", f"cam_stub_{os.path.basename(input_path)}.pkl")
        )

        cam_est.add_adjust_positions_to_tracks(tracks, cam_movements)

        # ------------------------- VIEW TRANSFORM -------------------------
        vt = ViewTransformer()
        vt.add_transformed_position_to_tracks(tracks)

        # ------------------------- BALL + SPEED -------------------------
        tracks["ball"] = tracker.interpolate_ball_positions(tracks.get("ball", []))

        speed_calc = SpeedAndDistanceEstimator()
        speed_calc.add_speed_and_distance_to_tracks(tracks)

        # ------------------------- TEAM ASSIGN -------------------------
        ta = TeamAssigner()
        first_frame = frames[0] if frames else None

        ta.assign_team_color(first_frame, tracks.get("players", [])[0] if tracks["players"] else {})

        for fi, frame in enumerate(frames):
            if fi >= len(tracks["players"]):
                break
            for pid, pdata in tracks["players"][fi].items():
                team = ta.get_player_team(frame, pdata.get("bbox"), pid)
                pdata["team"] = int(team)
                pdata["team_color"] = ta.team_colors.get(team, (0, 255, 0))

        # ------------------------- BALL POSSESSION -------------------------
        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        ball_owner = []      # <=== FIXED: actual player who owns the ball

        for fi, players in enumerate(tracks.get("players", [])):

            # ball missing
            if fi >= len(tracks["ball"]):
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
                ball_owner.append(-1)
                continue

            ball_frame = tracks["ball"][fi] if isinstance(tracks["ball"][fi], dict) else {}
            ball_bbox = ball_frame.get(1, {}).get("bbox")

            if ball_bbox:
                assigned_pid = player_assigner.assign_ball_to_player(players, ball_bbox)
            else:
                assigned_pid = -1

            # Determine team ball possession
            if assigned_pid != -1:
                team_ball_control.append(players[assigned_pid].get("team", 0))
                ball_owner.append(assigned_pid)
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
                ball_owner.append(-1)

        team_ball_control = np.array(team_ball_control)

        # ---------------------- DRAW FINAL ANNOTATED FRAMES ----------------------
        print("Drawing annotations...")

        annotated = list(
            tracker.draw_annotations(
                frames,
                tracks,
                team_ball_control,
                ball_owner       # <=== passed into draw_annotations()
            )
        )

        annotated = list(cam_est.draw_camera_movement(annotated, cam_movements))
        annotated = list(speed_calc.draw_speed_and_distance(annotated, tracks))

        # ---------------------- SAVE VIDEO ----------------------
        if output_path is None:
            base = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(OUTPUT_DIR, f"processed_{base}.mp4")

        save_video(annotated, output_path, fps=fps)

        print("Saved processed video:", output_path)

        # ---------------------- CHART OUTPUT FILES ----------------------
        base = os.path.splitext(os.path.basename(output_path))[0]

        speed_png = _norm(os.path.join(OUTPUT_DIR, f"speed_{base}.png"))
        dist_png = _norm(os.path.join(OUTPUT_DIR, f"distance_{base}.png"))
        poss_png = _norm(os.path.join(OUTPUT_DIR, f"possession_{base}.png"))
        radar_png = _norm(os.path.join(OUTPUT_DIR, f"radar_{base}.png"))
        analysis_json = _norm(os.path.join(OUTPUT_DIR, f"analysis_{base}.json"))

        # ---------------------- CHART GENERATION ----------------------
        try: plot_player_speed(tracks, speed_png)
        except: _ensure_png(speed_png)

        try: plot_distance_covered(tracks, dist_png)
        except: _ensure_png(dist_png)

        try: plot_possession_timeline(team_ball_control.tolist(), poss_png)
        except: _ensure_png(poss_png)

        # ---------------------- RADAR METRICS ----------------------
        player_stats = {}

        for fi, frame in enumerate(tracks.get("players", [])):
            for pid, pdata in frame.items():
                pid_s = str(pid)
                if pid_s not in player_stats:
                    player_stats[pid_s] = {
                        "team": int(pdata.get("team", 0)),
                        "speeds": [],
                        "distance": 0.0
                    }

                player_stats[pid_s]["speeds"].append(float(pdata.get("speed", 0.0)))
                player_stats[pid_s]["distance"] = float(pdata.get("distance", player_stats[pid_s]["distance"]))

        def avg_safe(v):
            arr = np.array(v, dtype=float) if v else np.array([0.0])
            return float(np.nanmean(arr))

        t1_pos = float(np.mean(team_ball_control == 1) * 100)
        t2_pos = float(np.mean(team_ball_control == 2) * 100)
        t1_dist = sum(p["distance"] for p in player_stats.values() if p["team"] == 1)
        t2_dist = sum(p["distance"] for p in player_stats.values() if p["team"] == 2)
        t1_avg = avg_safe([avg_safe(p["speeds"]) for p in player_stats.values() if p["team"] == 1])
        t2_avg = avg_safe([avg_safe(p["speeds"]) for p in player_stats.values() if p["team"] == 2])

        metrics = {
            "labels": ["Possession", "Avg Speed", "Total Distance"],
            "team1": [t1_pos, t1_avg, t1_dist],
            "team2": [t2_pos, t2_avg, t2_dist]
        }

        try:
            team_radar(metrics, radar_png)
        except:
            _ensure_png(radar_png)

        # ---------------------- TOP PERFORMANCE ----------------------
        compiled = {}
        for pid, pdata in player_stats.items():
            compiled[pid] = {
                "team": pdata["team"],
                "max_speed": float(np.nanmax(pdata["speeds"])),
                "avg_speed": float(np.nanmean(pdata["speeds"])),
                "distance": pdata["distance"]
            }

        top_dist = sorted(compiled.items(), key=lambda x: x[1]["distance"], reverse=True)[:3]
        top_speed = sorted(compiled.items(), key=lambda x: x[1]["max_speed"], reverse=True)[:3]

        # longest possession streak
        cur1 = cur2 = best1 = best2 = 0
        for t in team_ball_control:
            if t == 1:
                cur1 += 1; best1 = max(best1, cur1); cur2 = 0
            elif t == 2:
                cur2 += 1; best2 = max(best2, cur2); cur1 = 0
            else:
                cur1 = cur2 = 0

        performance = {
            "top_distance": [{"player": int(pid), "distance": round(d["distance"], 2)} for pid, d in top_dist],
            "top_speed":    [{"player": int(pid), "speed": round(d["max_speed"], 2)} for pid, d in top_speed],
            "possession_streak": {
                "team1_seconds": round(best1 / max(fps,1), 2),
                "team2_seconds": round(best2 / max(fps,1), 2),
            }
        }

        analysis = {
            "fps": fps,
            "total_frames": total_frames,
            "team_ball_control": team_ball_control.tolist(),
            "player_stats": compiled,
            "images": {
                "speed": speed_png,
                "distance": dist_png,
                "possession": poss_png,
                "radar": radar_png
            }
        }

        _save_json(analysis, analysis_json)

        return {
            "total_frames": total_frames,
            "video_fps": fps,
            "team_possession": {"team_1": round(t1_pos, 2), "team_2": round(t2_pos, 2)},
            "processed_filename": os.path.basename(output_path),
            "speed_chart": os.path.basename(speed_png),
            "distance_map": os.path.basename(dist_png),
            "possession_map": os.path.basename(poss_png),
            "radar_chart": os.path.basename(radar_png),
            "analysis_json": os.path.basename(analysis_json),
            "player_stats": compiled,
            "top_performance": performance
        }

    except Exception as e:
        traceback.print_exc()
        print("Pipeline Error:", e)
        return None

