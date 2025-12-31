"""Object detection and tracking using YOLO."""
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    """Tracker class for detecting and tracking objects in video."""

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for tid, tinfo in track.items():
                    bbox = tinfo['bbox']
                    if obj == 'ball':
                        pos = get_center_of_bbox(bbox)
                    else:
                        pos = get_foot_position(bbox)
                    tracks[obj][frame_num][tid]['position'] = pos
        return tracks

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
        df = df.interpolate().bfill()
        return [{1: {"bbox": row}} for row in df.to_numpy().tolist()]

    def detect_frames(self, frames, batch_size=32):
        detections = []
        frames = list(frames)
        total = len(frames)

        print(f"🔍 YOLO inference on {total} frames (batch={batch_size})")

        for i in range(0, total, batch_size):
            batch = frames[i:i+batch_size]
            results = self.model.predict(batch, conf=0.1, verbose=False)
            detections.extend(results)

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            return pickle.load(open(stub_path, 'rb'))

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for fi, det in enumerate(detections):
            cls_names = det.names
            cls_inv = {v: k for k, v in cls_names.items()}

            det_super = sv.Detections.from_ultralytics(det)

            # goalkeeper → player
            for i, cid in enumerate(det_super.class_id):
                if cls_names[cid] == "goalkeeper":
                    det_super.class_id[i] = cls_inv["player"]

            tracked = self.tracker.update_with_detections(det_super)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for obj in tracked:
                bbox = obj[0].tolist()
                cid = obj[3]
                tid = obj[4]

                if cid == cls_inv["player"]:
                    tracks["players"][fi][tid] = {"bbox": bbox}
                if cid == cls_inv["referee"]:
                    tracks["referees"][fi][tid] = {"bbox": bbox}

            for d in det_super:
                bbox = d[0].tolist()
                cid = d[3]
                if cid == cls_inv["ball"]:
                    tracks["ball"][fi][1] = {"bbox": bbox}

        if stub_path:
            pickle.dump(tracks, open(stub_path, 'wb'))

        return tracks

    # ------------------- DRAW UTILS -------------------

    def draw_ellipse(self, frame, bbox, color, tid=None):
        y2 = int(bbox[3])
        x = int((bbox[0]+bbox[2])/2)
        w = int(bbox[2]-bbox[0])

        cv2.ellipse(frame, (x, y2), (w, int(0.35*w)), 0, -45, 235, color, 2)

        if tid is not None:
            cv2.rectangle(frame, (x-20, y2+5), (x+20, y2+25), color, -1)
            cv2.putText(frame, str(tid), (x-10, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x = int((bbox[0]+bbox[2]) / 2)
        pts = np.array([[x, y], [x-12, y-22], [x+12, y-22]])
        cv2.drawContours(frame, [pts], 0, color, -1)
        cv2.drawContours(frame, [pts], 0, (0,0,0), 2)

    # ------------------- MAIN DRAW FUNCTION -------------------

    def draw_annotations(self, video_frames, tracks, team_ball_control, ball_owner):
        """
        FIXED VERSION:
        Draw triangle for correct player based on ball_owner list.
        """
        for fi, frame in enumerate(video_frames):

            if fi >= len(tracks["players"]):
                break

            frame = frame.copy()
            owner_pid = ball_owner[fi] if fi < len(ball_owner) else -1

            # Draw players
            for pid, pdata in tracks["players"][fi].items():
                color = pdata.get("team_color", (0, 0, 255))
                self.draw_ellipse(frame, pdata["bbox"], color, pid)

                if pid == owner_pid:
                    self.draw_triangle(frame, pdata["bbox"], (0, 0, 255))

            # Draw referees
            for _, ref in tracks["referees"][fi].items():
                self.draw_ellipse(frame, ref["bbox"], (0, 255, 255))

            # Draw ball
            for _, ball in tracks["ball"][fi].items():
                self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            yield frame
