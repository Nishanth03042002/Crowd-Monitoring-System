import cv2
import json
import os
import datetime
import csv
import numpy as np
from ultralytics import YOLO
from system_logger import SystemLogger

ZONE_FILE = "zones.json"
CSV_FILE = "count_data.csv"
ALERTS_DIR = "alerts"
MAX_CROWD_LIMIT = 5 # Default limit for overcrowding

if not os.path.exists(ALERTS_DIR):
    os.makedirs(ALERTS_DIR)

class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Zone Name", "Entry Count", "Exit Count", "Total People"])

    def log(self, zone_name, entry_count, exit_count, total_people):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, zone_name, entry_count, exit_count, total_people])

class ZoneManager:
    def __init__(self, filepath, logger):
        self.filepath = filepath
        self.logger = logger
        self.zones = []
        self.colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 0, 128),  # Purple
            (128, 128, 0)   # Olive
        ]
        self.counts = {} # {zone_id: {"entry": 0, "exit": 0, "current": 0}}
        self.load_zones()

    def add_zone(self, rect):
        zone_id = len(self.zones) + 1
        color = self.colors[(zone_id - 1) % len(self.colors)]
        timestamp = datetime.datetime.now().isoformat()
        
        zone = {
            "id": zone_id,
            "rect": rect, # (x, y, w, h)
            "color": color,
            "created_at": timestamp
        }
        self.zones.append(zone)
        self.counts[zone_id] = {"entry": 0, "exit": 0, "current": 0}
        SystemLogger.log("ZONE_ADDED", f"Added Zone {zone_id} at {rect}")
        self.save_zones()

    def delete_last_zone(self):
        if self.zones:
            removed = self.zones.pop()
            self.counts.pop(removed['id'], None)
            SystemLogger.log("ZONE_REMOVED", f"Removed Zone {removed['id']}")
            self.save_zones()

    def clear_zones(self):
        self.zones = []
        self.counts = {}
        SystemLogger.log("ZONES_CLEARED", "All zones cleared.")
        self.save_zones()

    def save_zones(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.zones, f, indent=4)
        except Exception as e:
            SystemLogger.log("ERROR", f"Error saving zones: {e}")

    def load_zones(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.zones = json.load(f)
                for zone in self.zones:
                    self.counts[zone['id']] = {"entry": 0, "exit": 0, "current": 0}
                SystemLogger.log("SYSTEM_INIT", f"Loaded {len(self.zones)} zones.")
            except Exception as e:
                SystemLogger.log("ERROR", f"Error loading zones: {e}")
                self.zones = []

    def draw_zones(self, frame):
        for zone in self.zones:
            x, y, w, h = zone['rect']
            color = tuple(zone['color'])
            zid = zone['id']
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"Zone {zid}"
            entry_txt = f"E:{self.counts[zid]['entry']} X:{self.counts[zid]['exit']}"
            curr_txt = f"Current: {self.counts[zid]['current']}"
            
            # Label
            (fw, fh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - 25), (x + fw + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Counts
            cv2.putText(frame, entry_txt, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, curr_txt, (x + 5, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

class TrackState:
    def __init__(self):
        self.history = {} # {id: {"inside_zones": set(), "pos": (cx, cy)}}

# Global state
drawing = False
ix, iy = -1, -1
current_rect = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, current_rect
    zone_manager = param
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_rect = (x, y, 0, 0)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_rect = (ix, iy, x - ix, y - iy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w, h = abs(x - ix), abs(y - iy)
        start_x, start_y = min(ix, x), min(iy, y)
        if w > 10 and h > 10:
            zone_manager.add_zone((start_x, start_y, w, h))
        current_rect = None

def draw_dashboard(frame, zone_manager, total_people, alert_active):
    h, w = frame.shape[:2]
    # Dashboard Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 280, 0), (w, 350), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.rectangle(frame, (w - 280, 0), (w, 350), (0, 255, 0), 2)
    
    cv2.putText(frame, "CROWD DASHBOARD", (w - 260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(frame, (w - 270, 55), (w - 10, 55), (0, 255, 0), 1)
    
    cv2.putText(frame, f"Total People Detected: {total_people}", (w - 260, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y_off = 130
    cv2.putText(frame, "Zone Breakdowns:", (w - 260, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_off += 30
    
    for zid, counts in zone_manager.counts.items():
        txt = f"Zone {zid}: E:{counts['entry']} | X:{counts['exit']} | C:{counts['current']}"
        cv2.putText(frame, txt, (w - 260, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y_off += 25

    if alert_active:
        # Warning Message (Task 4)
        t = datetime.datetime.now().second
        if t % 2 == 0: # Blinking effect
            cv2.putText(frame, "OVERCROWDING ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (0, 0, 255), 10)

def main():
    global drawing, current_rect
    SystemLogger.log("SYSTEM_START", "Crowd monitoring system initiated.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        SystemLogger.log("ERROR", "Could not open webcam.")
        return
    SystemLogger.log("CAMERA_START", "Camera feed started successfully.")

    # Initialize YOLOv8
    model = YOLO('yolov8n.pt') 
    SystemLogger.log("DETECTION_START", "YOLOv8 model loaded.")

    logger = CSVLogger(CSV_FILE)
    zone_manager = ZoneManager(ZONE_FILE, logger)
    track_state = TrackState()

    window_name = "Crowd Monitoring System"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rectangle, zone_manager)

    last_alert_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Run Tracking
        results = model.track(frame, persist=True, classes=[0], verbose=False) 
        
        tracks_data = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            for box in boxes:
                id = int(box.id[0])
                x1, y1, x2, y2 = box.xyxy[0]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                tracks_data.append((id, cx, cy))
                
                # Draw Box and ID
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update Zone Counts
        current_tids = [t[0] for t in tracks_data]
        total_people = len(tracks_data)
        
        # Reset current counts for this frame
        for zid in zone_manager.counts:
            zone_manager.counts[zid]["current"] = 0

        # 1. Update counts for currently tracked people
        for tid, cx, cy in tracks_data:
            if tid not in track_state.history:
                track_state.history[tid] = {"inside_zones": set(), "pos": (cx, cy)}
            
            for zone in zone_manager.zones:
                zid = zone['id']
                zx, zy, zw, zh = zone['rect']
                is_inside = (zx <= cx <= zx + zw and zy <= cy <= zy + zh)
                was_inside = zid in track_state.history[tid]["inside_zones"]
                
                if is_inside:
                    zone_manager.counts[zid]["current"] += 1
                    if not was_inside:
                        # Just Entered
                        zone_manager.counts[zid]["entry"] += 1
                        track_state.history[tid]["inside_zones"].add(zid)
                        logger.log(f"Zone {zid}", zone_manager.counts[zid]["entry"], zone_manager.counts[zid]["exit"], total_people)
                elif was_inside:
                    # Just Exited
                    zone_manager.counts[zid]["exit"] += 1
                    track_state.history[tid]["inside_zones"].remove(zid)
                    logger.log(f"Zone {zid}", zone_manager.counts[zid]["entry"], zone_manager.counts[zid]["exit"], total_people)
            
            track_state.history[tid]["pos"] = (cx, cy)

        # 2. Handle cleanup for people who disappeared from view
        for tid in list(track_state.history.keys()):
            if tid not in current_tids:
                # If they were inside any zones when they disappeared, count as an exit
                for zid in list(track_state.history[tid]["inside_zones"]):
                    zone_manager.counts[zid]["exit"] += 1
                    logger.log(f"Zone {zid}", zone_manager.counts[zid]["entry"], zone_manager.counts[zid]["exit"], total_people)
                del track_state.history[tid]

        # Overcrowding Logic (Task 3 & 4)
        alert_active = False
        for zid, counts in zone_manager.counts.items():
            if counts["current"] > MAX_CROWD_LIMIT:
                alert_active = True
                break
        
        if alert_active:
            # Capture Screenshot (Task 5) - throttle to once every 5 seconds
            current_time = datetime.datetime.now().timestamp()
            if current_time - last_alert_time > 5:
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(ALERTS_DIR, f"alert_{timestamp_str}.jpg")
                cv2.imwrite(filename, frame)
                SystemLogger.log("ALERT_TRIGGERED", f"Overcrowding in zones! Screenshot saved: {filename}")
                last_alert_time = current_time

        zone_manager.draw_zones(frame)
        draw_dashboard(frame, zone_manager, total_people, alert_active)

        if drawing and current_rect:
            x, y, w, h = current_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break
        elif key == ord('d'): zone_manager.delete_last_zone()
        elif key == ord('r'): zone_manager.clear_zones()

    SystemLogger.log("SYSTEM_SHUTDOWN", "Crowd monitoring system closed.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
