import cv2
import json
import os
import datetime
import csv
import numpy as np
from collections import OrderedDict

ZONE_FILE = "zones.json"
CSV_FILE = "count_data.csv"

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - inputCentroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Zone Name", "Entry Count", "Exit Count"])

    def log(self, zone_name, entry_count, exit_count):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([zone_name, entry_count, exit_count])

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
        self.counts = {} # {zone_name: {"entry": 0, "exit": 0, "counted_ids": set()}}
        self.load_zones()

    def add_zone(self, rect):
        zone_id = len(self.zones) + 1
        zone_name = f"Zone {zone_id}"
        color = self.colors[(zone_id - 1) % len(self.colors)]
        timestamp = datetime.datetime.now().isoformat()
        
        zone = {
            "id": zone_id,
            "name": zone_name,
            "rect": rect, # (x, y, w, h)
            "color": color,
            "created_at": timestamp
        }
        self.zones.append(zone)
        self.counts[zone_name] = {"entry": 0, "exit": 0, "counted_ids": set()}
        print(f"Added {zone_name} at {rect}")
        self.save_zones()

    def delete_last_zone(self):
        if self.zones:
            removed = self.zones.pop()
            self.counts.pop(removed['name'], None)
            print(f"Removed {removed['name']}")
            self.save_zones()

    def clear_zones(self):
        self.zones = []
        self.counts = {}
        print("All zones cleared.")
        self.save_zones()

    def save_zones(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.zones, f, indent=4)
        except Exception as e:
            print(f"Error saving zones: {e}")

    def load_zones(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.zones = json.load(f)
                for zone in self.zones:
                    if 'name' not in zone:
                        zone['name'] = f"Zone {zone['id']}"
                    self.counts[zone['name']] = {"entry": 0, "exit": 0, "counted_ids": set()}
                print(f"Loaded {len(self.zones)} zones from {self.filepath}")
            except Exception as e:
                print(f"Error loading zones: {e}")
                self.zones = []

    def draw_zones(self, frame):
        for zone in self.zones:
            x, y, w, h = zone['rect']
            color = tuple(zone['color'])
            name = zone['name']
            
            # Draw Zone Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Virtual Line (Horizontal Middle)
            line_y = y + h // 2
            cv2.line(frame, (x, line_y), (x + w, line_y), color, 1, cv2.LINE_AA)
            cv2.putText(frame, "Virtual Line", (x + 5, line_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            entry_txt = f"Entry: {self.counts[name]['entry']}"
            exit_txt = f"Exit: {self.counts[name]['exit']}"
            
            # Label
            (fw, fh), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - 25), (x + fw + 10, y), color, -1)
            cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Counts
            cv2.putText(frame, entry_txt, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, exit_txt, (x + 5, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

class TrackState:
    def __init__(self):
        self.history = {} # {id: (cx, cy)}

    def update(self, objects, zone_manager, logger):
        for objectID, centroid in objects.items():
            cx, cy = centroid
            
            for zone in zone_manager.zones:
                x, y, w, h = zone['rect']
                name = zone['name']
                
                # Check if the person's current centroid is inside the physical zone
                is_inside = (x <= cx <= x + w) and (y <= cy <= y + h)
                
                if is_inside and objectID not in zone_manager.counts[name]["counted_ids"]:
                    # Person has entered or is newly detected inside a new zone
                    zone_manager.counts[name]["entry"] += 1
                    zone_manager.counts[name]["counted_ids"].add(objectID)
                    logger.log(name, zone_manager.counts[name]["entry"], zone_manager.counts[name]["exit"])
                    print(f"DEBUG: Person {objectID} ENTERED {name}.")
                    
                elif not is_inside and objectID in zone_manager.counts[name]["counted_ids"]:
                    # Person was previously in the zone, but has now moved outside
                    zone_manager.counts[name]["exit"] += 1
                    zone_manager.counts[name]["counted_ids"].remove(objectID)
                    logger.log(name, zone_manager.counts[name]["entry"], zone_manager.counts[name]["exit"])
                    print(f"DEBUG: Person {objectID} EXITED {name}.")

        # Cleanup obsolete IDs that the tracker completely dropped
        current_ids = set(objects.keys())
        for zone in zone_manager.zones:
            name = zone['name']
            
            # Find IDs that were in this zone, but are no longer tracked at all (they left the camera fully)
            missing_counted_ids = zone_manager.counts[name]["counted_ids"] - current_ids
            for objectID in missing_counted_ids:
                zone_manager.counts[name]["exit"] += 1
                zone_manager.counts[name]["counted_ids"].remove(objectID)
                logger.log(name, zone_manager.counts[name]["entry"], zone_manager.counts[name]["exit"])
                print(f"DEBUG: Dropped Person {objectID} exited {name} (Lost by tracker).")

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

def draw_dashboard(frame, zone_manager, total_people):
    h, w = frame.shape[:2]
    # Calculate height dynamically based on number of zones
    box_height = 80 + (len(zone_manager.counts) * 25)
    cv2.rectangle(frame, (w - 250, 0), (w, box_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (w - 250, 0), (w, box_height), (0, 255, 0), 2)
    
    cv2.putText(frame, "DASHBOARD", (w - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Total People: {total_people}", (w - 230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y_off = 90
    for name, counts in zone_manager.counts.items():
        txt = f"{name} | Ent: {counts['entry']} | Ext: {counts['exit']}"
        cv2.putText(frame, txt, (w - 230, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_off += 25

def main():
    global drawing, current_rect
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize OpenCV HOG People Detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    logger = CSVLogger(CSV_FILE)
    zone_manager = ZoneManager(ZONE_FILE, logger)
    ct = CentroidTracker(maxDisappeared=10, maxDistance=80)
    track_state = TrackState()

    window_name = "Person Tracking System"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rectangle, zone_manager)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Maintain original size for drawing, resize only for quicker HOG processing
        orig_h, orig_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect people
        (rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
        
        # Adjust rects to fit proper format arrays
        rects_array = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        
        # Apply Non-Maxima Suppression
        pick = non_max_suppression_fast(rects_array, overlapThresh=0.65)
        
        objects = ct.update(pick)
        track_state.update(objects, zone_manager, logger)

        total_people = len(objects)

        for objectID, centroid in objects.items():
            cx, cy = centroid
            
            # Find the original rect to draw
            matched_rect = None
            for (startX, startY, endX, endY) in pick:
                if int((startX + endX) / 2.0) == cx and int((startY + endY) / 2.0) == cy:
                    matched_rect = (startX, startY, endX, endY)
                    break
            
            if matched_rect:
                startX, startY, endX, endY = matched_rect
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {objectID}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw Centroid
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        zone_manager.draw_zones(frame)
        draw_dashboard(frame, zone_manager, total_people)

        if drawing and current_rect:
            x, y, w, h = current_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break
        elif key == ord('d'): zone_manager.delete_last_zone()
        elif key == ord('r'): zone_manager.clear_zones()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
