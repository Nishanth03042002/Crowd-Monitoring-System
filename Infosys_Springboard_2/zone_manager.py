import cv2
import json
import os
import time
from datetime import datetime
import random
import numpy as np

# Global variables
ZONES_FILE = "zones.json"
zones = []
drawing = False
ix, iy = -1, -1
temp_px, temp_py = -1, -1

def get_random_color():
    """Generates a random bgr color"""
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def load_zones():
    """Feature 1: Load saved zones automatically"""
    global zones
    if os.path.exists(ZONES_FILE):
        try:
            with open(ZONES_FILE, 'r') as f:
                saved_zones = json.load(f)
                zones = saved_zones
            print("Zones loaded successfully.")
        except Exception as e:
            print(f"Error loading zones: {e}")
            zones = []

def save_zones():
    try:
        with open(ZONES_FILE, 'w') as f:
            json.dump(zones, f, indent=4)
        print("Zones saved successfully.")
    except Exception as e:
        print(f"Error saving zones: {e}")

def draw_zone_callback(event, x, y, flags, param):
    global ix, iy, drawing, temp_px, temp_py, zones
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        temp_px, temp_py = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_px, temp_py = x, y
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Feature 10: Log zone creation time
        timestamp = datetime.now().isoformat()
        
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        
        # Make sure it's a valid rectangle
        if x1 != x2 and y1 != y2:
            zone_id = len(zones) + 1
            zone_name = f"Zone {zone_id}"
            
            zones.append({
                "id": zone_id,
                "name": zone_name,
                "p1": [x1, y1],
                "p2": [x2, y2],
                "color": get_random_color(),
                "timestamp": timestamp
            })
            print(f"Created {zone_name} at {timestamp}")

def main():
    global zones, drawing, ix, iy, temp_px, temp_py
    
    # Load previously saved zones when the program starts
    load_zones()
    
    window_name = 'Video Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, draw_zone_callback)
    
    cap = cv2.VideoCapture(0)
    
    # Feature 7: Handle camera errors
    if not cap.isOpened():
        print("Error: Could not open camera.")
        # Show a message instead of crashing
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_image, "Error: Could not open camera.", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        while True:
            cv2.imshow(window_name, blank_image)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    # Text for Feature 8: Add instruction overlay
    instructions = [
        "Controls:",
        "Mouse : Draw Zone",
        "'d'   : Delete last zone",
        "'r'   : Reset all zones",
        "'p'   : Save screenshot",
        "'f'   : Toggle fullscreen",
        "'q'   : Save & Quit"
    ]
    
    fullscreen = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Draw saved zones
        for zone in zones:
            p1 = tuple(zone['p1'])
            p2 = tuple(zone['p2'])
            color = tuple(zone['color'])
            name = zone['name']
            
            cv2.rectangle(frame, p1, p2, color, 2)
            
            # Feature 2: Display zone names on screen
            cv2.putText(frame, name, (p1[0], p1[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Feature 5: Show zone count placeholder
            cv2.putText(frame, "Count: 0", (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Draw current zone while dragging
        if drawing:
            cv2.rectangle(frame, (ix, iy), (temp_px, temp_py), (0, 255, 0), 2)
            
        # Draw instruction overlay
        y_offset = 30
        for line in instructions:
            # Using a background rect could make it clearer, but simple PutText works too
            cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
            
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            save_zones()
            break
        # Feature 3: Add delete zone feature
        elif key == ord('d'):
            if zones:
                zones.pop()
                print("Last zone deleted.")
        # Feature 4: Add reset zones feature
        elif key == ord('r'):
            zones = []
            print("All zones reset.")
        # Feature 6: Save zone screenshot
        elif key == ord('p'):
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved as {filename}")
        # Feature 9: Add fullscreen toggle
        elif key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
