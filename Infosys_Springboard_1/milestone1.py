import cv2
import json
import os

ZONES_FILE = "zones.json"
zones = []
drawing = False
start_point = (-1, -1)
current_end_point = (-1, -1)

# List of predefined colors (B, G, R)
COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (128, 0, 128),  # Purple
    (0, 165, 255)   # Orange
]

def get_color(index):
    return COLORS[index % len(COLORS)]

def load_zones():
    global zones
    if os.path.exists(ZONES_FILE):
        try:
            with open(ZONES_FILE, "r") as f:
                zones = json.load(f)
        except Exception as e:
            print(f"Error loading zones: {e}")
            zones = []

def save_zones():
    with open(ZONES_FILE, "w") as f:
        json.dump(zones, f)
    print("Zones saved successfully.")

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, current_end_point, zones
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_end_point = (x, y)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_end_point = (x, y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_end_point = (x, y)
        # Ensure it's a valid rectangle (not just a click)
        if abs(start_point[0] - current_end_point[0]) > 5 and abs(start_point[1] - current_end_point[1]) > 5:
            # Add zone as [x_min, y_min, x_max, y_max]
            x_min = min(start_point[0], current_end_point[0])
            y_min = min(start_point[1], current_end_point[1])
            x_max = max(start_point[0], current_end_point[0])
            y_max = max(start_point[1], current_end_point[1])
            zones.append([x_min, y_min, x_max, y_max])
            print(f"Added Zone {len(zones)}")
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Check if click is inside any existing zone to delete it (iterate backwards to delete top-most)
        for i in range(len(zones) - 1, -1, -1):
            x_min, y_min, x_max, y_max = zones[i]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                zones.pop(i)
                print(f"Deleted Zone {i + 1}")
                break

def main():
    load_zones()
    
    cap = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', mouse_callback)
    
    print("------------------------------------------")
    print("Controls:")
    print("- Left Click & Drag: Draw a new zone")
    print("- Right Click on a zone: Delete that zone")
    print("- Press 'c': Clear all zones")
    print("- Press 'q': Quit and save zones")
    print("------------------------------------------")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Draw saved zones
        for i, zone in enumerate(zones):
            x_min, y_min, x_max, y_max = zone
            color = get_color(i)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Label the zone
            label = f"Zone {i + 1}"
            cv2.putText(frame, label, (x_min, max(y_min - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
        # Draw current zone actively being drawn
        if drawing:
            cv2.rectangle(frame, start_point, current_end_point, (255, 255, 255), 2)
            
        cv2.imshow('Video', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            zones.clear()
            print("Cleared all zones.")
            
    # Save zones automatically when closing
    save_zones()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
