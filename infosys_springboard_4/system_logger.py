import datetime
import os

LOG_FILE = "system_logs.txt"

class SystemLogger:
    @staticmethod
    def log(event_type, message):
        """
        Records important system events with a timestamp.
        event_type: e.g., 'SYSTEM_START', 'ALERT_TRIGGERED', 'DETECTION_START'
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{event_type}] {message}\n"
        
        with open(LOG_FILE, "a") as f:
            f.write(log_entry)
        print(log_entry.strip())

# Example usage:
# SystemLogger.log("SYSTEM_START", "Crowd monitoring system initiated.")
