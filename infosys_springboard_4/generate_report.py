import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

CSV_FILE = "count_data.csv"
REPORT_FILE = "crowd_report.md"
PLOTS_DIR = "reports/plots"

def generate_report():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please run the monitoring system first.")
        return

    # Load data
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Error reading {CSV_FILE}: {e}")
        return

    if df.empty:
        print("Error: No data in CSV file. Please run the monitoring system (main.py) first to collect data.")
        return

    # Check for required columns (Task 6)
    required_cols = ['Timestamp', 'Zone Name', 'Entry Count', 'Exit Count', 'Total People']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: CSV file is missing required columns: {missing_cols}")
        print("This usually happens if you try to run the report on old data. Please run main.py to generate new data.")
        return

    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp')

    # Analyze Data (Task 6)
    total_entries = df.groupby('Zone Name')['Entry Count'].max().sum()
    total_exits = df.groupby('Zone Name')['Exit Count'].max().sum()
    peak_crowd = df['Total People'].max()
    peak_time = df.loc[df['Total People'].idxmax(), 'Timestamp']
    
    # Most crowded zone (by max entry)
    most_crowded_zone = df.groupby('Zone Name')['Entry Count'].max().idxmax()

    # Reporting Metrics
    report_md = f"""# Crowd Activity Analysis Report
Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Key Metrics
- **Total Visitors Detected**: {total_entries}
- **Total Exits Recorded**: {total_exits}
- **Peak Crowd Size**: {peak_crowd} people (at {peak_time})
- **Most Popular Zone**: {most_crowded_zone}

## Zone Statistics
"""

    zone_stats = df.groupby('Zone Name').agg({
        'Entry Count': 'max',
        'Exit Count': 'max'
    }).reset_index()

    for _, row in zone_stats.iterrows():
        report_md += f"- **{row['Zone Name']}**: {row['Entry Count']} Entries, {row['Exit Count']} Exits\n"

    # Save Markdown Report
    with open(REPORT_FILE, "w") as f:
        f.write(report_md)
    print(f"Report saved to {REPORT_FILE}")

    # Create Graphs (Task 7)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    # 1. Crowd Trend Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(df['Timestamp'], df['Total People'], label='Total People', color='blue')
    plt.title('Crowd Trend Over Time')
    plt.xlabel('Time')
    plt.ylabel('People Count')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'crowd_trend.png'))
    plt.close()

    # 2. Zone Comparison (Bar Chart)
    plt.figure(figsize=(10, 6))
    plt.bar(zone_stats['Zone Name'], zone_stats['Entry Count'], color='green', label='Total Entries')
    plt.title('Zone-wise Comparison (Total Entries)')
    plt.xlabel('Zone')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'zone_comparison.png'))
    plt.close()

    print(f"Graphs saved to {PLOTS_DIR}/")

if __name__ == "__main__":
    generate_report()
