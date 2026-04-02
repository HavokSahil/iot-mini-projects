import requests
import pandas as pd
import os
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "soil_data.csv")

API_URL = "http://10.42.0.1:5000/latest" 

COLUMN_ORDER = [
    "time", "moisture", "temperature", "ec", "ph", 
    "nitrogen", "phosphorus", "potassium", "salinity", "tds"
]

def continuous_update():
    file_exists = os.path.isfile(CSV_PATH)
    
    print(f"[{datetime.now()}] Starting continuous high-frequency logging...")
    print(f"Target file: {CSV_PATH}")
    print("Press Ctrl+C to stop.")

    while True:
        start_time = time.time() 
        
        try:
            response = requests.get(API_URL, timeout=0.5)
            response.raise_for_status()
            data = response.json()

            df_new = pd.DataFrame([data] if isinstance(data, dict) else data)
            
            df_new = df_new[COLUMN_ORDER]

            df_new.to_csv(CSV_PATH, mode='a', index=False, header=not file_exists)
            
            if not file_exists:
                file_exists = True

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Loop Error: {e}")

        elapsed = time.time() - start_time
        time.sleep(max(0, 0.1 - elapsed))

if __name__ == "__main__":
    continuous_update()