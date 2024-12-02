import os
import requests
import pandas as pd
import json
import logging
from typing import List, Dict
from global_variables import LANDING_DIR, CLEAN_LOCATIONS_API, CLEAN_LOCATION_DATASET_NAME

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

def fetch_location_data(api_url: str) -> List[Dict]:
    logging.info("🌐 Sending request to fetch location data")
    response = requests.get(api_url)
    if response.status_code != requests.codes.ok:
        logging.error(f"❌ Failed to fetch data: {response.status_code} - {response.text}")
        raise Exception(response.status_code, response.text)
    logging.info("✅ Successfully fetched data from API")
    return json.loads(response.content)["data"]

def process_location_data(data: List[Dict]) -> pd.DataFrame:
    logging.info("📊 Extracting location data into a DataFrame")
    rows = [
        {"iso2": entry["iso2"], "iso3": entry["iso3"], "country":entry["country"]} for entry in data
    ]
    logging.debug(f"Extracted rows: {rows[:10]}")  
    return pd.DataFrame(rows)

def save_location_data(df: pd.DataFrame, file_path: str) -> None:
    logging.info(f"💾 Saving data to {file_path}")
    
    if file_path.lower().endswith("parquet"):
        df.to_parquet(file_path, compression="brotli")
        logging.info(f"🎉 Data successfully saved as Parquet to {file_path}")
    
    elif file_path.lower().endswith("csv"):
        df.to_csv(file_path, index=False)
        logging.info(f"🎉 Data successfully saved as CSV to {file_path}")
    
    else:
        logging.warning(f"⚠️ Unsupported file format for {file_path}")
        raise ValueError(f"Unsupported file extension for {file_path}")

def main():
    logging.info("🚀 Starting the location data processing script")
    try:
        temp_file_path = f"{os.path.splitext(full_file_path)[0]}_temp.csv"
        
        if not os.path.isfile(temp_file_path):
            data = fetch_location_data(CLEAN_LOCATIONS_API)
            location_df = process_location_data(data)
            logging.info("📊 Saving temporary file for inspection")
            save_location_data(location_df, temp_file_path)
        else:
            logging.info("📊 Retreiving temporary file")
            location_df = pd.read_csv(temp_file_path)
        
        logging.info("📈 Sorting data by 'country'")
        location_df = location_df.sort_values(by=["country"])
        
        save_location_data(location_df, full_file_path)
        logging.info("🏁 Location data processing script completed successfully")
    except Exception as e:
        logging.error(f"❌ An error occurred: {e}")
        raise

full_file_path = os.path.join(LANDING_DIR, CLEAN_LOCATION_DATASET_NAME)

if __name__ == "__main__":
    main()
