import requests
import pandas as pd
import os
import html

API_URL = "http://localhost:8000/api/scholarship"
SAVE_PATH = "data/cleaned_data.csv"

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)  # decode HTML entities
    return text.replace("\n", " ").replace("\r", " ").strip()

def fetch_and_clean_data():
    print("Fetching data from API...")
    response = requests.get(API_URL)
    response.raise_for_status()
    data = response.json()

    # Check for list or dictionary structure
    if isinstance(data, dict) and "data" in data:
        records = data["data"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Unexpected API response format.")

    print(f"Fetched {len(records)} records")

    # Normalize JSON
    df = pd.json_normalize(records)

    # Clean specific columns if they exist
    for column in ['description', 'title']:
        if column in df.columns:
            df[column] = df[column].apply(clean_text)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Save as UTF-8 CSV with proper quoting
    df.to_csv(SAVE_PATH, index=False, encoding='utf-8', quoting=1)
    print(f"✅ Cleaned data saved to {SAVE_PATH}")

if __name__ == "__main__":
    fetch_and_clean_data()
