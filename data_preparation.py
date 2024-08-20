import pandas as pd
from sklearn.model_selection import train_test_split
import os
import time
import psutil
import requests

def log_memory_usage(msg=None):
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 ** 2
    if msg:
        print(f"{msg}: {memory_usage:.2f} MB")
    else:
        print(f"Memory usage: {memory_usage:.2f} MB")

def download_dataset(url, file_name):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {file_name}")
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")

def prepare_astronomy_dataset(sample_size=5000, chunk_size=1000, save_interval=5000):
    start_time = time.time()
    log_memory_usage()

    print("Downloading and loading astronomy datasets...")

    # Exoplanet Archive Dataset
    exoplanet_url = "https://exoplanetarchive.ipac.caltech.edu/docs/data.html"
    exoplanet_file = "exoplanet_data.csv"
    download_dataset(exoplanet_url, exoplanet_file)

    # Load the dataset
    try:
        exoplanet_df = pd.read_csv(exoplanet_file)
    except pd.errors.ParserError:
        print("Error: Unable to parse Exoplanet Archive dataset. Skipping.")
        exoplanet_df = pd.DataFrame()

    # Replace the other datasets with valid datasets
    # For example, use Kaggle datasets (you need to have access to them and download manually):
    # NASA Astrophysics Data System (ADS), Astrobiology, and Astronomy Conversations datasets can be replaced with Kaggle datasets.

    # Combine all datasets
    astronomy_df = pd.concat([exoplanet_df], ignore_index=True)

    if len(astronomy_df) == 0:
        print("Astronomy datasets are empty. Skipping.")
        return None, None

    print("Sampling data...")
    astronomy_df = astronomy_df.sample(n=min(sample_size, len(astronomy_df)), random_state=42)

    train, val = train_test_split(astronomy_df, test_size=0.1, random_state=42)

    log_memory_usage("Final results")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

    return train, val

if __name__ == "__main__":
    print("Preparing astronomy datasets...")
    astronomy_train, astronomy_val = prepare_astronomy_dataset()
    if astronomy_train is not None and astronomy_val is not None:
        astronomy_train.to_csv('astronomy_train_data.csv', index=False)
        astronomy_val.to_csv('astronomy_val_data.csv', index=False)

    print("Process completed.")
