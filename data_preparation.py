import pandas as pd
from sklearn.model_selection import train_test_split
import os
import time
import psutil
import requests
from io import BytesIO
from zipfile import ZipFile

def log_memory_usage(msg=None):
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 ** 2
    if msg:
        print(f"{msg}: {memory_usage:.2f} MB")
    else:
        print(f"Memory usage: {memory_usage:.2f} MB")

def prepare_astronomy_dataset(sample_size=5000, chunk_size=1000, save_interval=5000):
    start_time = time.time()
    log_memory_usage()

    print("Downloading and loading astronomy datasets...")
    
    # Exoplanet Archive
    print("Downloading Exoplanet Archive dataset...")
    exoplanet_url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&format=csv"
    exoplanet_df = pd.read_csv(exoplanet_url)
    
    # Astrophysics Data System
    print("Downloading Astrophysics Data System dataset...")
    ads_url = "https://ui.adsabs.harvard.edu/api/v1/search/query?q=*&fl=id,title,author,date,citation_count,abstract&rows=10000&start=0"
    try:
        ads_response = requests.get(ads_url)
        ads_data = ads_response.json()
        ads_df = pd.DataFrame(ads_data["response"]["docs"])
    except requests.exceptions.JSONDecodeError:
        print("Error: Unable to parse ADS dataset. Skipping.")
        ads_df = pd.DataFrame()
    
    # Planetary Data System
    print("Downloading Planetary Data System dataset...")
    pds_url = "https://pds.nasa.gov/datasearch/metadata-service/datasetlist.jsp?category=all&page=1&sortcol=1&sort=asc&format=csv"
    pds_df = pd.read_csv(pds_url)
    
    # HEASARC
    print("Downloading HEASARC dataset...")
    heasarc_url = "https://heasarc.gsfc.nasa.gov/FTP/heasarc/dataseta.txt"
    heasarc_df = pd.read_csv(heasarc_url, delimiter="\t")
    
    # Combine all datasets
    astronomy_df = pd.concat([exoplanet_df, ads_df, pds_df, heasarc_df], ignore_index=True)
    
    if len(astronomy_df) == 0:
        print("Astronomy datasets are empty. Skipping.")
        return None, None
    
    print("Sampling data...")
    astronomy_df = astronomy_df.sample(n=min(sample_size, len(astronomy_df)), random_state=42)
    
    train_chunks = []
    val_chunks = []
    
    print("Processing data in chunks...")
    num_chunks = len(astronomy_df) // chunk_size + 1
    for i, chunk in enumerate(pd.read_csv(astronomy_df, chunksize=chunk_size)):
        train, val = train_test_split(chunk, test_size=0.1, random_state=42)
        train_chunks.append(train)
        val_chunks.append(val)
        
        if (i + 1) * chunk_size % save_interval == 0:
            print(f"Saving intermediate results (chunk {i+1} of {num_chunks})...")
            pd.concat(train_chunks).to_csv(f'astronomy_train_data_chunk_{i+1}.csv', index=False)
            pd.concat(val_chunks).to_csv(f'astronomy_val_data_chunk_{i+1}.csv', index=False)
        
        log_memory_usage(f"Processing chunk {i+1} of {num_chunks}")
    
    print("Concatenating final results...")
    train_data = pd.concat(train_chunks)
    val_data = pd.concat(val_chunks)
    
    log_memory_usage("Final results")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
    return train_data, val_data

if __name__ == "__main__":
    print("Preparing astronomy datasets...")
    astronomy_train, astronomy_val = prepare_astronomy_dataset()
    if astronomy_train is not None and astronomy_val is not None:
        astronomy_train.to_csv('astronomy_train_data.csv', index=False)
        astronomy_val.to_csv('astronomy_val_data.csv', index=False)
    
    print("Process completed.")
