import pandas as pd
from sklearn.model_selection import train_test_split
import os
import time
import psutil
from kaggle.api.kaggle_api_extended import KaggleApi

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

    print("Downloading and loading astronomy dataset...")
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Error authenticating with Kaggle API: {e}")
        return None, None
    api.dataset_download_file('kartashevaks/astronomy-dataset', file_name='astronomy_dataset.csv', path='.')
    astronomy_df = pd.read_csv("astronomy_dataset.csv")
    os.remove("astronomy_dataset.csv")
    
    if len(astronomy_df) == 0:
        print("Astronomy dataset is empty. Skipping.")
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

def prepare_astrology_dataset(sample_size=5000, chunk_size=1000, save_interval=5000):
    start_time = time.time()
    log_memory_usage()

    print("Downloading and loading astrology dataset...")
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Error authenticating with Kaggle API: {e}")
        return None, None
    api.dataset_download_file('divyansh22/astrology-dataset', file_name='astrology_dataset.csv', path='.')
    astrology_df = pd.read_csv("astrology_dataset.csv")
    os.remove("astrology_dataset.csv")
    
    if len(astrology_df) == 0:
        print("Astrology dataset is empty. Skipping.")
        return None, None
    
    print("Sampling data...")
    astrology_df = astrology_df.sample(n=min(sample_size, len(astrology_df)), random_state=42)
    
    train_chunks = []
    val_chunks = []
    
    print("Processing data in chunks...")
    num_chunks = len(astrology_df) // chunk_size + 1
    for i, chunk in enumerate(pd.read_csv(astrology_df, chunksize=chunk_size)):
        train, val = train_test_split(chunk, test_size=0.1, random_state=42)
        train_chunks.append(train)
        val_chunks.append(val)
        
        if (i + 1) * chunk_size % save_interval == 0:
            print(f"Saving intermediate results (chunk {i+1} of {num_chunks})...")
            pd.concat(train_chunks).to_csv(f'astrology_train_data_chunk_{i+1}.csv', index=False)
            pd.concat(val_chunks).to_csv(f'astrology_val_data_chunk_{i+1}.csv', index=False)
        
        log_memory_usage(f"Processing chunk {i+1} of {num_chunks}")
    
    print("Concatenating final results...")
    train_data = pd.concat(train_chunks)
    val_data = pd.concat(val_chunks)
    
    log_memory_usage("Final results")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
    return train_data, val_data

if __name__ == "__main__":
    print("Preparing astronomy dataset...")
    astronomy_train, astronomy_val = prepare_astronomy_dataset()
    if astronomy_train is not None and astronomy_val is not None:
        astronomy_train.to_csv('astronomy_train_data.csv', index=False)
        astronomy_val.to_csv('astronomy_val_data.csv', index=False)
    
    print("Preparing astrology dataset...")
    astrology_train, astrology_val = prepare_astrology_dataset()
    if astrology_train is not None and astrology_val is not None:
        astrology_train.to_csv('astrology_train_data.csv', index=False)
        astrology_val.to_csv('astrology_val_data.csv', index=False)
    
    print("Process completed.")
