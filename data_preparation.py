import pandas as pd
from sklearn.model_selection import train_test_split
import os
import time
import psutil
from convokit import Corpus, download
from kaggle.api.kaggle_api_extended import KaggleApi

def log_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

def prepare_movie_corpus_dataset(sample_size=5000, chunk_size=1000, save_interval=5000):
    start_time = time.time()
    log_memory_usage()

    print("Downloading and loading movie corpus...")
    corpus = Corpus(filename=download("movie-corpus"))
    utterances_df = corpus.get_utterances_dataframe()
    
    print("Sampling data...")
    utterances_df = utterances_df.sample(n=min(sample_size, len(utterances_df)), random_state=42)
    
    train_chunks = []
    val_chunks = []
    
    print("Processing data in chunks...")
    for i, chunk in enumerate(pd.read_csv(utterances_df, chunksize=chunk_size)):
        chunk = chunk[chunk['text'].str.len().gt(50)]
        train, val = train_test_split(chunk, test_size=0.1, random_state=42)
        train_chunks.append(train)
        val_chunks.append(val)
        
        if (i + 1) * chunk_size % save_interval == 0:
            print(f"Saving intermediate results (chunk {i})...")
            pd.concat(train_chunks).to_csv(f'movie_train_data_chunk_{i}.csv', index=False)
            pd.concat(val_chunks).to_csv(f'movie_val_data_chunk_{i}.csv', index=False)
        
        log_memory_usage()
    
    print("Concatenating final results...")
    train_data = pd.concat(train_chunks)
    val_data = pd.concat(val_chunks)
    
    log_memory_usage()
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
    return train_data, val_data

def prepare_astronomy_dataset(sample_size=5000, chunk_size=1000, save_interval=5000):
    start_time = time.time()
    log_memory_usage()

    print("Downloading and loading astronomy dataset...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file('kartashevaks/astronomy-dataset', file_name='astronomy_dataset.csv', path='.')
    astronomy_df = pd.read_csv("astronomy_dataset.csv")
    os.remove("astronomy_dataset.csv")
    
    print("Sampling data...")
    astronomy_df = astronomy_df.sample(n=min(sample_size, len(astronomy_df)), random_state=42)
    
    train_chunks = []
    val_chunks = []
    
    print("Processing data in chunks...")
    for i, chunk in enumerate(pd.read_csv(astronomy_df, chunksize=chunk_size)):
        train, val = train_test_split(chunk, test_size=0.1, random_state=42)
        train_chunks.append(train)
        val_chunks.append(val)
        
        if (i + 1) * chunk_size % save_interval == 0:
            print(f"Saving intermediate results (chunk {i})...")
            pd.concat(train_chunks).to_csv(f'astronomy_train_data_chunk_{i}.csv', index=False)
            pd.concat(val_chunks).to_csv(f'astronomy_val_data_chunk_{i}.csv', index=False)
        
        log_memory_usage()
    
    print("Concatenating final results...")
    train_data = pd.concat(train_chunks)
    val_data = pd.concat(val_chunks)
    
    log_memory_usage()
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
    return train_data, val_data

def prepare_astrology_dataset(sample_size=5000, chunk_size=1000, save_interval=5000):
    start_time = time.time()
    log_memory_usage()

    print("Downloading and loading astrology dataset...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file('divyansh22/astrology-dataset', file_name='astrology_dataset.csv', path='.')
    astrology_df = pd.read_csv("astrology_dataset.csv")
    os.remove("astrology_dataset.csv")
    
    print("Sampling data...")
    astrology_df = astrology_df.sample(n=min(sample_size, len(astrology_df)), random_state=42)
    
    train_chunks = []
    val_chunks = []
    
    print("Processing data in chunks...")
    for i, chunk in enumerate(pd.read_csv(astrology_df, chunksize=chunk_size)):
        train, val = train_test_split(chunk, test_size=0.1, random_state=42)
        train_chunks.append(train)
        val_chunks.append(val)
        
        if (i + 1) * chunk_size % save_interval == 0:
            print(f"Saving intermediate results (chunk {i})...")
            pd.concat(train_chunks).to_csv(f'astrology_train_data_chunk_{i}.csv', index=False)
            pd.concat(val_chunks).to_csv(f'astrology_val_data_chunk_{i}.csv', index=False)
        
        log_memory_usage()
    
    print("Concatenating final results...")
    train_data = pd.concat(train_chunks)
    val_data = pd.concat(val_chunks)
    
    log_memory_usage()
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
    return train_data, val_data

if __name__ == "__main__":
    print("Preparing movie corpus dataset...")
    movie_train, movie_val = prepare_movie_corpus_dataset()
    movie_train.to_csv('movie_train_data.csv', index=False)
    movie_val.to_csv('movie_val_data.csv', index=False)
    
    print("Preparing astronomy dataset...")
    astronomy_train, astronomy_val = prepare_astronomy_dataset()
    astronomy_train.to_csv('astronomy_train_data.csv', index=False)
    astronomy_val.to_csv('astronomy_val_data.csv', index=False)
    
    print("Preparing astrology dataset...")
    astrology_train, astrology_val = prepare_astrology_dataset()
    astrology_train.to_csv('astrology_train_data.csv', index=False)
    astrology_val.to_csv('astrology_val_data.csv', index=False)
    
    print("Process completed.")
